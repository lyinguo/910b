"""Exact own Ferret/Gemma1 option teacher forcing on Ascend; no downloads.

All learned code and multimodal packing come from the own unmodified checkpoint.
The original language eager/no-cache graph is retained. Vision is recomputed for
every option as in the frozen 525-forward original; observers never feed values
back into the model. CPU/FP32 controls are diagnostics and never official scores.
"""
import contextlib,datetime,hashlib,inspect,json,logging,math,os,pathlib,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch,torch_npu
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.ferret_original_io_v3_0106 import WEIGHTS,MODEL_ID,CLIP_CONFIG,CLIP_SHA,sha,fp,record,blueprint,processor_and_tokenizer,prepare,target_key
log=logging.getLogger('universal_eval.model.adapters.ferret_original_option_0106')

class FerretOriginalOptionV40106Adapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.protocol_file=protocol_file;self.protocol_sha256=protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());assert self.p['model_id']==MODEL_ID
  self.devices=list(npu_devices or [0]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('Ferret_original_option_0106_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
  self.trace=self.evidence/'sample_execution.jsonl';self.candidate_trace=self.evidence/'candidate_execution.jsonl';self.activation_trace=self.evidence/'activation_execution.jsonl'
  for name in ['candidate_logits','visual_features']:(self.evidence/name).mkdir()
  assert sha(self.p['records_path'])==self.p['records_sha256'];self.cpu=json.loads(pathlib.Path(self.p['records_path']).read_text());assert len(self.cpu)==200
 @property
 def model_name(self):return MODEL_ID
 @property
 def model_type(self):return 'FerretGemmaForCausalLM'
 @property
 def inference_framework(self):return 'Native own Ferret/Gemma1 BF16 eager option teacher forcing, original native mean CE, torch_npu'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from safetensors import safe_open
  from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
  assert pathlib.Path(model_path).resolve()==WEIGHTS and self.p['original_runtime_dtype']=='BF16' and self.p['candidate_count']==525 and self.p['use_cache'] is False
  for rel,digest in self.p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(WEIGHTS/rel)==digest
  for path,digest in dict(self.p['historical_sources'],**self.p['native_source_sha256']).items():assert sha(path)==digest
  assert sha(CLIP_CONFIG)==CLIP_SHA and torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for k in ['conv','matmul']:getattr(torch.npu,k).allow_hf32=False
  self.model,self.modules=blueprint();self.model.to(dtype=torch.bfloat16);targets=dict(self.model.named_parameters(remove_duplicate=False));expected={r['source_key']:r for r in self.p['source_tensor_records']};filled={};records=[]
  for shard,review in self.p['weight_headers'].items():
   q=WEIGHTS/shard;assert q.stat().st_size==review['size'] and sha(q)==review['file_sha256']
   with safe_open(q,framework='pt',device='cpu') as f:
    for key in f.keys():
     r=expected[key];name=target_key(key,targets);value=f.get_tensor(key);assert name==r['target_key'] and list(value.shape)==r['shape'] and str(value.dtype)=='torch.bfloat16' and fp(value)==r['source_sha256']==r['original_BF16_cast_sha256']
     actual=value.to(self.device);assert fp(actual)==r['source_sha256'];parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(actual,requires_grad=False));filled[name]=r['source_sha256'];records.append(dict(r,actual_NPU_sha256=fp(actual),device=str(actual.device)))
   log.info('Ferret loaded own %s: %s/576',shard,len(records))
  assert len(records)==576 and set(filled)==set(targets)
  for group in self.p['native_alias_groups']:
   assert len({filled[n] for n in group})==1;parent,leaf=group[0].rsplit('.',1);parameter=getattr(self.model.get_submodule(parent),leaf)
   for n in group:
    parent,leaf=n.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,parameter)
  assert self.model.lm_head.weight is self.model.model.embed_tokens.weight
  for n,digest in filled.items():parent,leaf=n.rsplit('.',1);assert fp(getattr(self.model.get_submodule(parent),leaf))==digest
  self.model.model.rotary_emb=GemmaRotaryEmbedding(self.model.config).to(self.device)
  self.model.model.embed_tokens.embed_scale=torch.tensor(math.sqrt(self.model.config.hidden_size),dtype=torch.bfloat16,device=self.device)
  for n,b in list(self.model.named_buffers()):
   if b.is_meta:
    assert n.endswith('embeddings.position_ids') and list(b.shape)==[1,577];parent,leaf=n.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.arange(577,dtype=torch.int64,device=self.device).expand(1,-1))
   elif b.device!=self.device:
    parent,leaf=n.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,b.to(self.device))
  assert all(p.device==self.device and p.dtype==torch.bfloat16 for p in self.model.parameters()) and all(not b.is_meta and b.device==self.device for b in self.model.buffers())
  assert self.model.config._attn_implementation=='eager' and self.model.get_vision_tower().vision_tower.config._attn_implementation==self.p['native_vision_attention']
  self.model.eval();self.processor,self.tokenizer,self.modules=processor_and_tokenizer();self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['GELU','GELUActivation','GELUTanh','PytorchGELUTanh','QuickGELUActivation','ReLU']}
  self._reset();self._observe();loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2))
  probes={};values=torch.linspace(-8,8,20001).bfloat16();arrays={'input_BF16':values.view(torch.uint8).numpy()}
  for approximation in ['none','tanh']:
   cpu=torch.nn.functional.gelu(values,approximate=approximation);npu=torch.nn.functional.gelu(values.to(self.device),approximate=approximation).cpu();arrays['CPU_'+approximation]=cpu.view(torch.uint8).numpy();arrays['NPU_'+approximation]=npu.view(torch.uint8).numpy();probes[approximation]=float((cpu-npu).abs().max())
  probe=self.evidence/'native_GELU_probe.npz';np.savez_compressed(probe,**arrays)
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'original_Ferret_option_teacher_forced_mean_CE','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'own_source_tensor_count':576,'independent_learned_parameter_count':575,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(loading),'per_sample_evidence':str(self.trace),'per_candidate_evidence':str(self.candidate_trace),'activation_execution_trace':str(self.activation_trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':{'path':str(probe),'sha256':sha(probe),'CPU_vs_NPU_max_abs':probes},'native_attention_implementation':{'text':'eager','vision':self.p['native_vision_attention']},'original_use_cache':False,'actual_LM_batch_size':1,'original_option_continuation':'{letter}. {text}<end_of_turn>','full_vocab_size':256001,'activation_modules':self.activation_modules,'original_own_tied_embedding_head_preserved':True,'native_nonlearned_buffers':{n:record(v) for n,v in self.model.named_buffers()},'declared_CLIP_config':str(CLIP_CONFIG),'declared_CLIP_config_sha256':CLIP_SHA,'external_CLIP_learned_values_loaded':False,'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'precision_disclosure':'Own576 original BF16 checkpoint tensors including equal tied embedding/head and all391 ownCLIP tensors preserve complete source/cast/NPU SHA. No external learned values or quantization. Own native Gemma1 eager/no-cache/full-position full-vocab forward with original native FP32 continuation mean CE; original 2048 fallback budget remains metadata, all200 matched inputs use original option branch. Each candidate really recomputes original anyres visual encoder/projector; observed visual equality never feeds cached values into official computation. Region sampler original16 learned tensors retained on NPU but region_masks=None branch never executes them. CANN native numerical differences retained; CPU/FP32 controls diagnostic only.'}
 def _reset(self):
  self.arrays={};self.counts={n:0 for n in getattr(self,'activation_modules',{})};self.attention=[];self.vision_calls=self.projector_calls=self.text_calls=self.head_calls=0
 def _save(self,name,value):assert value.device==self.device;self.arrays[name]=value.detach().cpu().contiguous().clone()
 def _observe(self):
  vt=self.model.get_vision_tower();text=self.model.model
  def visual(module,args,out):self._save('selected_original_CLIP_penultimate_patches',out);self.vision_calls+=1
  def projection(module,args,out):self._projected=out;self._save('actual_projected_patches',out);self.projector_calls+=1
  def language(module,args,kw):
   x=kw['inputs_embeds'];assert x is not None and kw.get('past_key_values') is None and kw.get('use_cache') is False
   p=self._projected;gw,gh=self.modules['mm_utils'].get_anyres_image_grid_shape(self._image_size,self.model.config.image_grid_pinpoints,336)
   expected=torch.cat([p[0],p[1:].reshape(gh,gw,24,24,2048).permute(0,2,1,3,4).reshape(-1,2048)],dim=0)
   index=self._ids[0].tolist().index(-200);labels=self._labels[0];embeds=text.embed_tokens(self._ids[self._ids!=-200]);before=embeds[:index];after=embeds[index:];joined=torch.cat([before,expected,after],dim=0)[:4096]
   assert torch.equal(x[0],joined);self._merged_labels=torch.cat([labels[:index],torch.full((expected.shape[0],),-100,dtype=labels.dtype,device=self.device),labels[index+1:]])[:4096]
   self._save('actual_injected_visual',x[0,index:min(index+len(expected),4096)]);self._text_input_record=record(x);self.text_calls+=1
  def rope(module,args,kw,out):
   pos=kw.get('position_ids',args[1] if len(args)>1 else None);self._save('actual_positions',pos);self._save('actual_cos',out[0]);self._save('actual_sin',out[1])
  def attention(index):
   def hook(module,args,kw):
    mask=kw.get('attention_mask');assert mask is not None and kw.get('past_key_values') is None
    if not self.attention:self._mask=mask;self._mask_record=record(mask)
    else:assert torch.equal(mask,self._mask)
    self.attention.append({'layer':index,'mask':self._mask_record,'cache':None,'scaling':module.scaling})
   return hook
  def head(module,args,out):assert out.ndim==3 and out.shape[0]==1 and out.shape[-1]==256001;self._head_shape=list(out.shape);self.head_calls+=1
  vt.register_forward_hook(visual);self.model.model.mm_projector.register_forward_hook(projection);text.register_forward_pre_hook(language,with_kwargs=True);text.rotary_emb.register_forward_hook(rope,with_kwargs=True);self.model.lm_head.register_forward_hook(head)
  for i,layer in enumerate(text.layers):layer.self_attn.register_forward_pre_hook(attention(i),with_kwargs=True)
  for name,m in self.model.named_modules():
   if name in self.activation_modules:
    def activation(module,args,out,name=name):assert out.device==self.device;self.counts[name]+=1
    m.register_forward_hook(activation)
 def predict(self,prompt,media_path=None):
  self.calls+=1;start=time.monotonic();entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None};raw=None
  try:
   a=prepare(prompt,media_path,self.processor,self.tokenizer,self.modules);r=self.cpu[self.calls-1];assert a['clean'].strip()==r['input'].strip();assert sha(media_path)==r['media_sha256'] and record(a['pixel_values'])==r['pixels'] and a['base_prompt']==r['base_prompt'] and a['candidates']==r['candidates']
   images=a['pixel_values'].unsqueeze(0).to(device=self.device,dtype=torch.bfloat16);self._image_size=a['image_size'];scores={};candidate_records=[];visual_reference=None
   for j,candidate in enumerate(a['candidates']):
    self._reset();self._ids=torch.tensor(candidate['input_ids'],dtype=torch.long,device=self.device).unsqueeze(0);self._labels=torch.tensor(candidate['labels'],dtype=torch.long,device=self.device).unsqueeze(0)
    with torch.inference_mode():
     call=lambda:self.model(self._ids,attention_mask=None,labels=self._labels,images=images,image_sizes=[a['image_size']],region_masks=None,use_cache=False)
     if not self.profile_done:
      with contextlib.chdir(self.evidence):
       with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:out=call();torch.npu.synchronize(self.device)
       prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
      self.profile_done=True
     else:out=call()
    assert out.loss is not None and torch.isfinite(out.loss) and out.past_key_values is None and torch.isfinite(out.logits).all() and list(out.logits.shape)==self._head_shape and len(self.attention)==18 and self.vision_calls==self.projector_calls==self.text_calls==self.head_calls==1
    expected_active={n:1 for n in self.counts if '.region_geo_sampler.' not in n};assert all(self.counts[n]==v for n,v in expected_active.items()) and all(v==0 for n,v in self.counts.items() if '.region_geo_sampler.' in n)
    target=self._merged_labels[1:];keep=target!=-100;selected=out.logits[0,:-1][keep].float();labels=target[keep];assert len(labels)>0
    scores[candidate['letter']]=-float(out.loss.float().item());arrays={'full_vocab_scored_logits_FP32':selected.detach().cpu().numpy(),'continuation_labels':labels.cpu().numpy(),'merged_labels':self._merged_labels.cpu().numpy(),**{k:v.view(torch.uint8).numpy() for k,v in self.arrays.items() if k in ['actual_positions','actual_cos','actual_sin']}}
    archive=self.evidence/'candidate_logits'/('%06d_%s.npz'%(r['index'],candidate['letter']));np.savez_compressed(archive,**arrays)
    visual={k:v for k,v in self.arrays.items() if k not in ['actual_positions','actual_cos','actual_sin']}
    if visual_reference is None:
     visual_reference=visual;varchive=self.evidence/'visual_features'/('%06d.npz'%r['index']);np.savez_compressed(varchive,**{k:v.view(torch.uint8).numpy() for k,v in visual.items()})
    else:assert set(visual)==set(visual_reference) and all(torch.equal(v,visual_reference[k]) for k,v in visual.items())
    ce={'attempt':self.calls,'index':r['index'],'letter':candidate['letter'],'original_continuation':candidate['continuation'],'input_ids':candidate['input_ids'],'labels_before_original_multimodal_packing':candidate['labels'],'score':scores[candidate['letter']],'actual_native_loss':float(out.loss.float()),'native_loss_dtype':str(out.loss.dtype),'actual_full_logits_dtype':str(out.logits.dtype),'actual_full_logits_shape':list(out.logits.shape),'actual_text_input':self._text_input_record,'actual_visual_recomputed':True,'actual_visual_injection_bitwise_verified':True,'actual_vision_calls':self.vision_calls,'actual_attention_layers':self.attention,'tensor_records':{k:record(v) for k,v in self.arrays.items()},'scored_logits_path':str(archive),'scored_logits_sha256':sha(archive),'visual_path':str(varchive),'visual_sha256':sha(varchive),'activation_call_counts':dict(self.counts),'cache':None,'error':None}
    candidate_records.append(ce)
    with self.candidate_trace.open('a') as f:f.write(json.dumps(ce,ensure_ascii=False)+'\n')
    with self.activation_trace.open('a') as f:f.write(json.dumps({'attempt':self.calls,'index':r['index'],'letter':candidate['letter'],'module_call_counts':dict(self.counts),'device':str(self.device),'error':None})+'\n')
    del out,selected,arrays;self._reset()
   best=max(scores,key=scores.get);raw=f'<answer>{best}</answer>';entry.update(index=r['index'],raw_output=raw,candidate_scores=scores,candidates=candidate_records,pixel_values=r['pixels'],source_image_sha256=r['media_sha256'],original_image_size=a['image_size'],original_anyres_tile_count=len(a['pixel_values']),empty=False,invalid=False,truncated=False,generation_count=0,generation_token_ids=[],no_generation=True,original_2048_generation_fallback_not_executed=True)
   log.info('Ferret original option NPU sample %s index=%s selected=%s scores=%s',self.calls,r['index'],best,scores);return raw
  except Exception:entry['error']=traceback.format_exc();log.error(entry['error']);raise
  finally:
   entry['elapsed_seconds']=time.monotonic()-start;clean=self.modules['mm_utils'] if hasattr(self,'modules') else None
   from universal_eval.model.adapters.ferret_original_io_v3_0106 import parse_options
   self.details[(parse_options(prompt)[0].strip(),media_path)]=entry
   with self.trace.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
 def batch_predict(self,batch):return [self.predict(p,m) for p,m in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

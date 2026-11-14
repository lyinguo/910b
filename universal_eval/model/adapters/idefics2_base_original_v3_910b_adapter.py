"""Exact original Idefics2-base first-option-token scoring on Ascend NPU."""
import os,pathlib,json,hashlib,datetime,logging,time,traceback,contextlib,types
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.idefics2_base_original_910b')
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}

def compatible_vision_embedding(self,pixel_values,patch_attention_mask):
 """Only SearchSorted receives FP32 representations of already BF16-rounded coords."""
 batch,_,h,w=pixel_values.shape
 embeddings=self.patch_embedding(pixel_values).flatten(2).transpose(1,2)
 boundaries=torch.arange(1/self.num_patches_per_side,1.,1/self.num_patches_per_side,device=pixel_values.device)
 positions=torch.full((batch,(h//self.patch_size)*(w//self.patch_size)),0,device=pixel_values.device)
 nh=patch_attention_mask[:,:,0].sum(1);nw=patch_attention_mask[:,0,:].sum(1)
 hs=(torch.arange(patch_attention_mask.size(1),device=pixel_values.device,dtype=torch.float32)[None,:]*(1./nh)[:,None]).clamp(max=1.-1e-6).to(pixel_values.dtype)
 ws=(torch.arange(patch_attention_mask.size(2),device=pixel_values.device,dtype=torch.float32)[None,:]*(1./nw)[:,None]).clamp(max=1.-1e-6).to(pixel_values.dtype)
 hb=torch.bucketize(hs.float(),boundaries,right=True);wb=torch.bucketize(ws.float(),boundaries,right=True)
 values=(hb[:,:,None]*self.num_patches_per_side+wb[:,None,:]).reshape(batch,-1);mask=patch_attention_mask.reshape(batch,-1);positions[mask]=values[mask]
 return embeddings+self.position_embedding(positions)

class Idefics2BaseOriginalV3_910BAdapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256
  self.devices=list(npu_devices or [5]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('Idefics2_base_original_score_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
  (self.evidence/'full_vocab_and_visual').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.activation_trace=self.evidence/'native_activation_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Idefics2ForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers/torch_npu original BF16 first-option-token mc_score'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoProcessor,Idefics2ForConditionalGeneration
  from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name=='HuggingFaceM4/idefics2-8b-base'
  for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
  for src,digest in dict(p['runtime_sha256'],**p['historical_sources']).items():assert sha(src)==digest
  for key in ['reference_report','reference_log','dataset_path','original_dataset','historical_transformers_before_run']:assert sha(p[key])==p[key+'_sha256']
  assert p['source_dtype']=='F32' and p['runtime_dtype']=='BF16' and p['original_algorithm']=='mc_score' and p['original_algorithm_batch_size']==8 and p['actual_inference_batch_size']==1 and p['model_vocab_size']==32002 and p['image_splitting'] is False
  assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.tie_word_embeddings is False and cfg.text_config.use_cache is True and cfg.text_config.vocab_size==32002 and cfg.vision_config.num_hidden_layers==27
  for c in [cfg,cfg.text_config,cfg.vision_config]:c._attn_implementation='eager'
  with torch.device('meta'):self.model=Idefics2ForConditionalGeneration(cfg)
  parameters=dict(self.model.named_parameters());assert len(parameters)==763;expected={r['source_key']:r for r in p['source_tensor_records']};assert set(expected)==set(parameters)
  records=[];seen=set()
  for shard,header in p['weight_headers'].items():
   assert pathlib.Path(shard).name==shard and (wp/shard).stat().st_size==header['size'] and sha(wp/shard)==header['file_sha256']
   with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
    for key in f.keys():
     assert key not in seen and key in parameters;source=f.get_tensor(key);cast=source.bfloat16();r=expected[key];assert source.dtype==torch.float32 and list(source.shape)==r['shape']==list(parameters[key].shape) and fp(source)==r['source_sha256'] and fp(cast)==r['original_BF16_cast_sha256'] and torch.isfinite(cast).all()
     target=cast.to(self.device);actual=fp(target);assert actual==r['original_BF16_cast_sha256'];parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));seen.add(key);records.append(dict(r,device=str(target.device),actual_NPU_sha256=actual))
   log.info('Idefics2 original source loaded %s %s/763',shard,len(seen))
  assert seen==set(parameters);self.model.model.text_model.rotary_emb=MistralRotaryEmbedding(cfg.text_config,device='cpu').to(self.device)
  assert self.model.lm_head.weight is not self.model.model.text_model.embed_tokens.weight and fp(self.model.lm_head.weight)!=fp(self.model.model.text_model.embed_tokens.weight)
  assert all(x.device==self.device and x.dtype==torch.bfloat16 for x in self.model.parameters()) and all(not x.is_meta and x.device==self.device for x in self.model.buffers())
  self.model.eval();self.processor=AutoProcessor.from_pretrained(wp,local_files_only=True);self.processor.tokenizer.padding_side='left';self.tokenizer=self.processor.tokenizer
  assert type(self.processor.image_processor).__name__=='Idefics2ImageProcessor' and json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['image_processor'] and not self.processor.image_processor.do_image_splitting and self.processor.image_seq_len==64 and len(self.tokenizer)==p['tokenizer_size_after_own_processor']
  self.model.model.vision_model.embeddings.forward=types.MethodType(compatible_vision_embedding,self.model.model.vision_model.embeddings)
  self.activation_names={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['GELUActivation','PytorchGELUTanh','NewGELUActivation','QuickGELUActivation','SiLU','GELUTanh','SiLUActivation']}
  assert len(self.activation_names)==63 and sum(t=='GELUTanh' for t in self.activation_names.values())==27 and sum(t=='SiLUActivation' for t in self.activation_names.values())==36
  self._reset()
  def vision_pre(module,args,kw):
   self._actual_pixels=kw['pixel_values'];self._patch_mask=kw['patch_attention_mask'];assert self._actual_pixels.device==self.device and self._actual_pixels.dtype==torch.bfloat16
  def vision(module,args,out):self._vision=out.last_hidden_state;self._vision_calls+=1
  def connector(module,args,out):self._features=out;self._connector_calls+=1
  def positions(module,args):self._vision_positions=args[0].detach().clone();assert self._vision_positions.device==self.device
  def text_input(module,args,kw):
   x=kw['inputs_embeds'];assert x.device==self.device and self._features is not None
   injected=x[self._active_ids==cfg.image_token_id];assert injected.shape==(64,4096) and torch.equal(injected,self._features.reshape(-1,4096))
   originals=module.embed_tokens(self._active_ids);mask=self._active_ids!=cfg.image_token_id;assert torch.equal(x[mask],originals[mask]);self._merged=x.detach().clone();self._injected=injected.detach().clone();self._text_calls+=1
  def rope(module,args,kw,out):
   ids=kw.get('position_ids',args[1] if len(args)>1 else None);assert ids is not None;self._text_positions=ids.detach().clone();self._cos=out[0].detach().clone();self._sin=out[1].detach().clone()
  def attn(module,args,kw):
   mask=kw.get('attention_mask');assert mask is not None and mask.device==self.device;self._text_mask=mask.detach().clone()
  def activation(name):
   def hook(module,args,out):assert out.device==self.device and out.dtype==torch.bfloat16;self._activation_counts[name]+=1
   return hook
  self.model.model.vision_model.register_forward_pre_hook(vision_pre,with_kwargs=True);self.model.model.vision_model.register_forward_hook(vision);self.model.model.connector.register_forward_hook(connector);self.model.model.vision_model.embeddings.position_embedding.register_forward_pre_hook(positions);self.model.model.text_model.register_forward_pre_hook(text_input,with_kwargs=True);self.model.model.text_model.rotary_emb.register_forward_hook(rope,with_kwargs=True);self.model.model.text_model.layers[0].self_attn.register_forward_pre_hook(attn,with_kwargs=True)
  for n,m in self.model.named_modules():
   if n in self.activation_names:m.register_forward_hook(activation(n))
  values=torch.linspace(-8,8,20001,dtype=torch.float32).bfloat16();probe_arrays={'input_BF16_bytes':values.view(torch.uint8).numpy()}
  self.probe={}
  for approximate in ['none','tanh']:
   npu=torch.nn.functional.gelu(values.to(self.device),approximate=approximate).cpu();cpu=torch.nn.functional.gelu(values,approximate=approximate);probe_arrays['NPU_'+approximate+'_BF16_bytes']=npu.view(torch.uint8).numpy();probe_arrays['CPU_'+approximate+'_BF16_bytes']=cpu.view(torch.uint8).numpy();self.probe[approximate]={'max_abs_CPU_same_declared_GELU':float((npu-cpu).abs().max())}
  probe=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(probe,**probe_arrays);self.probe.update(path=str(probe),sha256=sha(probe),device=str(self.device),CPU_threads=torch.get_num_threads(),diagnostic_only=True)
  loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir();(snapshot/'original_protocol_review.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
  for src in set(p['runtime_sha256'])|set(p['historical_sources'])|{p['historical_transformers_before_run']}:
   source=pathlib.Path(src);dest=snapshot/source.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'idefics2_base_original_first_letter_BF16_log_softmax','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'source_dtype':'F32','runtime_dtype':'BF16','weight_tensor_count':763,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(loading),'per_sample_evidence':str(self.trace),'activation_execution_trace':str(self.activation_trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':self.probe,'activation_modules':self.activation_names,'independent_lm_head_preserved':True,'source_underflow_total':p['source_underflow_total'],'source_BF16_roundtrip_changed_elements':p['source_BF16_roundtrip_changed_elements'],'native_attention_implementation':'eager','original_cache':True,'original_algorithm_batch_size':8,'actual_inference_batch_size':1,'image_splitting':False,'image_seq_len':64,'model_vocab_size':32002,'tokenizer_size_after_own_processor':len(self.tokenizer),'image_processor':self.processor.image_processor.to_dict(),'text_inv_freq':tr(self.model.model.text_model.rotary_emb.inv_freq),'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'precision_disclosure':'Every one of the own763 sourceF32 tensors is converted by the original explicit CPU BF16 cast, with full source/cast/NPU SHA and original rounding/underflow retained. All learned vision/perceiver/Mistral/head layers run on real NPU BF16; HF32off. Native eager attention, original cacheTrue and original raw BF16 full-vocabulary log_softmax are unchanged. The only vision indexing compatibility represents already BF16-rounded coordinates exactly as FP32 for CANN SearchSorted; learned BF16 arithmetic is not replaced. Original serial candidate scoring within algorithm batches8: one full forward per sample, first token of each candidate letter, original insertion-order tie-break. No free generation, option-text averaging, forced vocabulary mask, CPU official outputs or claim of low-bit acceleration/CPU-NPU bitwise agreement. Own processor may register its own end-of-utterance token beyond base head vocab32002; no resizing/masked logits or fabricated resource, and every actual input/candidate ID must remain within the unchanged base vocabulary.'}
 def _reset(self):
  self._vision=None;self._features=None;self._merged=None;self._injected=None;self._actual_pixels=None;self._patch_mask=None;self._vision_positions=None;self._text_positions=None;self._cos=None;self._sin=None;self._text_mask=None;self._vision_calls=0;self._connector_calls=0;self._text_calls=0;self._activation_counts={n:0 for n in getattr(self,'activation_names',{})}
 def score_choice(self,prompt,options,media_path=None,image=None):
  self.calls+=1;self._reset();e={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None};started=time.monotonic()
  try:
   assert image is None and media_path;media=pathlib.Path(media_path).resolve();assert str(media) in self.p['reference_first200_image_sha256'] and sha(media)==self.p['reference_first200_image_sha256'][str(media)]
   with Image.open(media) as im:image=im.convert('RGB')
   prefix=(self.processor.image_token+'\n'+prompt).rstrip('\n')+'\nAnswer:';prepared=self.processor(text=prefix,images=[image],return_tensors='pt');assert (prepared['input_ids']==32001).sum().item()==64 and prepared['input_ids'].max()<32002 and prepared['input_ids'].min()>=0
   e.update(prefix_text=prefix,CPU_inputs={k:tr(v) for k,v in prepared.items()},input_ids=prepared['input_ids'].tolist(),attention_mask=prepared['attention_mask'].tolist(),source_image_sha256=sha(media))
   inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()};self._active_ids=inputs['input_ids'];length=self._active_ids.shape[-1]
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
       out=self.model(**inputs);torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True
    else:out=self.model(**inputs)
    assert out.logits.dtype==torch.bfloat16 and out.logits.shape==(1,length,32002) and torch.isfinite(out.logits).all();last=out.logits[0,-1];log_probs=torch.log_softmax(last,dim=-1);assert log_probs.dtype==torch.bfloat16 and torch.isfinite(log_probs).all()
   scores={};candidate_ids={}
   for opt in options:
    cand=opt.strip()
    if not cand:continue
    ids=self.tokenizer.encode(cand,add_special_tokens=False)
    if not ids:ids=self.tokenizer.encode(' '+cand,add_special_tokens=False)
    assert ids and 0<=ids[0]<32002;candidate_ids[opt]=ids[0];scores[opt]=float(log_probs[ids[0]].item())
   assert scores and self._vision_calls==self._connector_calls==self._text_calls==1 and all(v==1 for v in self._activation_counts.values()) and self._vision is not None and self._features.shape==(1,64,4096) and self._injected is not None and self._text_positions is not None and self._text_mask is not None
   assert self._text_positions.tolist()==[list(range(length))];cache=out.past_key_values;assert cache is not None and cache.get_seq_length()==length
   cache_info=[]
   for i,layer in enumerate(cache.layers):
    assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.bfloat16
    cache_info.append({'slot':i,'key_shape':list(layer.keys.shape),'value_shape':list(layer.values.shape),'device':str(layer.keys.device),'dtype':str(layer.keys.dtype)})
   assert len(cache_info)==32
   tensors={'actual_encoder':self._vision,'actual_connector':self._features,'actual_injected_visual':self._injected,'actual_text_inputs_embeds':self._merged,'actual_vision_pixels':self._actual_pixels,'actual_patch_attention_mask':self._patch_mask,'actual_vision_positions':self._vision_positions,'actual_text_positions':self._text_positions,'actual_text_causal_mask':self._text_mask,'actual_rope_cos':self._cos,'actual_rope_sin':self._sin,'actual_last_logits':last,'actual_BF16_log_softmax':log_probs}
   assert all(torch.isfinite(v).all() for k,v in tensors.items() if v.is_floating_point());archive=self.evidence/'full_vocab_and_visual'/('%06d.npz'%(self.calls-1));np.savez_compressed(archive,**{k:v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy() for k,v in tensors.items()})
   best=max(scores,key=scores.get);e.update(raw_output=best,candidate_scores=scores,candidate_first_token_ids=candidate_ids,original_full_forward_logit_shape=list(out.logits.shape),full_vocab_size=32002,BF16_log_softmax=True,no_generation=True,generation_token_ids=[],generation_count=0,truncated=False,empty=False,invalid=False,actual_vision_calls=1,actual_connector_calls=1,actual_text_model_calls=1,actual_visual_injection_verified=True,full_vocab_and_visual_archive=str(archive),full_vocab_and_visual_archive_sha256=sha(archive),tensor_records={k:tr(v) for k,v in tensors.items()},native_activation_call_counts=dict(self._activation_counts),cache_device_and_shapes=cache_info,final_cache_length=int(cache.get_seq_length()))
   with self.activation_trace.open('a',encoding='utf8') as f:f.write(json.dumps({'attempt':self.calls,'device':str(self.device),'dtype':'torch.bfloat16','module_call_counts':dict(self._activation_counts),'error':None})+'\n')
   log.info('Idefics2 base original NPU sample %s options=%s selected=%s',self.calls,scores,best);return scores
  except Exception:e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   e['elapsed_seconds']=time.monotonic()-started;self.details[(prompt.strip(),media_path)]=e
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._reset()
 def predict(self,prompt,media_path=None):raise RuntimeError('The original base model protocol is score_choice, not free generation')
 def batch_predict(self,batch):raise RuntimeError('The original mc_score algorithm invokes one score_choice per sample')
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

"""Original Idefics2 instruction generation: true size groups within batches10."""
import os,pathlib,json,hashlib,datetime,time,logging,types,contextlib,traceback,collections
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
from universal_eval.model.adapters.idefics2_base_original_v3_910b_adapter import sha,fp,tr,compatible_vision_embedding
log=logging.getLogger('universal_eval.model.adapters.idefics2_grouped_generation_910b')
INSTRUCTION='\n\nAnswer the question by providing only the option letter (A, B, C, D, or E) in the format <answer>X</answer>.'
def clean(text):
 import re
 text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(text))
 text=re.sub(r'[ \t]+',' ',text)
 return re.sub(r'<image\d+>\s*','',re.sub(r'\n{3,}','\n\n',text).strip())
def append(path,obj):
 with path.open('a',encoding='utf8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
def cpu(t):return t.detach().cpu().contiguous()
def save_arrays(path,tensors):
 np.savez_compressed(path,**{k:cpu(v).reshape(-1).view(torch.uint8).numpy() for k,v in tensors.items()})
 return {k:tr(v) for k,v in tensors.items()}
class Idefics2GroupedGenerationV2_910BAdapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256
  self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256
  self.devices=list(npu_devices or [6]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]))
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
  self.evidence=pathlib.Path(evidence_dir).resolve()/('Idefics2_original_grouped_generation_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
  (self.evidence/'generation_groups').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.activations=self.evidence/'native_activation_execution.jsonl';self.progress=self.evidence/'generation_progress.jsonl';self.finiteness=self.evidence/'next_token_logits_finiteness.jsonl'
  self.group_count=0;self.sample_count=0;self.details={};self.lineage={};self.profile_done=False
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Idefics2ForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers torch_npu original BF16 size-grouped greedy generation'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoProcessor,Idefics2ForConditionalGeneration,GenerationConfig
  from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name in ['HuggingFaceM4/idefics2-8b','HuggingFaceM4/idefics2-8b-chatty']
  for rel,h in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(wp/rel)==h
  for key in ['reference_report','reference_log','dataset_path','original_dataset','CPU_grouped_input_preflight','original_protocol_review']:assert sha(p[key])==p[key+'_sha256']
  for src,h in dict(p['runtime_sha256'],**p['historical_sources'],**p['frozen_before_original_generation_sources']).items():assert sha(src)==h
  assert p['original_max_new_tokens']==2048 and p['original_algorithm_batch_size']==10 and p['image_splitting'] is True and p['model_vocab_size']==32003 and p['image_tokens_per_original_image']==320
  torch.set_num_threads(4);assert torch.npu.is_available();torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.tie_word_embeddings is False and cfg.text_config.vocab_size==32003
  for c in [cfg,cfg.text_config,cfg.vision_config,cfg.perceiver_config]:c._attn_implementation='eager'
  with torch.device('meta'):self.model=Idefics2ForConditionalGeneration(cfg)
  params=dict(self.model.named_parameters());expected={v['source_key']:v for v in p['source_tensor_records']};assert len(params)==763 and set(params)==set(expected)
  loaded=[];seen=set()
  for shard,header in p['weight_headers'].items():
   path=wp/shard;assert path.stat().st_size==header['size'] and sha(path)==header['file_sha256']
   with safe_open(str(path),framework='pt',device='cpu') as f:
    for name in f.keys():
     assert name in expected and name not in seen;r=expected[name];x=f.get_tensor(name);y=x.bfloat16()
     assert x.dtype==torch.float32 and list(x.shape)==r['shape']==list(params[name].shape) and fp(x)==r['source_sha256'] and fp(y)==r['original_BF16_cast_sha256']
     target=y.to(self.device);assert fp(target)==r['original_BF16_cast_sha256']
     parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));seen.add(name);loaded.append(dict(r,device=str(self.device),actual_NPU_sha256=fp(target)))
   log.info('Original grouped Idefics2 source loaded %s %s/763',shard,len(seen))
  assert seen==set(params)
  self.model.model.text_model.rotary_emb=MistralRotaryEmbedding(cfg.text_config,device='cpu').to(self.device)
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters()) and all(not t.is_meta and t.device==self.device for t in self.model.buffers())
  assert self.model.lm_head.weight is not self.model.model.text_model.embed_tokens.weight and fp(self.model.lm_head.weight)!=fp(self.model.model.text_model.embed_tokens.weight)
  self.model.eval();self.model.generation_config=GenerationConfig.from_pretrained(wp,local_files_only=True)
  gen,_=self.model._prepare_generation_config(None,**p['original_generate_kwargs']);assert gen.to_dict()==p['resolved_original_generation_config']
  self.processor=AutoProcessor.from_pretrained(wp,local_files_only=True);self.tokenizer=self.processor.tokenizer;self.tokenizer.padding_side='left'
  assert len(self.tokenizer)==32003 and self.processor.image_processor.do_image_splitting and self.processor.image_seq_len==64 and json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['own_image_processor'] and self.processor.chat_template==p['own_chat_template']
  self.model.model.vision_model.embeddings.forward=types.MethodType(compatible_vision_embedding,self.model.model.vision_model.embeddings)
  self.activation_names={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['GELUTanh','SiLUActivation']}
  assert collections.Counter(self.activation_names.values())=={'GELUTanh':27,'SiLUActivation':36}
  self._install_observers()
  dataset=pathlib.Path(p['dataset_path']);self.rows=[json.loads(s) for s in dataset.read_text().splitlines()][:200];self.lookup={}
  for i,r in enumerate(self.rows):
   media=(dataset.parent/r['media_path']).resolve();prompt=clean(r['prompt']);self.lookup[(prompt+INSTRUCTION,str(media))]=i
  probe_values=torch.linspace(-8,8,20001).bfloat16();arrays={'input_BF16_bytes':probe_values.view(torch.uint8).numpy()};probe_record={}
  for approximate in ['none','tanh']:
   n=torch.nn.functional.gelu(probe_values.to(self.device),approximate=approximate).cpu();c=torch.nn.functional.gelu(probe_values,approximate=approximate)
   arrays['NPU_'+approximate+'_BF16_bytes']=n.view(torch.uint8).numpy();arrays['CPU_'+approximate+'_BF16_bytes']=c.view(torch.uint8).numpy();probe_record[approximate]={'max_abs_CPU_same_declared':float((n-c).abs().max())}
  probe=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(probe,**arrays);probe_record.update(path=str(probe),sha256=sha(probe),device=str(self.device),diagnostic_only=True,CPU_threads=4)
  weights=self.evidence/'weight_loading.json';weights.write_text(json.dumps(loaded,indent=2))
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'idefics2_original_grouped_generation_BF16','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'weight_loading_evidence':str(weights),'per_sample_evidence':str(self.trace),'activation_execution_trace':str(self.activations),'generation_progress':str(self.progress),'next_token_logits_finiteness':str(self.finiteness),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':probe_record,'weight_tensor_count':763,'source_dtype':'F32','runtime_dtype':'BF16','all_expected_parameters_loaded':True,'activation_modules':self.activation_names,'native_attention_implementation':'eager','image_splitting':True,'images_per_sample':5,'image_tokens_per_sample':320,'model_vocab_size':32003,'original_algorithm_batch_size':10,'actual_generation_batch_size':'Original within-batch10 insertion-ordered size groups, including real batches10; never silently serialized','original_max_new_tokens':2048,'original_cache':True,'official_padding_position_resolution':'GenerationMixin._prepare_position_ids_for_generation: cumsum(attention_mask)-1, masked padding positions0; valid positions0..N preserved','independent_lm_head_preserved':True,'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'text_inv_freq':tr(self.model.model.text_model.rotary_emb.inv_freq),'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'source_underflow_total':p['source_underflow_total'],'source_BF16_roundtrip_changed_elements':p['source_BF16_roundtrip_changed_elements'],'precision_disclosure':'Own763 sourceF32 tensors converted by the original explicit CPU BF16 cast; each source/cast/NPU tensor SHA preserved, own untied32003 output head retained. All learned vision/perceiver/Mistral/head layers compute natively on NPU BF16 with HF32off and native eager attention/cache. Original5 image crops and original true size groups inside batches10 preserved. SearchSorted alone uses lossless FP32 representations of already BF16-rounded coordinates. Read-only hooks preserve full actual next-token logits and exact input/position/cache/vision evidence. No forced option mask, changed generation budget, CPU official outputs, low-bit acceleration claim or CPU/NPU bitwise arithmetic claim.'}
 def _reset(self,batch_size):
  self._B=batch_size;self._steps=[];self._logits=[];self._counts={n:0 for n in self.activation_names};self._vision_calls=0;self._connector_calls=0;self._vision=None;self._features=None;self._visual={};self._active=None;self._step=None
 def _install_observers(self):
  def root_pre(module,args,kw):
   ids=kw.get('input_ids',args[0] if args else None);assert ids is not None and ids.device==self.device
   cache=kw.get('past_key_values');past=int(cache.get_seq_length()) if cache is not None else 0
   self._active=ids;self._step={'step':len(self._steps),'input_ids':ids.cpu().tolist(),'input_ids_sha256':fp(ids),'past_length_before':past,'use_cache':kw.get('use_cache'),'attention_mask':kw['attention_mask'].cpu().tolist()}
   entered=len(self._steps)+1
   if entered==1 or entered%128==0:append(self.progress,{'group':self.group_count,'batch_size':self._B,'decoder_step_entered':entered,'generated_before_this_step':entered-1,'completed':False,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  def root_post(module,args,out):
   x=out.logits[:,-1,:];assert x.shape==(self._B,32003) and x.dtype==torch.bfloat16 and x.device==self.device
   finite=bool(torch.isfinite(x).all().item());append(self.finiteness,{'group':self.group_count,'step':len(self._steps),'batch_size':self._B,'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'all_finite':finite});assert finite
   self._logits.append(cpu(x));cache=out.past_key_values;assert cache is not None
   self._step['cache_length_after']=int(cache.get_seq_length());self._step['cache_layers']=[]
   for i,layer in enumerate(cache.layers):
    assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.bfloat16
    self._step['cache_layers'].append({'layer':i,'keys':list(layer.keys.shape),'values':list(layer.values.shape),'dtype':str(layer.keys.dtype),'device':str(self.device)})
   assert len(self._step['cache_layers'])==32;self._steps.append(self._step)
  def vision_pre(module,args,kw):
   self._visual['actual_vision_pixels']=cpu(kw['pixel_values']);self._visual['actual_patch_attention_mask']=cpu(kw['patch_attention_mask'])
  def vision_post(module,args,out):self._vision=out.last_hidden_state;self._vision_calls+=1;self._visual['actual_encoder']=cpu(self._vision)
  def connector(module,args,out):self._features=out;self._connector_calls+=1;self._visual['actual_connector']=cpu(out)
  def vision_pos(module,args):self._visual['actual_vision_positions']=cpu(args[0])
  def text_pre(module,args,kw):
   x=kw['inputs_embeds'];assert x.device==self.device and x.dtype==torch.bfloat16
   if not self._steps:
    assert self._features is not None and self._features.shape==(self._B*5,64,4096)
    mask=self._active==32001;injected=x[mask];assert injected.shape==(self._B*320,4096) and torch.equal(injected,self._features.reshape(-1,4096))
    orig=module.embed_tokens(self._active);assert torch.equal(x[~mask],orig[~mask]);self._visual['actual_prefill_text_inputs_embeds']=cpu(x);self._visual['actual_injected_visual']=cpu(injected)
  def rope(module,args,kw,out):
   positions=kw.get('position_ids',args[1] if len(args)>1 else None);assert positions is not None
   self._step['position_ids']=positions.cpu().tolist();self._step['rope_cos']=tr(out[0]);self._step['rope_sin']=tr(out[1])
   self._visual['rope_cos_%04d'%len(self._steps)]=cpu(out[0]);self._visual['rope_sin_%04d'%len(self._steps)]=cpu(out[1])
  def attention(module,args,kw):
   mask=kw.get('attention_mask');assert mask is not None and mask.device==self.device
   self._visual['causal_mask_%04d'%len(self._steps)]=cpu(mask);self._step['actual_causal_mask']=tr(mask)
  def activation(name):
   def hook(module,args,out):assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   return hook
  self.model.register_forward_pre_hook(root_pre,with_kwargs=True);self.model.register_forward_hook(root_post)
  vision=self.model.model.vision_model;vision.register_forward_pre_hook(vision_pre,with_kwargs=True);vision.register_forward_hook(vision_post);vision.embeddings.position_embedding.register_forward_pre_hook(vision_pos);self.model.model.connector.register_forward_hook(connector)
  text=self.model.model.text_model;text.register_forward_pre_hook(text_pre,with_kwargs=True);text.rotary_emb.register_forward_hook(rope,with_kwargs=True);text.layers[0].self_attn.register_forward_pre_hook(attention,with_kwargs=True)
  for n,m in self.model.named_modules():
   if n in self.activation_names:m.register_forward_hook(activation(n))
 def _group(self,items):
  self.group_count+=1;self._reset(len(items));start=time.monotonic();images=[];texts=[];records=[]
  for index,prompt,media_path in items:
   media=pathlib.Path(media_path).resolve();assert str(media) in self.p['reference_first200_image_sha256'] and sha(media)==self.p['reference_first200_image_sha256'][str(media)]
   with Image.open(media) as im:images.append(im.convert('RGB'))
   messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
   text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False);texts.append(text)
   records.append({'index':index,'prompt':prompt,'media_path':media_path,'resolved_media_path':str(media),'source_image_sha256':sha(media),'original_size':list(images[-1].size),'chat_text':text})
  prepared=self.processor(text=texts if len(items)>1 else texts[0],images=[[im] for im in images] if len(items)>1 else images,return_tensors='pt',padding=True)
  assert prepared['input_ids'].shape[0]==len(items) and (prepared['input_ids']==32001).sum().item()==320*len(items) and prepared['input_ids'].max()<32003
  L=prepared['input_ids'].shape[-1];assert L+2048<4096
  group={'group':self.group_count,'indices':[x[0] for x in items],'batch_size':len(items),'size_group':[images[0].width//64,images[0].height//64],'samples':records,'CPU_inputs':{k:tr(v) for k,v in prepared.items()},'input_ids':prepared['input_ids'].tolist(),'attention_mask':prepared['attention_mask'].tolist(),'padded_input_length':L,'error':None,'original_max_new_tokens':2048,'original_use_cache':True}
  inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()}
  try:
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
       out=self.model.generate(**inputs,**self.p['original_generate_kwargs'],return_dict_in_generate=True,output_scores=False);torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True
    else:out=self.model.generate(**inputs,**self.p['original_generate_kwargs'],return_dict_in_generate=True,output_scores=False)
   sequences=out.sequences.cpu();generated=sequences[:,L:];steps=len(self._steps)
   assert 0<steps<=2048 and generated.shape==(len(items),steps) and torch.equal(sequences[:,:L],prepared['input_ids']) and self._vision_calls==self._connector_calls==1
   for n,v in self._counts.items():assert v==(steps if '.text_model.' in n else 1),(n,v,steps)
   logits=torch.stack(self._logits);raw=self.processor.batch_decode(generated,skip_special_tokens=True)
   root=self.evidence/'generation_groups'/('%04d'%self.group_count);visual=root.with_name(root.name+'_visual.npz');logit_path=root.with_name(root.name+'_full_logits.npz')
   group['visual_tensor_records']=save_arrays(visual,self._visual);group['logits_tensor_record']=save_arrays(logit_path,{'actual_next_token_full_logits':logits})
   group.update(visual_archive=str(visual),visual_archive_sha256=sha(visual),full_logits_archive=str(logit_path),full_logits_archive_sha256=sha(logit_path),actual_steps=self._steps,decoder_steps=steps,full_sequences=sequences.tolist(),generation_token_ids=generated.tolist(),actual_vision_calls=self._vision_calls,actual_connector_calls=self._connector_calls,native_activation_call_counts=dict(self._counts),elapsed_seconds=time.monotonic()-start)
   outputs=[]
   for j,r in enumerate(records):
    ids=generated[j].tolist();ends=[k for k,v in enumerate(ids) if v in [2,32002]];active=ends[0]+1 if ends else len(ids);budget=active>=2048
    text=raw[j].strip();e=dict(r,attempt=r['index']+1,raw_output=text,full_decoded_output_including_special_tokens=self.tokenizer.decode(ids,skip_special_tokens=False),generation_token_ids=ids,first_EOS_index=ends[0] if ends else None,active_generation_count=active,padded_generation_count=len(ids),generation_count=active,truncated=budget,budget_reached=budget,empty=not bool(text),invalid=explicit_choice(text) is None,error=None,actual_generation_batch_size=len(items),actual_group=self.group_count,actual_group_row=j,actual_visual_injection_verified=True)
    outputs.append(e)
   append(self.activations,{'group':self.group_count,'batch_size':len(items),'decoder_steps':steps,'device':str(self.device),'dtype':'torch.bfloat16','module_call_counts':dict(self._counts),'counts_are_per_true_group_not_per_sample':True,'error':None})
   append(self.progress,{'group':self.group_count,'batch_size':len(items),'batch_complete':True,'decoder_steps_completed':steps,'generated_tokens_in_padded_group':steps,'indices':group['indices'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
   record=root.with_suffix('.json');record.write_text(json.dumps(group,ensure_ascii=False,indent=2))
   for e in outputs:e.update(group_evidence=str(record),group_evidence_sha256=sha(record),visual_archive=str(visual),visual_archive_sha256=sha(visual),full_logits_archive=str(logit_path),full_logits_archive_sha256=sha(logit_path))
   log.info('Original grouped Idefics2 NPU group %s batch=%s steps=%s indices=%s',self.group_count,len(items),steps,group['indices']);return outputs
  except Exception:
   group['error']=traceback.format_exc();(root if 'root' in locals() else self.evidence/'generation_groups'/('%04d'%self.group_count)).with_suffix('.failed.json').write_text(json.dumps(group,ensure_ascii=False,indent=2));raise
  finally:
   self._reset(0)
 def batch_predict(self,batch):
  items=[]
  for item in batch:
   prompt,media=item;resolved=str(pathlib.Path(media).resolve());index=self.lookup[(prompt,resolved)];items.append((index,prompt,media))
  expected=list(range(self.sample_count,self.sample_count+len(items)));assert [x[0] for x in items]==expected and len(items)<=10 and self.sample_count%10==0
  groups=collections.OrderedDict()
  for item in items:
   with Image.open(item[2]) as image:key=(image.width//64,image.height//64)
   groups.setdefault(key,[]).append(item)
  results={}
  for group in groups.values():
   for e in self._group(group):results[e['index']]=e
  ordered=[]
  for index,prompt,media in items:
   e=results[index];append(self.trace,e);self.details[(prompt,media)]=e;self.details[(clean(self.rows[index]['prompt']),media)]=e;ordered.append(e['raw_output'])
  self.sample_count+=len(items);return ordered
 def predict(self,prompt,media_path=None):raise RuntimeError('Original instruction models require original true size-grouped batch_predict')
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing original grouped execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

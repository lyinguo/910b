"""Own Zelk actual-A100 HighNine HF BF16 serial generation; native Ascend learned forward and full evidence."""
import os,pathlib,json,hashlib,datetime,logging,time,traceback,contextlib,math,functools,collections
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.zelk12_A100_HF_0913_20261106')

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
def mapped(key):
 for old,new in [('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.'),('vision_tower.vision_model.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.')]:
  if key.startswith(old):return new+key[len(old):]
 raise ValueError('Unreviewed original checkpoint key '+key)

class Zelk12A100HighNineHF0913Adapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',protocol=None,**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256
  assert protocol=={'model_id':self.p['model_id'],'protocol_file_sha256':protocol_sha256},'Own inline identity must bind the exact external protocol'
  self.devices=list(npu_devices or [9]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('Zelk12_A100_HighNine_HF_0913_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
  (self.evidence/'groups').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.activation_trace=self.evidence/'native_activation_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Gemma3ForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers/torch_npu actual A100 original HighNine HF BF16 serial1 greedy generation'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoProcessor,Gemma3ForConditionalGeneration,GenerationConfig
  from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name=='zelk12/MT-Gen1-gemma-3-12B' and p['original_runtime_dtype']=='BF16'
  for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
  for src,digest in dict(p['runtime_sha256'],**p['historical_sources'],**p['evidence_sha256']).items():assert sha(src)==digest
  assert p['original_algorithm']=='greedy_inference' and p['original_algorithm_batch_size']==1 and p['original_max_new_tokens']==2048 and p['actual_original_serial_request_count']==200
  assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.architectures==['Gemma3ForConditionalGeneration'] and not getattr(cfg,'quantization_config',None) and cfg.tie_word_embeddings and cfg.text_config.use_cache and cfg.text_config.vocab_size==262208 and cfg.mm_tokens_per_image==256
  cfg.dtype=torch.bfloat16;cfg.text_config.dtype=torch.bfloat16;cfg.vision_config.dtype=torch.bfloat16;cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='eager'
  with torch.device('meta'):self.model=Gemma3ForConditionalGeneration(cfg)
  assert cfg.text_config._attn_implementation=='sdpa' and cfg.vision_config._attn_implementation=='eager';targets=dict(self.model.named_parameters(remove_duplicate=False));assert len(dict(self.model.named_parameters()))==1065
  expected={r['source_key']:r for r in p['source_tensor_records']};records=[];seen=set();filled={}
  for shard,header in p['weight_headers'].items():
   assert pathlib.Path(shard).name==shard and (wp/shard).stat().st_size==header['bytes'] and sha(wp/shard)==header['file_sha256']
   with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
    for key in f.keys():
     assert key not in seen and key in expected;name=mapped(key);r=expected[key];source=f.get_tensor(key);cast=source.bfloat16()
     assert name==r['target_key'] and name in targets and list(source.shape)==r['shape']==list(targets[name].shape) and str(source.dtype)==r['source_dtype'] and fp(source)==r['source_sha256'] and fp(cast)==r['original_BF16_cast_sha256'] and torch.isfinite(cast).all()
     target=cast.to(self.device);actual=fp(target);assert actual==r['original_BF16_cast_sha256'];parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));seen.add(key);filled[name]=actual;records.append(dict(r,actual_NPU_sha256=actual,runtime_dtype='torch.bfloat16',device=str(target.device)))
   log.info('Gemma3 original score %s loaded %s %s/1065',self.model_name,shard,len(seen))
  assert seen==set(expected) and len(seen)==1065
  for group in p['native_alias_groups']:
   saved=[name for name in group if name in filled];assert saved and len({filled[name] for name in saved})==1
   parent,leaf=saved[0].rsplit('.',1);value=getattr(self.model.get_submodule(parent),leaf)
   for name in group:
    parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,value)
  self.model.tie_weights();assert self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight
  for r in records:
   parent,leaf=r['target_key'].rsplit('.',1);assert fp(getattr(self.model.get_submodule(parent),leaf))==r['actual_NPU_sha256']
  text=self.model.model.language_model;text.rotary_emb=Gemma3RotaryEmbedding(cfg.text_config).to(self.device);text.embed_tokens.embed_scale=torch.tensor(math.sqrt(cfg.text_config.hidden_size),dtype=torch.bfloat16).to(self.device)
  vision=self.model.model.vision_tower;vision.embeddings.position_ids=torch.arange((cfg.vision_config.image_size//cfg.vision_config.patch_size)**2,dtype=torch.int64).expand((1,-1)).to(self.device)
  assert all(v.device==self.device and v.dtype==torch.bfloat16 for v in self.model.parameters()) and all(not v.is_meta and v.device==self.device for v in self.model.buffers())
  self.model.eval();self.processor=AutoProcessor.from_pretrained(wp,local_files_only=True);self.tokenizer=self.processor.tokenizer
  assert type(self.processor).__name__=='Gemma3Processor' and json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['own_image_processor'] and self.processor.image_seq_length==256 and self.processor.chat_template is None and self.tokenizer.chat_template==p['own_tokenizer_template'] and len(self.tokenizer.chat_template)==1532
  self.processor.chat_template=self.tokenizer.chat_template;self.tokenizer.padding_side='right';self.tokenizer.eos_token='<end_of_turn>';self.tokenizer.model_max_length=131072;assert self.tokenizer.eos_token_id==106;self.model.generation_config=GenerationConfig.from_dict(p['original_resolved_generation_config']);self.generation_config,self._generation_model_kwargs=self.model._prepare_generation_config(None,max_new_tokens=2048,do_sample=False);assert self.generation_config.to_dict()==p['original_resolved_generation_config'] and not (wp/'generation_config.json').exists()
  self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['GELUTanh','GELUActivation','PytorchGELUTanh','SiLUActivation','NewGELUActivation','QuickGELUActivation','SiLU']};assert len(self.activation_modules)==75 and set(self.activation_modules.values())=={'GELUTanh'}
  self.group_calls=0;self._reset();self._install_observers()
  values=torch.linspace(-8,8,20001,dtype=torch.float32).bfloat16();arrays={'input_BF16_bytes':values.view(torch.uint8).numpy()};probe={}
  for approximate in ['none','tanh']:
   native=torch.nn.functional.gelu(values.to(self.device),approximate=approximate).cpu();cpu=torch.nn.functional.gelu(values,approximate=approximate);arrays['NPU_'+approximate+'_BF16_bytes']=native.view(torch.uint8).numpy();arrays['CPU_'+approximate+'_BF16_bytes']=cpu.view(torch.uint8).numpy();probe[approximate]={'max_abs_CPU_same_declared_GELU':float((native-cpu).abs().max())}
  pq=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(pq,**arrays);self.probe=dict(probe,path=str(pq),sha256=sha(pq),CPU_threads=torch.get_num_threads(),device=str(self.device),diagnostic_only=True)
  loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir();(snapshot/'original_protocol_review.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
  for src in set(p['runtime_sha256'])|set(p['historical_sources']):
   q=pathlib.Path(src);dest=snapshot/q.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(q.read_bytes())
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'zelk12_actual_A100_HighNine_HF_BF16_serial1_generation','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'runtime_dtype':'BF16','source_dtype_counts':p['source_dtype_counts'],'weight_tensor_count':1065,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(loading),'per_sample_evidence':str(self.trace),'per_group_evidence':str(self.evidence/'groups'),'activation_execution_trace':str(self.activation_trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':self.probe,'activation_modules':self.activation_modules,'original_tied_output_head_preserved':True,'original_algorithm_batch_size':1,'actual_A100_attention':{'text':'sdpa','vision':'eager'},'original_AA_live_generation_defaults_explicitly_reconstructed':True,'own_instance_tokenizer_config_differences':p['own_tokenizer_config_instance_differences'],'actual_batching':'Actual A100 original HighNine algorithm batch1; each request is serial1, no size grouping','max_new_tokens':2048,'original_resolved_generation_config':self.generation_config.to_dict(),'original_two_BOS_preserved':True,'own_template_explicit_injection':self.processor.chat_template,'padding_side':'right','image_seq_len':256,'model_vocab_size':262208,'image_processor':self.processor.image_processor.to_dict(),'text_rotary_buffers':{n:tr(v) for n,v in text.rotary_emb.named_buffers()},'original_embedding_scale':tr(text.embed_tokens.embed_scale),'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'precision_disclosure':'Own1065 source BF16 parameters preserve actual original A100 HighNine explicit BF16; full source/cast/actual NPU SHA, own tied head, all learned vision/projector/Gemma3/head on own assigned NPU BF16, HF32off. Original FP32 analytic global/sliding RoPE and original BF16 embedding-scale rounding. Original text SDPA + vision eager, own resolved native cache/EOS defaults, original serial1 greedy generation max2048. Original two BOS and own1532 template/rightpadding preserved. No grammar, budget reduction, forced early EOS, replacement weights, CPU learned layer offload or native low-bit claim. CPU/FP32 diagnostic only; native arithmetic differences retained.'}
 def _reset(self):
  self._arrays={};self._steps=[];self._step=None;self._counts={n:0 for n in getattr(self,'activation_modules',{})};self._features=None;self._vision_calls=0;self._connector_calls=0;self._bridge=False;self._last_prefix=None
 def _store(self,name,x):
  assert x.device==self.device;self._arrays[name]=x.detach().cpu().contiguous().clone();return tr(self._arrays[name])
 def _install_observers(self):
  vision=self.model.model.vision_tower;text=self.model.model.language_model;cfg=self.model.config
  original_forward=self.model.forward
  @functools.wraps(original_forward)
  def profiled(*args,**kwargs):
   if self.profile_done:return original_forward(*args,**kwargs)
   with contextlib.chdir(self.evidence):
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
     out=original_forward(*args,**kwargs);torch.npu.synchronize(self.device)
    prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
   self.profile_done=True;return out
  self.model.forward=profiled
  original_prepare=self.model.prepare_inputs_for_generation
  @functools.wraps(original_prepare)
  def prepare(input_ids,*args,**kwargs):
   self._last_prefix=input_ids.detach().cpu().clone();return original_prepare(input_ids,*args,**kwargs)
  self.model.prepare_inputs_for_generation=prepare
  def root_pre(module,args,kw):
   assert self._step is None and self._last_prefix is not None
   self._step={'step':len(self._steps)+1,'full_generation_prefix':tr(self._last_prefix),'input_ids':kw['input_ids'].cpu().tolist(),'position_ids':kw['position_ids'].cpu().tolist(),'attention_mask_2D':kw['attention_mask'].cpu().tolist(),'pixel_values_present':kw.get('pixel_values') is not None,'all48_attention':[],'masks':{}}
  def root_post(module,args,out):
   step=self._step;n=step['step'];logits=out.logits[:,-1];assert logits.shape==(len(self._group_items),262208) and logits.dtype==torch.bfloat16 and logits.device==self.device and torch.isfinite(logits).all()
   self._store('%04d_logits'%n,logits);step['full_vocab_logits']=tr(logits);step['actual_logits_shape']=list(out.logits.shape)
   cache=out.past_key_values;assert len(cache.layers)==48;step['cache']=[]
   for i,layer in enumerate(cache.layers):
    assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.bfloat16
    step['cache'].append({'layer':i,'class':type(layer).__name__,'shape':list(layer.keys.shape),'value_shape':list(layer.values.shape),'cumulative_length':int(layer.get_seq_length()),'device':str(layer.keys.device),'dtype':str(layer.keys.dtype),'key_sha256':fp(layer.keys),'value_sha256':fp(layer.values)})
   assert len(step['all48_attention'])==48;self._steps.append(step);self._step=None
   if n==1 or n%128==0:
    with (self.evidence/'generation_progress.jsonl').open('a') as f:f.write(json.dumps({'group':self.group_calls,'batch_size':len(self._group_items),'completed_decoder_forward':n,'generated_before_this_forward':n-1})+'\n')
  def pixels(module,args,kw):self._store('actual_vision_pixels',kw.get('pixel_values',args[0] if args else None))
  def vision_post(module,args,out):self._store('actual_encoder',out.last_hidden_state);self._vision_calls+=1
  def projection(module,args,out):self._features=out;self._store('actual_connector',out);self._connector_calls+=1
  def positions(module,args):self._store('actual_vision_positions',args[0])
  def language(module,args,kw):
   if self._step['step']!=1:return
   x=kw['inputs_embeds'];mask=self._active_ids==cfg.image_token_id;assert torch.equal(x[mask],self._features.reshape(-1,cfg.text_config.hidden_size));original=module.embed_tokens(self._active_ids);assert torch.equal(x[~mask],original[~mask]);self._store('actual_prefill_inputs_embeds',x);self._store('actual_injected_visual',x[mask]);self._bridge=True
  def rope(module,args,kw,out):
   kind=kw.get('layer_type',args[2] if len(args)>2 else None);pos=kw.get('position_ids',args[1] if len(args)>1 else None);n=self._step['step'];assert pos.cpu().tolist()==self._step['position_ids'];assert kind in ['full_attention','sliding_attention'];self._store('%04d_%s_cos'%(n,kind),out[0]);self._store('%04d_%s_sin'%(n,kind),out[1])
  def attention(i):
   def hook(module,args,kw):
    kind=module.layer_type;mask=kw.get('attention_mask');n=self._step['step'];record=None if mask is None else tr(mask)
    if kind in self._step['masks']:assert self._step['masks'][kind]==record
    else:
     self._step['masks'][kind]=record
     if mask is not None:self._store('%04d_%s_mask'%(n,kind),mask)
    assert kw['position_ids'].cpu().tolist()==self._step['position_ids'];self._step['all48_attention'].append({'layer':i,'layer_type':kind,'mask':record,'scaling':module.scaling,'sliding_window':module.sliding_window})
   return hook
  def activation(name):
   def hook(module,args,out):assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   return hook
  self.model.register_forward_pre_hook(root_pre,with_kwargs=True);self.model.register_forward_hook(root_post);vision.register_forward_pre_hook(pixels,with_kwargs=True);vision.register_forward_hook(vision_post);vision.embeddings.position_embedding.register_forward_pre_hook(positions);self.model.model.multi_modal_projector.register_forward_hook(projection);text.register_forward_pre_hook(language,with_kwargs=True);text.rotary_emb.register_forward_hook(rope,with_kwargs=True)
  for i,layer in enumerate(text.layers):layer.self_attn.register_forward_pre_hook(attention(i),with_kwargs=True)
  for name,module in self.model.named_modules():
   if name in self.activation_modules:module.register_forward_hook(activation(name))
 def _generate_group(self,items):
  assert len(items)==1,'Actual original HighNine serial1';self.group_calls+=1;self._reset();self._group_items=items;record={'group':self.group_calls,'batch_size':len(items),'indices':[x['index'] for x in items],'error':None};start=time.monotonic();home=self.evidence/'groups'/('%04d'%self.group_calls);home.mkdir()
  try:
   images=[];rendered=[]
   for item in items:
    media=pathlib.Path(item['media_path']).resolve();assert sha(media)==self.p['reference_first200_image_sha256'][str(media)]
    with Image.open(media) as im:images.append(im.convert('RGB'))
    rendered.append(self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':item['model_prompt']}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
   if len(items)==1:prepared=self.processor(text=rendered[0],images=[images[0]],return_tensors='pt',padding=True)
   else:prepared=self.processor(text=rendered,images=[[im] for im in images],return_tensors='pt',padding=True)
   assert set(prepared)=={'pixel_values','input_ids','attention_mask','token_type_ids'} and (prepared['input_ids']==262144).sum()==256*len(items)
   for i,item in enumerate(items):
    expected=self.p['all200_independent_input_records'][item['index']];active=prepared['input_ids'][i][prepared['attention_mask'][i].bool()];assert active.tolist()==expected['independent_unpadded_input_ids'] and active[:2].tolist()==[2,2] and rendered[i]==expected['own_original_template_rendered_text'];assert fp(prepared['pixel_values'][i:i+1].bfloat16())==expected['pixel_values_sha256']
   record.update(rendered_texts=rendered,CPU_inputs={k:tr(v) for k,v in prepared.items()},input_ids=prepared['input_ids'].tolist(),attention_mask=prepared['attention_mask'].tolist(),token_type_ids=prepared['token_type_ids'].tolist())
   inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()};self._active_ids=inputs['input_ids'];length=self._active_ids.shape[1]
   with torch.inference_mode():result=self.model.generate(**inputs,max_new_tokens=2048,do_sample=False)
   tokens=result[:,length:].cpu().tolist();raw=self.processor.batch_decode(result[:,length:],skip_special_tokens=True);assert self._bridge and self._vision_calls==self._connector_calls==1 and len(self._steps)==len(tokens[0])<=2048
   for name,count in self._counts.items():assert count==(1 if '.vision_tower.' in name else len(self._steps)),(name,count,len(self._steps))
   npz=home/'actual_visual_positions_masks_full_vocab.npz';np.savez_compressed(npz,**{k:v.reshape(-1).view(torch.uint8).numpy() for k,v in self._arrays.items()});record.update(steps=self._steps,tensor_records={k:tr(v) for k,v in self._arrays.items()},archive=str(npz),archive_sha256=sha(npz),actual_visual_injection_verified=True,actual_vision_calls=1,actual_connector_calls=1,native_activation_call_counts=self._counts,generated_padded_token_ids=tokens,raw_outputs=[s.strip() for s in raw],elapsed_seconds=time.monotonic()-start)
   (home/'batch_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));rows=[]
   for i,item in enumerate(items):
    from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
    eos=self.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos);stop=next((j+1 for j,t in enumerate(tokens[i]) if t in eos),len(tokens[i]));active=tokens[i][:stop];truncated=stop>=2048 and not any(t in eos for t in tokens[i]);raw=record['raw_outputs'][i];entry={'attempt':item['index']+1,'index':item['index'],'prompt':item['original_prompt'],'actual_original_model_prompt':item['model_prompt'],'media_path':item['media_path'],'source_image_sha256':self.p['reference_first200_image_sha256'][str(pathlib.Path(item['media_path']).resolve())],'group':self.group_calls,'group_batch_size':len(items),'group_slot':i,'group_execution':str(home/'batch_execution.json'),'group_execution_sha256':sha(home/'batch_execution.json'),'generation_token_ids':tokens[i],'active_generation_token_ids':active,'generation_count':stop,'first_EOS_index':next((j for j,t in enumerate(tokens[i]) if t in eos),None),'raw_output':raw,'empty':not raw,'invalid':explicit_choice(raw) is None,'truncated':truncated,'error':None};rows.append(entry);self.details[(item['original_prompt'].strip(),item['media_path'])]=entry
   with self.activation_trace.open('a') as f:f.write(json.dumps({'group':self.group_calls,'batch_size':len(items),'decoder_forwards':len(self._steps),'module_call_counts':self._counts,'device':str(self.device),'dtype':'torch.bfloat16','error':None})+'\n')
   with self.trace.open('a',encoding='utf8') as f:
    for e in rows:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   log.info('Zelk12 true NPU group %s batch%s indices%s steps%s',self.group_calls,len(items),record['indices'],len(self._steps));return record['raw_outputs']
  except Exception:
   record.update(error=traceback.format_exc(),completed_steps=self._steps);(home/'failed_batch_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));log.error(record['error']);raise
  finally:self._reset()
 def batch_predict(self,batch):
  lookup={(r['actual_original_model_prompt_after_instruction'],r['media_path']):r for r in self.p['input_identity_records']};result=[]
  for prompt,media in batch:
   key=(prompt,str(pathlib.Path(media).resolve()));assert key in lookup;row=lookup[key]
   item={'index':row['index'],'original_prompt':row['original_prompt'],'model_prompt':prompt,'media_path':media,'result_slot':0}
   result.append(self._generate_group([item])[0])
  return result
 def predict(self,prompt,media_path=None):return self.batch_predict([(prompt,media_path)])[0]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

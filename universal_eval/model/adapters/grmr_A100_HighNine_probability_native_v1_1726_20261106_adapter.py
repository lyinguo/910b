"""Own GRMR actual-A100 HighNine probability protocol: BF16 textSDPA/visionEager full forward."""
import os,pathlib,json,hashlib,datetime,logging,time,traceback,contextlib,math,zipfile
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.GRMR_A100_probability_1726')

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

def save_byte_npz(path,arrays):
 # Lossless evidence I/O only; every key/dtype/shape/byte is independently verified.
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for name,value in arrays.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,value,allow_pickle=False)

class GRMRActualA100HighNineProbability1726Adapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',protocol=None,**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256
  assert protocol=={'model_id':self.p['model_id'],'protocol_file_sha256':protocol_sha256}
  self.devices=list(npu_devices or [1]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('GRMR_actual_A100_HighNine_probability_1726_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
  (self.evidence/'full_vocab_and_visual').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.activation_trace=self.evidence/'native_activation_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Gemma3ForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers/torch_npu original Gemma3 BF16 three-variant first-token mc_score with FP32 full-vocab log_softmax'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoProcessor,Gemma3ForConditionalGeneration
  from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name=='qingy2024/GRMR-V3-G4B' and p['original_runtime_dtype']=='BF16'
  for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
  for src,digest in dict(p['runtime_sha256'],**p['historical_sources'],**p['evidence_sha256']).items():assert sha(src)==digest
  assert p['original_score_target']=='letter' and p['original_score_tail']=='\n\nAnswer:' and p['original_algorithm']=='mc_score' and p['original_algorithm_batch_size']==p['actual_inference_batch_size']==1
  assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.architectures==['Gemma3ForConditionalGeneration'] and not getattr(cfg,'quantization_config',None) and cfg.tie_word_embeddings and cfg.text_config.use_cache and cfg.text_config.vocab_size==262208 and cfg.mm_tokens_per_image==256
  assert cfg.text_config.hidden_size==2560 and cfg.text_config.num_hidden_layers==34 and cfg.text_config.num_attention_heads==8 and cfg.text_config.num_key_value_heads==4 and cfg.text_config.head_dim==256 and cfg.vision_config.hidden_size==1152 and cfg.vision_config.image_size==896
  cfg.dtype=torch.bfloat16;cfg.text_config.dtype=torch.bfloat16;cfg.vision_config.dtype=torch.bfloat16;cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='eager'
  with torch.device('meta'):self.model=Gemma3ForConditionalGeneration(cfg)
  assert cfg.text_config._attn_implementation=='sdpa' and cfg.vision_config._attn_implementation=='eager';targets=dict(self.model.named_parameters(remove_duplicate=False));assert len(dict(self.model.named_parameters()))==883
  expected={r['source_key']:r for r in p['source_tensor_records']};records=[];seen=set();filled={}
  for shard,header in p['weight_headers'].items():
   assert pathlib.Path(shard).name==shard and (wp/shard).stat().st_size==header['size'] and sha(wp/shard)==header['file_sha256']
   with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
    for key in f.keys():
     assert key not in seen and key in expected;name=mapped(key);r=expected[key];source=f.get_tensor(key);cast=source.bfloat16()
     assert name==r['target_key'] and name in targets and list(source.shape)==r['shape']==list(targets[name].shape) and str(source.dtype)==r['source_dtype'] and fp(source)==r['source_sha256'] and fp(cast)==r['original_BF16_cast_sha256'] and torch.isfinite(cast).all()
     target=cast.to(self.device);actual=fp(target);assert actual==r['original_BF16_cast_sha256'];parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));seen.add(key);filled[name]=actual;records.append(dict(r,actual_NPU_sha256=actual,runtime_dtype='torch.bfloat16',device=str(target.device)))
   log.info('Gemma3 original score %s loaded %s %s/883',self.model_name,shard,len(seen))
  assert seen==set(expected) and len(seen)==883
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
  assert type(self.processor).__name__=='Gemma3Processor' and json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['own_image_processor'] and self.processor.image_seq_length==256 and self.processor.chat_template==p['own_chat_template']
  assert not p['effective_original_chat_template_used'] and (not p['original_use_chat_template'] or self.processor.chat_template is None)
  self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['GELUTanh','GELUActivation','PytorchGELUTanh','SiLUActivation','NewGELUActivation','QuickGELUActivation','SiLU']};assert len(self.activation_modules)==61 and set(self.activation_modules.values())=={'GELUTanh'}
  self.input_lookup={(r['original_prompt'],r['media_path']):r for r in p['all200_independent_input_records']};assert len(self.input_lookup)==200
  self._reset();self._install_observers()
  values=torch.linspace(-8,8,20001,dtype=torch.float32).bfloat16();arrays={'input_BF16_bytes':values.view(torch.uint8).numpy()};probe={}
  for approximate in ['none','tanh']:
   native=torch.nn.functional.gelu(values.to(self.device),approximate=approximate).cpu();cpu=torch.nn.functional.gelu(values,approximate=approximate);arrays['NPU_'+approximate+'_BF16_bytes']=native.view(torch.uint8).numpy();arrays['CPU_'+approximate+'_BF16_bytes']=cpu.view(torch.uint8).numpy();probe[approximate]={'max_abs_CPU_same_declared_GELU':float((native-cpu).abs().max())}
  pq=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(pq,**arrays);self.probe=dict(probe,path=str(pq),sha256=sha(pq),CPU_threads=torch.get_num_threads(),device=str(self.device),diagnostic_only=True)
  loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir();(snapshot/'original_protocol_review.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
  for src in set(p['runtime_sha256'])|set(p['historical_sources']):
   q=pathlib.Path(src);dest=snapshot/q.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(q.read_bytes())
  import sys
  if '--config' in sys.argv:
   actual=pathlib.Path(sys.argv[sys.argv.index('--config')+1]).resolve();assert actual.is_relative_to('/root/projects/comperror/configs');dest=snapshot/actual.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(actual.read_bytes())
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'GRMR_actual_A100_HighNine_three_variant_probability_serial1','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'source_dtype_counts':p['source_dtype_counts'],'runtime_dtype':'BF16','weight_tensor_count':883,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(loading),'per_sample_evidence':str(self.trace),'activation_execution_trace':str(self.activation_trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':self.probe,'activation_modules':self.activation_modules,'original_tied_output_head_preserved':True,'source_underflow_total':p['source_to_BF16_underflow_total'],'source_BF16_roundtrip_changed_elements':p['source_BF16_roundtrip_changed_elements'],'native_attention_implementation':'sdpa','native_vision_attention_implementation':'eager','actual_A100_attention':{'text':'sdpa','vision':'eager'},'original_cache':True,'original_algorithm_batch_size':1,'actual_inference_batch_size':1,'original_score_target':'letter','original_score_tail':'\n\nAnswer:','original_use_chat_template':p['original_use_chat_template'],'effective_original_chat_template_used':False,'image_seq_len':256,'model_vocab_size':262208,'image_processor':self.processor.image_processor.to_dict(),'text_rotary_buffers':{n:tr(v) for n,v in text.rotary_emb.named_buffers()},'original_embedding_scale':tr(text.embed_tokens.embed_scale),'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'precision_disclosure':'Own GRMR883 originalBF16 source/cast/actualNPU fullSHA, actual A100 HighNine forcedBF16 textSDPA+visionEager and native34cache. Own tied head and complete262208 vocabulary, original full-position head GEMM, FP32 full-vocab log_softmax, first-token maximum over letter/space-letter/newline-letter and stable candidate order. No generation or EOS/length stop: max2048 is unused original metadata, not a reduced decoding budget. Original literal completion, RGB/896BILINEAR/256soft/singleBOS2; own chat template unused and not changed. All learned vision/projector/text/head run on assigned NPU BF16, HF32off; original F32 RoPE and BF16 embedding scale. CPU/FP32 are diagnostics only; native differences retained, no FP32 official score, hardware bitwise equality or native low-bit claim.'}
 def _reset(self):
  self._arrays={};self._counts={n:0 for n in getattr(self,'activation_modules',{})};self._vision_calls=0;self._connector_calls=0;self._text_calls=0;self._head_calls=0;self._features=None;self._attention=[];self._cache=None;self._head_shapes=None
 def _store(self,name,tensor):
  assert tensor.device==self.device;self._arrays[name]=tensor.detach().cpu().contiguous().clone()
 def _install_observers(self):
  cfg=self.model.config;vision=self.model.model.vision_tower;text=self.model.model.language_model
  def pixels(module,args,kw):self._store('actual_vision_pixels',kw.get('pixel_values',args[0] if args else None))
  def vision_post(module,args,out):self._store('actual_encoder',out.last_hidden_state);self._vision_calls+=1
  def projection(module,args,out):self._features=out;self._store('actual_connector',out);self._connector_calls+=1
  def positions(module,args):self._store('actual_vision_positions',args[0])
  def language(module,args,kw):
   x=kw['inputs_embeds'];mask=self._active_ids==cfg.image_token_id;assert self._features is not None and torch.equal(x[mask],self._features.reshape(-1,cfg.text_config.hidden_size));original=module.embed_tokens(self._active_ids);assert torch.equal(x[~mask],original[~mask]);self._store('actual_text_inputs_embeds',x);self._store('actual_injected_visual',x[mask]);self._text_calls+=1
  def rope(module,args,kw,out):
   lt=kw.get('layer_type',args[2] if len(args)>2 else None);pos=kw.get('position_ids',args[1] if len(args)>1 else None);assert lt in ['full_attention','sliding_attention'];self._store('actual_text_positions',pos);self._store('actual_'+lt+'_cos',out[0]);self._store('actual_'+lt+'_sin',out[1])
  def attention(i):
   def observe(module,args,kw):
    mask=kw['attention_mask'];name='actual_'+module.layer_type+'_mask';assert mask is not None and mask.dtype in [torch.bool,torch.bfloat16,torch.float32];record=tr(mask)
    if name in self._arrays:assert record==tr(self._arrays[name])
    else:self._store(name,mask)
    self._attention.append({'layer':i,'layer_type':module.layer_type,'mask':record,'position_ids':kw['position_ids'].cpu().tolist(),'scaling':module.scaling,'sliding_window':module.sliding_window})
   return observe
  def head_pre(module,args):self._head_input_shape=list(args[0].shape);assert args[0].shape[:2]==self._active_ids.shape
  def head_post(module,args,out):self._head_shapes={'input':self._head_input_shape,'output':list(out.shape)};assert out.shape==(*self._active_ids.shape,262208);self._head_calls+=1
  def activation(name):
   def observe(module,args,out):assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   return observe
  vision.register_forward_pre_hook(pixels,with_kwargs=True);vision.register_forward_hook(vision_post);vision.embeddings.position_embedding.register_forward_pre_hook(positions);self.model.model.multi_modal_projector.register_forward_hook(projection);text.register_forward_pre_hook(language,with_kwargs=True);text.rotary_emb.register_forward_hook(rope,with_kwargs=True)
  for i,layer in enumerate(text.layers):layer.self_attn.register_forward_pre_hook(attention(i),with_kwargs=True)
  self.model.lm_head.register_forward_pre_hook(head_pre);self.model.lm_head.register_forward_hook(head_post)
  for name,module in self.model.named_modules():
   if name in self.activation_modules:module.register_forward_hook(activation(name))
 def score_choice(self,prompt,options,media_path=None,image=None):
  self.calls+=1;self._reset();e={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None};start=time.monotonic()
  try:
   assert image is None and media_path;media=pathlib.Path(media_path).resolve();assert str(media) in self.p['reference_first200_image_sha256'] and sha(media)==self.p['reference_first200_image_sha256'][str(media)]
   key=(prompt,str(media));assert key in self.input_lookup;identity=self.input_lookup[key];e['index']=identity['index']
   with Image.open(media) as image:image=image.convert('RGB')
   rendered=prompt.rstrip()+self.p['original_score_tail']
   if '<start_of_image>' not in rendered:rendered='<start_of_image>\n'+rendered
   prepared=self.processor(text=rendered,images=[image],return_tensors='pt',padding=True);assert set(prepared)=={'pixel_values','input_ids','attention_mask','token_type_ids'} and (prepared['input_ids']==262144).sum()==256
   assert rendered==identity['original_model_completion_text'] and prepared['input_ids'].tolist()==[identity['input_ids']] and fp(prepared['pixel_values'].bfloat16())==identity['CPU_inputs']['pixel_values']['SHA256'] and list(options)==identity['options'];assert prepared['input_ids'][0,0].item()==2 and int((prepared['input_ids']==2).sum())==1
   e.update(rendered_text=rendered,CPU_inputs={k:tr(v) for k,v in prepared.items()},input_ids=prepared['input_ids'].tolist(),attention_mask=prepared['attention_mask'].tolist(),token_type_ids=prepared['token_type_ids'].tolist(),source_image_sha256=sha(media))
   inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()};self._active_ids=inputs['input_ids'];length=self._active_ids.shape[-1]
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
       out=self.model(**inputs);torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True
    else:out=self.model(**inputs)
    assert out.logits.dtype==torch.bfloat16 and out.logits.shape==(1,length,262208) and bool(torch.isfinite(out.logits).all());last=out.logits[0,-1];lp=torch.log_softmax(last.float(),dim=-1);assert lp.dtype==torch.float32 and torch.isfinite(lp).all()
   scores={};candidate_ids={};variant_tokens={}
   for opt in options:
    letter=str(opt).strip().upper();best=None;variant_tokens[letter]={};candidate_ids[letter]=[]
    for variant in [letter,' '+letter,'\n'+letter]:
     ids=self.tokenizer.encode(variant,add_special_tokens=False);assert ids and 0<=ids[0]<262208;variant_tokens[letter][variant]=ids;candidate_ids[letter].append(ids[0]);value=float(lp[ids[0]].item())
     if best is None or value>best:best=value
    scores[letter]=best
   assert scores and self._vision_calls==self._connector_calls==self._text_calls==self._head_calls==1 and all(v==1 for v in self._counts.values()) and len(self._attention)==34
   cache=out.past_key_values;assert cache is not None and cache.get_seq_length()==length;cache_records=[]
   for i,layer in enumerate(cache.layers):
    assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.bfloat16
    self._store('actual_cache_%02d_key'%i,layer.keys);self._store('actual_cache_%02d_value'%i,layer.values)
    cache_records.append({'key_sha256':fp(layer.keys),'value_sha256':fp(layer.values),'slot':i,'class':type(layer).__name__,'key_shape':list(layer.keys.shape),'value_shape':list(layer.values.shape),'device':str(layer.keys.device),'dtype':str(layer.keys.dtype),'cumulative_length':int(layer.get_seq_length())})
   assert len(cache_records)==34;self._store('actual_last_logits',last);self._store('actual_FP32_log_softmax',lp);archive=self.evidence/'full_vocab_and_visual'/('%06d.npz'%(self.calls-1));save_byte_npz(archive,{k:v.reshape(-1).view(torch.uint8).numpy() for k,v in self._arrays.items()})
   best=max(scores,key=scores.get);e.update(raw_output=best,selected_candidate=best,original_mc_score_raw_output=None,raw_output_represents='selected_probability_candidate_not_generated_text',candidate_scores=scores,candidate_first_token_ids=candidate_ids,full_candidate_variant_token_ids=variant_tokens,original_full_forward_logit_shape=list(out.logits.shape),actual_head_forward_shapes=self._head_shapes,full_vocab_size=262208,original_FP32_log_softmax=True,no_generation=True,generation_token_ids=[],generation_count=0,truncated=False,empty=False,invalid=False,actual_vision_calls=1,actual_connector_calls=1,actual_text_model_calls=1,actual_head_calls=1,actual_visual_injection_verified=True,full_vocab_and_visual_archive=str(archive),full_vocab_and_visual_archive_sha256=sha(archive),tensor_records={k:tr(v) for k,v in self._arrays.items()},native_activation_call_counts=dict(self._counts),actual_attention_layers=self._attention,cache_device_and_shapes=cache_records,final_cache_length=int(cache.get_seq_length()))
   with self.activation_trace.open('a',encoding='utf8') as f:f.write(json.dumps({'attempt':self.calls,'device':str(self.device),'dtype':'torch.bfloat16','module_call_counts':dict(self._counts),'error':None})+'\n')
   log.info('Gemma3 original candidate NPU sample %s options=%s selected=%s',self.calls,scores,best);return scores
  except Exception:e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   e['elapsed_seconds']=time.monotonic()-start;self.details[(prompt.strip(),media_path)]=e
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._reset()
 def predict(self,prompt,media_path=None):raise RuntimeError('Original protocol is score_choice, not free generation')
 def batch_predict(self,batch):raise RuntimeError('Original mc_score performs serial score_choice')
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

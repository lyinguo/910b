"""Actual A100 Nougat FP16/SDPA caption protocol, with original native computation and evidence."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import contextlib,datetime,functools,hashlib,inspect,json,logging,pathlib,sys,time,traceback
import numpy as np
import torch,torch_npu
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.Nougat_A100_FP16_2116')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def rec(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}

class NougatA100FP16Native2116Adapter(IModel):
 def __init__(self,protocol=None,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  torch.set_num_threads(4);self.p=dict(protocol);self.devices=list(npu_devices or [3]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False;self._profile_active=False;self._phase='idle'
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('Nougat_A100_FP16_2116_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'visual_features').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.gelu_trace=self.evidence/'native_GELU_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'VisionEncoderDecoderModel'
 @property
 def inference_framework(self):return 'Native PyTorch/torch_npu actual A100 FP16 Nougat Swin eager / MBart SDPA'
 def download(self,*a,**kwargs):raise RuntimeError('No model/data downloads')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,VisionEncoderDecoderModel,NougatProcessor,GenerationConfig
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and wp.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and self.model_name=='facebook/nougat-small'
  assert p['source_dtype']=='F32' and p['execution_dtype']=='FP16' and p['max_new_tokens']==2048 and p['extract_answer'] is False and p['truncate_first_sentence'] is False
  for name,h in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==h
  for path,h in p['runtime_sha256'].items():assert sha(path)==h,path
  root=pathlib.Path(__file__).resolve().parents[3]
  for rel,h in p['project_source_sha256'].items():
   name=pathlib.Path(rel);assert not name.is_absolute() and '..' not in name.parts and sha(root/name)==h,rel
  for name in ['reference_report','reference_log','dataset_path','CPU200_reference','source_cast_manifest','original_A100_real3_diagnostic']:assert sha(p[name])==p[name+'_sha256'],name
  assert p['dataset_path_sha256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b'
  assert sha(p['native_GELU_method_note'])==p['native_GELU_method_note_sha256']=='0842aa75ef490056d6ccf8035fbb63684a98d709b193f59dcb2dba10b21d41d6'
  self.cpu=json.loads(pathlib.Path(p['CPU200_reference']).read_text());self.cast={r['source_key']:r for r in json.loads(pathlib.Path(p['source_cast_manifest']).read_text())};assert len(self.cpu)==200 and len(self.cast)==484
  self.aa=json.loads(pathlib.Path(p['original_A100_real3_diagnostic']).read_text());assert self.aa['all3_historical_raw_equal'] and self.aa['original_caption_raw_equals_prediction_count']==200
  torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for name in ['conv','matmul']:getattr(torch.npu,name).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);cfg._attn_implementation=p['actual_A100_attention']['root'];cfg.encoder._attn_implementation=p['actual_A100_attention']['encoder'];cfg.decoder._attn_implementation=p['actual_A100_attention']['decoder']
  assert not cfg.tie_word_embeddings and not cfg.decoder.tie_word_embeddings and cfg.encoder.model_type=='donut-swin' and cfg.decoder.model_type=='mbart'
  with torch.device('meta'):self.model=VisionEncoderDecoderModel(cfg)
  params=dict(self.model.named_parameters());buffers=dict(self.model.named_buffers());expected=dict(params,**buffers);assert len(params)==464 and len(buffers)==20;loading=[];loaded=set()
  for name,w in p['weights'].items():
   f=wp/name;assert f.stat().st_size==w['size'] and sha(f)==w['sha256']
   with safe_open(f,framework='pt',device='cpu') as handle:
    for key in handle.keys():
     source=handle.get_tensor(key);c=self.cast[key];assert key not in loaded and list(source.shape)==c['shape']==list(expected[key].shape) and str(source.dtype)==c['source_dtype'] and fp(source)==c['source_sha256']
     value=source.half() if key in params else source;assert str(value.dtype)==c['execution_dtype'] and fp(value)==c['original_FP16_sha256'];target=value.to(self.device).contiguous();assert fp(target)==c['original_FP16_sha256'];parent,leaf=key.rsplit('.',1)
     if key in params:setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False))
     else:setattr(self.model.get_submodule(parent),leaf,target)
     assert (self.aa['actual_parameter_sha256'][key] if key in params else self.aa['actual_buffers'][key]['sha256'])==fp(target)
     loading.append(dict(c,device=str(target.device),actual_NPU_sha256=fp(target)));loaded.add(key)
  assert loaded==set(expected) and all(t.device==self.device and not t.is_meta for t in self.model.state_dict().values());assert self.model.decoder.lm_head.weight is not self.model.decoder.model.decoder.embed_tokens.weight
  self.model.eval();self.processor=NougatProcessor.from_pretrained(wp,local_files_only=True);self.tok=self.processor.tokenizer;assert json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['image_processor']
  self.model.generation_config=GenerationConfig.from_pretrained(wp,local_files_only=True);self.gen,_=self.model._prepare_generation_config(None,max_new_tokens=2048,do_sample=False,temperature=0.0);assert self.gen.to_dict()==p['effective_generation_config'] and self.gen.use_cache is True and self.gen.num_beams==1 and self.gen.forced_eos_token_id==2 and self.gen.bos_token_id==0
  assert {'root':cfg._attn_implementation,'encoder':cfg.encoder._attn_implementation,'decoder':cfg.decoder._attn_implementation}==p['actual_A100_attention']=={'root':'sdpa','encoder':'eager','decoder':'sdpa'}
  sample=torch.linspace(-8.,8.,20001).half();probe=torch.nn.functional.gelu(sample.to(self.device),approximate='none').cpu();erf=torch.nn.functional.gelu(sample,approximate='none');tanh=torch.nn.functional.gelu(sample,approximate='tanh');probe_path=self.evidence/'native_FP16_GELU_probe.npz';np.savez_compressed(probe_path,input_FP16_bits=sample.view(torch.uint8).numpy(),NPU_FP16_bits=probe.view(torch.uint8).numpy(),CPU_erf_FP16_bits=erf.view(torch.uint8).numpy(),CPU_tanh_diagnostic_FP16_bits=tanh.view(torch.uint8).numpy())
  self.probe={'path':str(probe_path),'sha256':sha(probe_path),'dtype':'torch.float16','device':str(self.device),'CPU_threads':torch.get_num_threads(),'max_abs_CPU_erf':float((probe-erf).abs().max()),'max_abs_CPU_tanh_diagnostic':float((probe-tanh).abs().max()),'NPU_call':"torch.nn.functional.gelu(approximate='none')"}
  def encoder_out(m,args,value):
   if self._phase=='generate':self._vision=value.last_hidden_state;self._vision_calls+=1
  def positional(m,args,kw,value):
   if self._phase!='generate':return
   ids=args[0] if args else kw['input_ids'];past=args[1] if len(args)>1 else kw.get('past_key_values_length',0);positions=kw.get('position_ids');assert ids.numel()==1 and int(past)==len(self._positions) and positions is not None and value.dtype==torch.float16 and value.device==self.device
   self._positions.append({'input_ids':ids.detach().cpu().tolist(),'past_length':int(past),'position_ids':positions.detach().cpu().tolist(),'actual_learned_position_output':rec(value),'device':str(value.device)})
  def mask(m,args,kw):
   if self._phase=='generate':value=kw.get('attention_mask');self._masks.append(None if value is None else rec(value));assert value is None or value.device==self.device
  def cross(index):
   def hook(m,args,kw):
    if self._phase!='generate':return
    value=kw.get('key_value_states');assert value is not None and self._vision is not None and value.data_ptr()==self._vision.data_ptr() and value.device==self.device;self._cross_calls[index]+=1
    if index==0 and self._cross is None:self._cross=value.detach().clone();assert torch.equal(self._cross,self._vision)
   return hook
  def logits(m,args,kw,value):
   if self._phase!='generate':return
   x=value.logits[:,-1,:];assert x.shape==(1,50000) and x.device==self.device and x.dtype==torch.float16 and torch.isfinite(x).all();tid=int(x.float().argmax(-1).item());self._logits.append({'step':len(self._logits),'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'all_finite':True,'actual_full50000_argmax_id':tid})
  self.model.encoder.register_forward_hook(encoder_out);dec=self.model.decoder.model.decoder;dec.embed_positions.register_forward_hook(positional,with_kwargs=True);dec.layers[0].self_attn.register_forward_pre_hook(mask,with_kwargs=True)
  for i,layer in enumerate(dec.layers):layer.encoder_attn.register_forward_pre_hook(cross(i),with_kwargs=True)
  self.model.register_forward_hook(logits,with_kwargs=True)
  gelus={n:m for n,m in self.model.named_modules() if type(m).__name__=='GELUActivation'};assert len(gelus)==24;self._gelu_names=list(gelus)
  def count(name):
   def hook(m,args,value):
    if self._phase=='generate':assert value.dtype==torch.float16 and value.device==self.device;self._gelu_counts[name]+=1
   return hook
  for n,m in gelus.items():m.register_forward_hook(count(n))
  original_forward=self.model.forward;signature=inspect.signature(original_forward)
  @functools.wraps(original_forward)
  def observed_forward(*args,**kw):
   value=original_forward(*args,**kw)
   if self._profile_active:
    torch.npu.synchronize(self.device);self._prof.__exit__(None,None,None);self._profile_active=False;self._prof.export_chrome_trace(str(self.evidence/'npu_trace.json'));self._cwd.__exit__(None,None,None);self.profile_done=True
   return value
  self.model.forward=observed_forward;assert inspect.signature(self.model.forward)==signature
  weights=self.evidence/'weight_loading.json';weights.write_text(json.dumps(loading,indent=2)+'\n')
  snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
  for rel,h in p['project_source_sha256'].items():
   source=root/rel;dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());assert sha(dest)==h
  if '--config' in sys.argv:
   source=pathlib.Path(sys.argv[sys.argv.index('--config')+1]).resolve();assert source.is_relative_to(root/'configs');dest=snapshot/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
  (snapshot/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'actual_A100_Nougat_FP16_native_SDPA_original_fullraw_caption1459','protocol_review':p,'source_dtype':'F32','execution_dtype':'FP16','all_expected_parameters_loaded':True,'weight_tensor_count':484,'learned_parameter_count':464,'original_buffer_count':20,'untied_original_output_head_preserved':True,'weight_loading_evidence':str(weights),'source_cast_changed':sum(v['roundtrip_changed_elements'] for v in loading),'source_cast_underflow':sum(v['nonzero_underflow'] for v in loading),'actual_attention':p['actual_A100_attention'],'effective_generation_config':self.gen.to_dict(),'effective_image_processor':self.processor.image_processor.to_dict(),'HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ['conv','matmul']},'per_sample_evidence':str(self.trace),'native_GELU_execution_trace':str(self.gelu_trace),'native_GELU_probe':self.probe,'profiler_trace':str(self.evidence/'npu_trace.json'),'profiler_scope':'First actual encoder and unmodified first native decoder prefill within official generate only; not the full generation or all200. All generated steps/full50000 argmax, positions, cache and all24 activations independently observed.','native_cache_allocated_slots':cfg.decoder.num_hidden_layers,'actual_decoder_layers':cfg.decoder.decoder_layers,'native_allocator_setting':'expandable_segments:True','CPU_threads_for_probe':torch.get_num_threads(),'precision_disclosure':'All464 own F32 learned tensors follow the actual A100 FP16 conversion and full CPU-cast/A100-CUDA/NPU SHA equality;20 own I64 position buffers remain exact. Source FP16 underflow and native CANN GELU/SDPA differences retained. Learned layers and native cache run NPU FP16, HF32off. CPU and A100 controls are diagnostics only, not substituted official outputs/scores or native lowbit acceleration. Original2048/greedy/cache/forcedEOS2, full raw and original false extraction/no firstsentence preserved.','module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
 def reset(self):
  self._vision=None;self._cross=None;self._vision_calls=0;self._cross_calls=[0]*4;self._positions=[];self._masks=[];self._logits=[];self._gelu_counts={n:0 for n in self._gelu_names}
 def predict(self,prompt,media_path=None):
  self.calls+=1;self.reset();start=time.monotonic();e={'attempt':self.calls,'original_index':self.calls-1,'prompt':prompt,'media_path':media_path,'error':None}
  try:
   i=self.calls-1;ref=self.cpu[i];media=pathlib.Path(media_path).resolve();assert prompt==ref['prompt'] and str(media)==ref['resolved_media_path'] and sha(media)==ref['source_image_sha256']
   with Image.open(media) as im:image=im.convert('RGB')
   prepared=self.processor.image_processor(images=image,return_tensors='pt')['pixel_values'];assert rec(prepared)==ref['CPU_pixels'];pixels=prepared.half().to(self.device);assert rec(pixels)==ref['original_FP16_pixels']
   e.update(source_image_sha256=sha(media),CPU_pixels=rec(prepared),NPU_FP16_pixels=rec(pixels),model_receives_no_prompt_or_reference=True,allocator_before=torch.npu.memory_stats(self.device).get('num_ooms',0));self._phase='generate'
   if not self.profile_done:
    self._cwd=contextlib.chdir(self.evidence);self._cwd.__enter__();self._prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False);self._prof.__enter__();self._profile_active=True
   with torch.inference_mode():out=self.model.generate(pixel_values=pixels,generation_config=self.gen,return_dict_in_generate=True,output_scores=False)
   seq=out.sequences[0].cpu().tolist();count=len(seq)-1;assert seq[0]==0 and 1<=count<=2048 and seq[-1]==2;raw=self.processor.batch_decode([seq],skip_special_tokens=True)[0].strip();budget=count==2048
   assert len(self._positions)==len(self._masks)==len(self._logits)==count and self._vision_calls==1 and self._vision.shape==(1,588,1024) and self._cross_calls==[count]*4 and torch.equal(self._vision,self._cross)
   for n,v in enumerate(self._logits):assert seq[n+1]==(2 if n==2047 else v['actual_full50000_argmax_id'])
   cache=out.past_key_values;assert cache.get_seq_length()==count;cache_info={}
   for name in ['self_attention_cache','cross_attention_cache']:
    c=getattr(cache,name);assert len(c.layers)==self.model.config.decoder.num_hidden_layers;cache_info[name]=[]
    for j,layer in enumerate(c.layers):
     if j>=4:assert layer.keys is None and layer.values is None;cache_info[name].append({'slot':j,'initialized':False});continue
     assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.float16;cache_info[name].append({'slot':j,'initialized':True,'key_shape':list(layer.keys.shape),'value_shape':list(layer.values.shape),'device':str(layer.keys.device),'dtype':str(layer.keys.dtype)})
   archive=self.evidence/'visual_features'/('%06d.npz'%i);vision=self._vision.detach().cpu();crossv=self._cross.detach().cpu();assert torch.isfinite(vision).all();np.savez_compressed(archive,encoder_FP16_bits=vision.view(torch.uint8).numpy(),actual_cross_FP16_bits=crossv.view(torch.uint8).numpy())
   gelu={'index':i,'generation_token_count':count,'device':str(self.device),'dtype':'torch.float16','native_call':"GELUActivation torch.nn.functional.gelu(approximate='none')",'module_call_counts':dict(self._gelu_counts),'error':None}
   with self.gelu_trace.open('a',encoding='utf8') as f:f.write(json.dumps(gelu)+'\n')
   e.update(raw_output=raw,generation_sequence_ids=seq,generation_token_count=count,truncated=budget,forced_eos_at_original_budget=budget,text_positions=self._positions,self_attention_masks=self._masks,full_vocabulary_logits_execution=self._logits,visual_feature_archive=str(archive),visual_feature_archive_sha256=sha(archive),encoder_output=rec(vision),actual_decoder_cross_attention_input=rec(crossv),vision_forward_calls=self._vision_calls,decoder_cross_attention_calls=self._cross_calls,final_cache_length=count,cache_device_and_shapes=cache_info,native_GELU_call_counts=dict(self._gelu_counts),allocator_after=torch.npu.memory_stats(self.device).get('num_ooms',0));assert e['allocator_after']==e['allocator_before']==0
   log.info('Nougat A100 FP16 NPU index=%s tokens=%s truncated=%s raw=%r',i,count,budget,raw[:180]);return raw
  except Exception:e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   if self._profile_active:
    try:self._prof.__exit__(None,None,None)
    finally:self._profile_active=False;self._cwd.__exit__(None,None,None)
   self._phase='idle';e['elapsed_seconds']=time.monotonic()-start;self.details[(prompt.strip(),media_path)]=e
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._vision=None;self._cross=None
 def batch_predict(self,batch):return [self.predict(prompt,path) for prompt,path in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

"""Own PaliGemma2-3B-Docci-448 on native NPU, preserving the actual A100 FP16 caption contract."""
import contextlib,datetime,hashlib,json,logging,os,pathlib,re,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import functools,inspect
import torch,torch_npu
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.Docci3b_A100_FP16_1813')

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
def target_key(key):
 for a,b in [('vision_tower.vision_model.','model.vision_tower.'),('vision_tower.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.'),('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.')]:
  if key.startswith(a):return b+key[len(a):]
 return key
def clean_prompt(prompt):
 for pattern in (r'\n\n(?:请仔细分析图像和题目[^\n]*|Answer the question by providing only the option letter[^\n]*)\s*$',r'\n\nProvide only the final answer\.\s*$'):prompt=re.sub(pattern,'',prompt)
 return prompt

class Docci3bA100FP16PrefillProfileV4_1813Adapter(IModel):
 def __init__(self,npu_devices=None,protocol=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  self.devices=list(npu_devices or []);assert len(self.devices)==1
  self.device=torch.device('npu:'+str(self.devices[0]));self.dtype=torch.float16;self.protocol=dict(protocol or {})
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
  self.evidence=pathlib.Path(evidence_dir).resolve()/('Docci3b_A100_FP16_1813_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True)
  (self.evidence/'visual_features').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.steps=self.evidence/'generation_steps.jsonl'
  self.details={};self.calls=0;self.profile_done=False;self.lineage={}
 @property
 def model_name(self):return getattr(self,'_name','unloaded')
 @property
 def model_type(self):return 'PaliGemmaForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers PaliGemma2 FP16 / PyTorch + torch_npu; original A100 SDPA'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')
 def load(self,model_path,**kwargs):
  from safetensors import safe_open
  from transformers import AutoConfig,AutoProcessor,GenerationConfig,PaliGemmaForConditionalGeneration
  from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
  p=self.protocol;directory=pathlib.Path(model_path).resolve();self._name=directory.parent.name+'/'+directory.name
  assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and self.model_name==p['model_id']=='google/paligemma2-3b-ft-docci-448'
  assert torch.npu.is_available();torch.npu.set_device(self.device)
  for n in ('conv','matmul'):getattr(torch.npu,n).allow_hf32=False
  for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(directory/name)==digest,name
  for path,digest in dict(p['runtime_sha256'],**p['original_source_sha256']).items():assert sha(path)==digest,path
  for name in ['CPU200_reference','source_cast_manifest','reference_report','reference_log','original_A100_real3_diagnostic']:
   assert sha(p[name])==p[name+'_sha256'],name
  self.cpu200=json.loads(pathlib.Path(p['CPU200_reference']).read_text());assert len(self.cpu200)==200
  self.by_media={v['resolved_media_path']:v for v in self.cpu200};assert len(self.by_media)==200
  self.dataset=pathlib.Path(p['dataset_path']);assert sha(self.dataset)==p['dataset_sha256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b'
  old=json.loads(pathlib.Path(p['reference_report']).read_text());pc=old['pipeline_config']
  assert pc['model']['name']==self.model_name and pc['model']['adapter'].endswith(':PaliGemmaTransformers') and pc['model']['kwargs']['max_new_tokens']==p['max_new_tokens']==1024
  assert pc['model']['kwargs']['strip_eval_instruction'] is True and pc['model']['kwargs']['vqa_prefix']==p['caption_prefix']=='answer en ' and not any(pc['model']['kwargs'].get(k) for k in ['mc_instruction','truncate_first_sentence','discriminative_mc','span_to_option'])
  assert pc['algorithm']['kwargs']['extract_answer'] is True and pc['algorithm']['batch_size']==10 and pc['task_type']=='image-to-text' and len(old['detailed_results'])==200
  text=pathlib.Path(p['reference_log']).read_text();assert 'torch.float16' in text and 'batch_size=10' in text and '提取答案=True' in text
  cfg=AutoConfig.from_pretrained(directory,local_files_only=True);assert cfg.text_config.model_type=='gemma2' and cfg.text_config.num_hidden_layers==26 and cfg.vision_config.num_hidden_layers==27 and cfg.text_config.vocab_size==257216
  with torch.device('meta'):self.model=PaliGemmaForConditionalGeneration(cfg)
  attention={n:c._attn_implementation for n,c in [('root',cfg),('text',cfg.text_config),('vision',cfg.vision_config)]}
  assert attention==p['A100_resolved_attention']=={'root':'sdpa','text':'sdpa','vision':'sdpa'}
  expected=dict(self.model.named_parameters());buffer_types={n:t.dtype for n,t in self.model.named_buffers()};buffer_types['model.language_model.embed_tokens.embed_scale']=torch.float16;filled=set();loading=[]
  reviewed={r['source_key']:r for r in json.loads(pathlib.Path(p['source_cast_manifest']).read_text())};assert len(reviewed)==727
  index=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map'];assert set(index)==set(reviewed)
  for shard in sorted(p['weights']):
   w=directory/shard;assert w.stat().st_size==p['weights'][shard]['size'] and sha(w)==p['weights'][shard]['sha256'],shard
   with safe_open(w,framework='pt',device='cpu') as handle:
    assert set(handle.keys())=={k for k,v in index.items() if v==shard}
    for key in handle.keys():
     source=handle.get_tensor(key);target=target_key(key);r=reviewed[key]
     assert target in expected and target not in filled and source.shape==expected[target].shape and source.dtype==torch.bfloat16,key
     assert fp(source)==r['source_sha256']
     rounded=source.to(self.dtype);assert bool(torch.isfinite(rounded).all()) and fp(rounded)==r['FP16_sha256']
     value=rounded.to(self.device).contiguous();assert fp(value)==r['FP16_sha256']
     parent,leaf=target.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(value,requires_grad=False));filled.add(target)
     loading.append(dict(r,target_key=target,device=str(value.device),actual_NPU_sha256=fp(value)))
   log.info('Docci3b A100 FP16 loaded %s: %s/%s complete tensor hashes',shard,len(filled),len(expected))
  assert filled==set(expected) and cfg.tie_word_embeddings
  self.model.tie_weights();assert self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight
  self.model.model.vision_tower.embeddings.position_ids=torch.arange(1024).reshape(1,-1).to(self.device)
  embedding=self.model.model.language_model.embed_tokens
  embedding.embed_scale=torch.tensor(embedding.scalar_embed_scale,dtype=buffer_types['model.language_model.embed_tokens.embed_scale']).to(self.device)
  self.model.model.language_model.rotary_emb=Gemma2RotaryEmbedding(cfg.text_config,device='cpu').to(self.device)
  assert all(v.device==self.device and v.dtype==self.dtype for v in self.model.parameters())
  assert all(not v.is_meta and v.device==self.device and v.dtype==buffer_types[n] for n,v in self.model.named_buffers())
  self.model.eval();self.processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
  assert self.processor.tokenizer.add_bos_token is False and self.processor.tokenizer.add_eos_token is False
  assert json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['effective_image_processor'] and self.processor.image_processor.__class__.__name__=='SiglipImageProcessor'
  self.model.generation_config=GenerationConfig.from_pretrained(directory,local_files_only=True)
  self.gen,_=self.model._prepare_generation_config(None,max_new_tokens=1024,do_sample=False)
  assert self.gen.to_dict()==p['effective_generation_config'] and self.gen.num_beams==1 and self.gen.use_cache and self.gen.repetition_penalty==1 and not self.gen.forced_eos_token_id
  self._observing=False;self._encoder=None;self._projector=None;self._injected=None;self._forward=[];self._argmax=[];self._cache=[]
  self.activations={}
  def activation_hook(name):
   def hook(module,args,value):self.activations[name]=self.activations.get(name,0)+1
   return hook
  for name,module in self.model.named_modules():
   if name.endswith(('.mlp.activation_fn','.mlp.act_fn')):
    self.activations[name]=0;module.register_forward_hook(activation_hook(name))
  assert len(self.activations)==53
  def vision(module,args,value):self._encoder=value.last_hidden_state
  def projector(module,args,value):self._projector=value
  self.model.model.vision_tower.register_forward_hook(vision);self.model.model.multi_modal_projector.register_forward_hook(projector)
  def pre_root(module,args,kw):
   if not self._observing:return
   ids=kw.get('input_ids',args[0] if args else None);cache=kw.get('past_key_values');past=cache.get_seq_length() if cache is not None else 0
   self._root={'past':int(past),'input_ids':ids.cpu().tolist()}
  def pre_language(module,args,kw):
   if not self._observing:return
   ids=self._root['input_ids'];q=len(ids[0]);past=self._root['past'];pos=kw['position_ids'];expected_pos=torch.arange(past+1,past+q+1).reshape(1,-1)
   assert torch.equal(pos.cpu(),expected_pos)
   row=dict(self._root,position_ids=pos.cpu().tolist(),masks={})
   masks=kw['attention_mask'];assert isinstance(masks,dict) and set(masks)=={'full_attention','sliding_attention'}
   for name,mask in masks.items():
    if mask is None:
     assert q==1;row['masks'][name]=None;continue
    assert mask.shape[:3]==(1,1,q) and mask.device==self.device
    allowed=torch.arange(mask.shape[-1])<past+q
    ref=allowed[None,None,None,:].expand(1,1,q,-1)
    actual=mask.cpu();assert torch.equal(actual==0,ref) if actual.dtype!=torch.bool else torch.equal(actual,ref)
    row['masks'][name]=record(mask)
   if past==0:
    embeds=kw['inputs_embeds'];image_positions=self._input_ids==cfg.image_token_id;self._injected=embeds[image_positions].reshape(1,1024,cfg.text_config.hidden_size)
    assert self._projector is not None and torch.equal(self._injected,self._projector) and bool(self._injected.abs().sum()>0)
   self._forward.append(row)
  def post_root(module,args,kw,value):
   if not self._observing:return
   logits=value.logits[0,-1];assert logits.numel()==257216 and bool(torch.isfinite(logits).all())
   self._argmax.append(int(logits.argmax()));cache=value.past_key_values;assert cache is not None and len(cache.layers)==26
   lengths=[]
   for layer in cache.layers:
    assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==self.dtype and layer.keys.shape==layer.values.shape
    lengths.append(int(layer.get_seq_length()))
   assert len(set(lengths))==1 and lengths[0]==self._root['past']+len(self._root['input_ids'][0])
   self._cache.append({'seen':lengths[0],'layers':26,'key_value_dtype':'torch.float16','device':str(self.device),'key_shape':list(cache.layers[0].keys.shape)})
  self.model.register_forward_pre_hook(pre_root,with_kwargs=True);self.model.model.language_model.register_forward_pre_hook(pre_language,with_kwargs=True);self.model.register_forward_hook(post_root,with_kwargs=True)
  native_forward=self.model.forward;native_signature=inspect.signature(native_forward)
  @functools.wraps(native_forward)
  def forward_with_prefill_hardware_profile(*args,**kwargs):
   if self._observing and not self.profile_done:
    assert self._root['past']==0,'Only first actual native prefill is profiled'
    with contextlib.chdir(self.evidence):
     with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False) as profile:
      value=native_forward(*args,**kwargs);torch.npu.synchronize(self.device)
     profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
    self.profile_done=True;return value
   return native_forward(*args,**kwargs)
  self.model.forward=forward_with_prefill_hardware_profile
  assert inspect.signature(self.model.forward)==native_signature
  loading_path=self.evidence/'weight_loading.json';loading_path.write_text(json.dumps(loading,indent=2))
  self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'source_dtype':'BF16','execution_dtype':'FP16','precision':'All own BF16 source tensors cast to FP16 exactly as actual A100, including disclosed rounding and underflow; all learned layers on NPU. Native SDPA and native GELU/matmul preserved, HF32 disabled. CPU diagnostics never supply answers. No low-bit acceleration or bitwise-equivalence claim.',
   'HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ('conv','matmul')},'protocol':'actual_A100_Docci3b_FP16_answer_en_caption_serial1024','protocol_review':p,'all_expected_parameters_loaded':True,'weight_tensor_count':len(loading),'weight_loading_evidence':str(loading_path),'weight_files':p['weights'],'model_resource_root':str(directory),'model_resource_sha256':p['resource_sha256'],'runtime_source_sha256':p['runtime_sha256'],'actual_attention':attention,'effective_generation_config':self.gen.to_dict(),'effective_image_processor':self.processor.image_processor.to_dict(),'source_cast_roundtrip_changed_sum':sum(r['roundtrip_changed_elements'] for r in loading),'source_cast_underflow_sum':sum(r['nonzero_underflow'] for r in loading),'actual_tied_head_embedding':self.model.lm_head.weight is embedding.weight,'head_size':257216,'image_tokens':1024,'activation_modules':list(self.activations),'per_sample_evidence':str(self.trace),'generation_step_evidence':str(self.steps),'profiler_trace':str(self.evidence/'npu_trace.json'),'profiler_scope':'Requested NPU hardware activity for first real unmodified native root prefill only; not all generation steps or all200. Every generated token/cache/position and all53 activations remain independently checked.','profiler_activities':['NPU'],'profiler_record_shapes':False,'all53_activation_and_every_full_vocab_cache_position_checks_unchanged':True,'analytic_buffers':{n:record(t) for n,t in self.model.named_buffers()},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
 def predict(self,prompt,media_path=None):
  self.calls+=1;started=time.monotonic();e={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
  try:
   media=pathlib.Path(media_path).resolve();ref=self.by_media.get(str(media));assert ref is not None and sha(media)==ref['source_image_sha256']
   clean=clean_prompt(prompt);assert clean==ref['prompt'];text=self.protocol['caption_prefix']+clean;assert text==ref['transformed_prompt']
   with Image.open(media) as im:prepared=self.processor(text=text,images=im.convert('RGB'),return_tensors='pt')
   assert {k:v.tolist() for k,v in prepared.items() if k!='pixel_values'}==ref['inputs']
   pixels=prepared['pixel_values'];assert fp(pixels)==ref['pixel_values_F32_sha256'] and fp(pixels.half())==ref['pixel_values_FP16_sha256']
   assert set(prepared)=={'input_ids','attention_mask','token_type_ids','pixel_values','labels'} and bool((prepared['labels']==-100).all())
   inputs={k:v.to(self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()}
   self._input_ids=inputs['input_ids'];self._encoder=None;self._projector=None;self._injected=None;self._forward=[];self._argmax=[];self._cache=[];self.activations={k:0 for k in self.activations};self._observing=True
   with torch.inference_mode():generated=self.model.generate(**inputs,generation_config=self.gen,return_dict_in_generate=True,output_scores=False);torch.npu.synchronize(self.device)
   self._observing=False;tokens=generated.sequences[0,self._input_ids.shape[-1]:].cpu().tolist()
   assert tokens==self._argmax and len(tokens)==len(self._forward)==len(self._cache) and len(tokens)<=1024
   assert self._encoder is not None and self._projector is not None and self._injected is not None
   for name,count in self.activations.items():assert count==(1 if 'vision_tower' in name else len(tokens)),(name,count,len(tokens))
   raw=self.processor.batch_decode(generated.sequences[:,self._input_ids.shape[-1]:],skip_special_tokens=True)[0].strip()
   archive=self.evidence/'visual_features'/('%05d.npz'%(self.calls-1))
   np.savez_compressed(archive,encoder_bits=self._encoder.detach().cpu().contiguous().view(torch.uint8).numpy(),projector_bits=self._projector.detach().cpu().contiguous().view(torch.uint8).numpy(),actual_LM_image_bits=self._injected.detach().cpu().contiguous().view(torch.uint8).numpy())
   entry_steps={'attempt':self.calls,'original_index':ref['index'],'forward':self._forward,'cache':self._cache,'actual_full_vocab_argmax_ids':self._argmax,'activation_counts':self.activations}
   with self.steps.open('a') as f:f.write(json.dumps(entry_steps)+'\n')
   e.update(original_index=ref['index'],processor_text=text,source_image_sha256=ref['source_image_sha256'],CPU_pixels=record(pixels),original_A100_FP16_pixels=record(pixels.half()),actual_NPU_pixels=record(inputs['pixel_values']),processor_inputs=ref['inputs'],raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),truncated=len(tokens)>=1024,stop_reason='length' if len(tokens)>=1024 else 'eos' if tokens and tokens[-1]==self.gen.eos_token_id else 'other',visual_feature_archive=str(archive),visual_feature_archive_sha256=sha(archive),encoder=record(self._encoder),projector=record(self._projector),actual_LM_images=record(self._injected),native_SDPA_prefix_and_cache_checked=True,actual_full_vocabulary_size=257216,actual_activation_modules=53)
   log.info('Docci3b A100 FP16 NPU sample %s index=%s tokens=%s truncated=%s raw=%r',self.calls,ref['index'],len(tokens),e['truncated'],raw[:200]);return raw
  except Exception:
   e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   self._observing=False;e['elapsed_seconds']=time.monotonic()-started;self.details[(clean_prompt(prompt).strip(),media_path)]=e
   with self.trace.open('a') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._encoder=None;self._projector=None;self._injected=None
 def batch_predict(self,batch):return [self.predict(p,m) for p,m in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((clean_prompt(prompt).strip(),media_path),{'error':'Missing actual execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

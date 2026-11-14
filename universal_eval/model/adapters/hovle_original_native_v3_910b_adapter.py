"""HoVLE-HD own8+24 architecture, original BF16 native generation and dual stream."""
import os,pathlib,json,datetime,logging,time,traceback,contextlib,collections,importlib,types,functools
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.hovle_original_runtime_v2_910b import classes,tokenizer,compatible_model_class,apply_original_runtime_patches
from universal_eval.model.adapters.hovle_native_io_910b import sha,fp,tr,pixels
log=logging.getLogger('universal_eval.model.adapters.hovle_original_native_910b')

class HoVLEOriginalNativeV3_910BAdapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256;self.devices=list(npu_devices or [13]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('HoVLE_original_native_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'samples').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.act_trace=self.evidence/'native_activation_execution.jsonl';self.progress=self.evidence/'generation_progress.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'InternVLChatModel'
 @property
 def inference_framework(self):return 'Own HoVLE native torch_npu BF16 holistic8+InternLM2_24, original chat and GenerationMixin'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')
 def load(self,model_path,**kwargs):
  from safetensors import safe_open
  from transformers import GenerationConfig
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name=='OpenGVLab/HoVLE-HD'
  for n,d in p['resource_sha256'].items():assert pathlib.Path(n).name==n and sha(wp/n)==d
  for n,d in dict(p['runtime_sha256'],**p['evidence_sha256']).items():assert sha(n)==d
  assert sha(p['input_audit'])==p['input_audit_sha256'];audit=json.loads(pathlib.Path(p['input_audit']).read_text());self.records={r['index']:r for r in audit['records']};assert sorted(self.records)==list(range(200));self.by_media={r['media_path']:r for r in self.records.values()}
  assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for k in ['conv','matmul']:getattr(torch.npu,k).allow_hf32=False
  mod=classes(wp);C=importlib.import_module(mod.__package__+'.configuration_internvl_chat').InternVLChatConfig;cfg=C.from_dict(json.loads((wp/'config.json').read_text()));assert cfg.use_llm_lora==cfg.use_backbone_lora==0 and cfg.llm_config.architectures==['InternLM2ForCausalLM'] and not cfg.llm_config.tie_word_embeddings and cfg.embedding_config.pixel_shuffle_loc=='post' and cfg.normalize_encoder_output
  cfg.embedding_config.attn_implementation=cfg.llm_config.attn_implementation='eager';cfg.embedding_config._attn_implementation=cfg.llm_config._attn_implementation='eager'
  with torch.device('meta'):self.model=compatible_model_class(mod)(cfg)
  params=dict(self.model.named_parameters());expected={r['source_key']:r for r in p['source_tensor_records']};assert len(params)==len(expected)==236 and set(params)==set(expected)
  index=json.loads((wp/'model.safetensors.index.json').read_text())['weight_map'];assert set(index)==set(params);records=[]
  for shard,header in p['source_files'].items():
   assert sha(wp/shard)==header['sha256'] and (wp/shard).stat().st_size==header['bytes']
   with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
    assert set(f.keys())=={k for k,v in index.items() if v==shard}
    for key in f.keys():
     x=f.get_tensor(key);r=expected[key];assert x.dtype==torch.float32 and list(x.shape)==list(params[key].shape)==r['shape'] and fp(x)==r['source_full_sha256'];z=x.bfloat16();assert fp(z)==r['original_BF16_cast_full_sha256'] and torch.equal(z.float(),x) and r['original_F32_to_BF16_roundtrip_changed_elements']==r['original_nonzero_underflow']==0
     y=z.to(self.device).contiguous();digest=fp(y);assert digest==fp(z) and torch.isfinite(y).all();parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(y,requires_grad=False));records.append({'source_key':key,'target_key':key,'source_file':shard,'shape':list(x.shape),'source_dtype':str(x.dtype),'runtime_dtype':str(y.dtype),'device':str(y.device),'source_sha256':fp(x),'original_BF16_cast_sha256':fp(z),'actual_NPU_sha256':digest,'original_cast_changed_elements':0,'original_nonzero_underflow':0})
   log.info('Own HoVLE loaded %d/236 original parameters',len(records))
  assert self.model.language_model.output.weight is not self.model.language_model.model.tok_embeddings.weight
  rope_records=[];shared={}
  for stream,layers in [('holistic',self.model.embedding_model.encoder),('language',self.model.language_model.model.layers)]:
   assert len(layers)==(8 if stream=='holistic' else 24)
   for i,layer in enumerate(layers):
    rot=layer.attention.rotary_emb
    if stream not in shared:
     kw=dict(max_position_embeddings=rot.max_position_embeddings,base=rot.base,device='cpu')
     if hasattr(rot,'scaling_factor'):kw['scaling_factor']=rot.scaling_factor
     cpu=type(rot)(rot.dim,**kw);shared[stream]=(cpu,{n:b.to(self.device) for n,b in cpu.named_buffers()})
    cpu,buffers=shared[stream]
    for n,b in buffers.items():setattr(rot,n,b)
    rot.max_seq_len_cached=cpu.max_seq_len_cached
    assert rot.dim==128 and rot.base==1000000 and rot.max_seq_len_cached==32768
    rope_records.append({'stream':stream,'layer':i,'class':type(rot).__name__,'dim':rot.dim,'base':rot.base,'factor':getattr(rot,'scaling_factor',None),'max_seq_len_cached':rot.max_seq_len_cached,'buffers':{n:tr(b) for n,b in rot.named_buffers()}})
  assert all(x.device==self.device and x.dtype==torch.bfloat16 for x in self.model.parameters()) and all(not x.is_meta and x.device==self.device for x in self.model.buffers())
  self.model.eval();self.tok=tokenizer(wp,mod.__package__);assert len(self.tok)==92553 and self.tok.bos_token_id==1 and self.tok.convert_tokens_to_ids('<IMG_CONTEXT>')==92546 and self.tok.convert_tokens_to_ids('<|im_end|>')==92542 and self.model.num_image_token==256
  patch_records=apply_original_runtime_patches(self.model,p['original_runtime_patch_source'],p['original_runtime_patch_sha256'],self.device)
  self.model.generation_config=GenerationConfig.from_pretrained(wp,local_files_only=True);gen,unused=self.model._prepare_generation_config(self.model.generation_config,**p['original_generation_kwargs']);assert not unused and gen.use_cache is True and gen.num_beams==1 and gen.do_sample is False and gen.max_new_tokens==2048 and gen.eos_token_id==92542 and gen.repetition_penalty==1 and gen.forced_eos_token_id is None and gen.forced_bos_token_id is None
  self.resolved_generation=gen.to_dict();self.module=mod;self._reset();self._observers();loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));self.probe=self._probe()
  self.lineage={'protocol':'HoVLE_HD_original_BF16_8plus24_native_generation','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'devices':self.devices,'hardware':'Ascend910B','runtime_dtype':'BF16','source_dtype':'F32','weight_tensor_count':236,'all_expected_parameters_loaded':True,'original_untied_output_head_preserved':True,'own_vocab_size':92553,'own_BOS_ID':1,'original_EOS_ID':92542,'low_image_tokens_per_tile':256,'HD_image_tokens_per_tile':1024,'actual_language_sample_batch_size':1,'outer_algorithm_metadata_batch_size':10,'original_max_new_tokens':2048,'resolved_generation_config':self.resolved_generation,'native_cache_slots':32,'original_HD_decode_position_repeats_last_HD_prefill_position':True,'original_HD_mask_not_extended_by_GenerationMixin':True,'weight_loading_evidence':str(loading),'per_sample_evidence':str(self.trace),'activation_execution_trace':str(self.act_trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'generation_progress':str(self.progress),'native_GELU_probe':self.probe,'activation_modules':self.activation_modules,'original_32_analytic_rotary_buffers':rope_records,'original_runtime_patch_method_sha256':patch_records,'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'local_compatibility':['Isolated own HoVLE source, zero-LoRA and zero-drop_path_rate guards; inactive imports omitted without any stand-in; CPU meta schedule only','Exact frozen original HoVLE causal-mask, empty-cache, position-trim and chat device/output-slice patches replayed only in isolated model','Original native combined8+24 tuple cache; no unused initial DynamicCache allocation, exactly as original empty-cache->None guard','Own official GenerationMixin greedy generation and own chat, not FigCodifier manual or other architecture','FP32 analytic full32768 rotary buffers parsed on CPU4threads and transferred unchanged, no learned CPU offload'],'precision_disclosure':'Own236 source F32 tensors contain exactly BF16-representable values; original autoBF16 conversion preserved with full source/cast/actual NPU SHA, zero changed/underflow. All holistic8, projection and InternLM2_24/output learned layers on native NPU BF16 HF32off. Original eager attention/FP32 softmax/native tuple cache and own dual-stream chat/defaults preserved. Actual CANN numerical differences retained; CPU/FP32 controls diagnostic only, no low-bit acceleration claim.'}
 def _probe(self):
  x=torch.linspace(-8,8,20001);arrays={};out={'path':str(self.evidence/'native_GELU_semantics.npz'),'device':str(self.device),'CPU_threads':torch.get_num_threads(),'modes':{}}
  for dtype,label in [(torch.float32,'FP32'),(torch.bfloat16,'BF16')]:
   z=x.to(dtype);arrays[label+'_input_bytes']=z.view(torch.uint8).numpy();out['modes'][label]={}
   for mode in ['none','tanh']:
    c=torch.nn.functional.gelu(z,approximate=mode);n=torch.nn.functional.gelu(z.to(self.device),approximate=mode).cpu();arrays[label+'_CPU_'+mode+'_bytes']=c.view(torch.uint8).numpy();arrays[label+'_NPU_'+mode+'_bytes']=n.view(torch.uint8).numpy();out['modes'][label][mode]={'max_abs_CPU_same_declared':float((c-n).abs().max()),'finite':bool(torch.isfinite(n).all())}
  np.savez_compressed(out['path'],**arrays);out['sha256']=sha(out['path']);return out
 def _reset(self):
  self._arrays={};self._steps=[];self._step=None;self._counts={n:0 for n in getattr(self,'activation_modules',{})};self._pixels=None;self._vis=None;self._HD_hidden=None;self._image_features=None;self._oldcache={};self._actual_masks={};self._actual_pos={};self._rope_pair={};self._bridge=False;self._generated_full=None
 def _store(self,name,x):
  assert x.device==self.device;self._arrays[name]=x.detach().cpu().contiguous().clone();return tr(self._arrays[name])
 def _observers(self):
  self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['SiLUActivation','GELU']};assert len(self.activation_modules)==33
  for name in self.activation_modules:
   def act(m,args,out,name=name):assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   self.model.get_submodule(name).register_forward_hook(act)
  def root_pre(m,args,kw):
   assert kw['use_cache'] is True;self._step={'step':len(self._steps)+1,'layers':[],'cache':[]};self._actual_masks={};self._actual_pos={};self._rope_pair={}
   if not self._steps:
    assert kw['past_key_values'] is None and torch.equal(kw['input_ids'],self._low_ids) and torch.equal(kw['hd_input_ids'],self._hd_ids)
   else:
    assert len(kw['past_key_values'])==32 and kw['input_embeds'].shape==kw['hd_input_embeds'].shape==(1,1,2048) and torch.equal(kw['input_embeds'],kw['hd_input_embeds']);self._step['original_last_token_embedding']=self._store('%04d_decode_embedding'%self._step['step'],kw['input_embeds'])
  def root_post(m,args,out):
   st=self._step;assert len(st['layers'])==len(st['cache'])==len(out.past_key_values)==32
   logits=out.logits[:,-1,:];assert logits.shape==(1,92553) and logits.dtype==torch.float32 and torch.isfinite(logits).all();st['full_next_token_logits']=self._store('%04d_logits'%st['step'],logits);st['next_token_ID']=int(logits.argmax(-1));self._steps.append(st);self._step=None
   if st['step']==1 or st['step']%128==0:
    with self.progress.open('a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'native_root_forward_completed','index':self._index,'forward_completed':st['step'],'tokens_already_appended_before_this_return':st['step']-1,'actual_language_batch_size':1,'original_budget':2048,'device':str(self.device)})+'\n')
  self.model.register_forward_pre_hook(root_pre,with_kwargs=True);self.model.register_forward_hook(root_post)
  def vision_pre(m,args):assert torch.equal(args[0],self._pixels);self._store('actual_pixels',args[0])
  def vision_post(m,args,out):self._vis=out;self._store('actual_vision_patches_plus_position',out)
  self.model.embedding_model.vision_embeddings.register_forward_pre_hook(vision_pre);self.model.embedding_model.vision_embeddings.register_forward_hook(vision_post)
  def HD_layer_input(m,args,kw):
   if len(self._steps)==0:
    hidden=kw['hidden_states'];self._first_HD_input=hidden;assert torch.equal(hidden[self._hd_ids==92546],self._vis.reshape(-1,2048))
  self.model.embedding_model.encoder[0].register_forward_pre_hook(HD_layer_input,with_kwargs=True)
  def hol_post(m,args,out):
   self._image_features,self._HD_hidden,_=out;self._store('actual_HD_final_hidden',self._HD_hidden);self._store('actual_projected_low_visual',self._image_features)
  self.model.embedding_model.register_forward_hook(hol_post)
  def proj_pre(m,args):self._store('actual_pixel_shuffle_output',args[0])
  self.model.embedding_model.pixel_shuffle_proj.register_forward_pre_hook(proj_pre)
  def lm_pre(m,args,kw):
   actual=kw['inputs_embeds'];assert actual.device==self.device
   if len(self._steps)==0:
    low=self._low_ids.reshape(-1)==92546;high=self._hd_ids.reshape(-1)==92546;H=self._HD_hidden.reshape(-1,2048);before=torch.zeros(self._low_ids.shape[1],2048,device=self.device,dtype=torch.bfloat16);before[~low]=H[~high];before[low]=self._image_features.reshape(-1,2048);before=before[None];norm=before.norm(dim=-1,keepdim=True);assert torch.equal(actual,before/norm);self._store('actual_before_original_normalize',before);self._store('actual_original_prefix_norm',norm);self._store('actual_LM_normalized_prefix',actual);self._bridge=True
   else:
    # Original root normalizes the final holistic decode output too.
    last=self._last_HD_layer_output;norm=last.norm(dim=-1,keepdim=True);assert torch.equal(actual,last/norm)
  self.model.language_model.register_forward_pre_hook(lm_pre,with_kwargs=True)
  hol_cls=type(self.model.embedding_model.encoder[0].attention);wrapper=hol_cls.forward;cells=dict(zip(wrapper.__code__.co_freevars,wrapper.__closure__));assert 'orig_forward' in cells;native=cells['orig_forward'].cell_contents
  def observed_native(m,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,output_attentions=False,use_cache=False,**kw):
   self._actual_attention('holistic',self._hol_attention_ids[id(m)],hidden_states,attention_mask,position_ids,past_key_value,use_cache)
   return native(m,hidden_states,attention_mask=attention_mask,position_ids=position_ids,past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,**kw)
  self._hol_attention_ids={id(x.attention):i for i,x in enumerate(self.model.embedding_model.encoder)};cells['orig_forward'].cell_contents=observed_native
  for stream,layers in [('holistic',self.model.embedding_model.encoder),('language',self.model.language_model.model.layers)]:
   for i,layer in enumerate(layers):
    if stream=='language':
     def pre(m,args,kw,i=i):self._actual_attention('language',i,args[0],kw['attention_mask'],kw['position_ids'],kw['past_key_value'],kw['use_cache'])
     layer.register_forward_pre_hook(pre,with_kwargs=True)
    def post(m,args,out,i=i,stream=stream):
     k,v=out[1];old=self._oldcache.pop((stream,i));assert k.device==v.device==self.device and k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape and list(k.shape[:2])==[1,8] and k.shape[-1]==128;equal=old is None or bool(torch.equal(k[:,:,:old[0].shape[2]],old[0]) and torch.equal(v[:,:,:old[1].shape[2]],old[1]));assert equal
     self._step['cache'].append({'stream':stream,'layer':i,'key_shape':list(k.shape),'value_shape':list(v.shape),'device':str(k.device),'dtype':str(k.dtype),'original_tuple_type':isinstance(out[1],tuple),'actual_NPU_full_prior_key_value_prefix_equal':equal})
     if stream=='holistic' and i==7:
      self._last_HD_layer_output=out[0]
      if len(self._steps)==0:self._store('actual_HD_input_hidden',self._first_HD_input)
    def rope(m,args,out,i=i,stream=stream):
     pos=self._actual_pos[stream].reshape(-1);pair=(out[0][pos],out[1][pos])
     if i==0:
      self._rope_pair[stream]=pair;self._step[stream+'_actual_cos']=self._store('%04d_%s_cos'%(self._step['step'],stream),pair[0]);self._step[stream+'_actual_sin']=self._store('%04d_%s_sin'%(self._step['step'],stream),pair[1])
     else:assert all(torch.equal(a,b) for a,b in zip(pair,self._rope_pair[stream]))
    layer.register_forward_hook(post);layer.attention.rotary_emb.register_forward_hook(rope)
  original_generate=self.model.generate
  @functools.wraps(original_generate)
  def generate(*args,**kw):
   out=original_generate(*args,**kw);assert torch.is_tensor(out);self._generated_full=out.detach().cpu().tolist();return out
  self.model.generate=generate
 def _actual_attention(self,stream,i,hidden,mask,pos,past,use_cache):
  st=self._step;assert use_cache is True and hidden.device==mask.device==pos.device==self.device and hidden.dtype==mask.dtype==torch.bfloat16
  if i==0:
   self._actual_masks[stream]=mask;self._actual_pos[stream]=pos;st[stream+'_mask']=tr(mask);st[stream+'_positions']=pos.cpu().tolist()
  else:assert torch.equal(mask,self._actual_masks[stream]) and torch.equal(pos,self._actual_pos[stream])
  q=hidden.shape[1];past_len=0 if past is None else past[0].shape[2];assert tuple(mask.shape)==(1,1,q,past_len+q);st['layers'].append({'stream':stream,'layer':i,'device':str(hidden.device),'dtype':str(hidden.dtype),'past_length':past_len,'actual_same_stream_mask_and_positions_equal':True});self._oldcache[(stream,i)]=past
 def predict(self,prompt,media_path=None):
  self.calls+=1;started=time.monotonic();self._reset();r=self.by_media[str(pathlib.Path(media_path).resolve())];self._index=r['index'];entry={'attempt':self.calls,'index':r['index'],'prompt':r['original_prompt'],'actual_original_model_prompt':prompt,'media_path':str(pathlib.Path(media_path).resolve()),'error':None}
  try:
   assert prompt==r['actual_original_model_prompt'] and sha(media_path)==r['source_image_sha256']
   with Image.open(media_path) as im:x,grid=pixels(im.convert('RGB'))
   assert tr(x)==r['original_pixel_values'] and grid==r['original_grid'];self._pixels=x.to(device=self.device,dtype=torch.bfloat16);self._low_ids=torch.tensor([r['original_low_input_ids']],device=self.device);self._hd_ids=torch.tensor([r['original_hd_input_ids']],device=self.device)
   def call():return self.model.chat(self.tok,self._pixels,prompt,dict(self.p['original_generation_kwargs']),num_patches_list=[len(x)],return_history=True)
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      # Only the first real prefill is profiled; per-step traces cover the rest.
      orig=self.model.forward;done=[False]
      @functools.wraps(orig)
      def profiled(*a,**kw):
       if done[0]:return orig(*a,**kw)
       with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:result=orig(*a,**kw);torch.npu.synchronize(self.device)
       prof.export_chrome_trace(str(self.evidence/'npu_trace.json'));done[0]=True;return result
      self.model.forward=profiled
      try:raw,history=call()
      finally:self.model.forward=orig
     self.profile_done=True
    else:raw,history=call()
   full=self._generated_full;L=self._low_ids.shape[1];assert full and full[0][:L]==r['original_low_input_ids'];tokens=full[0][L:];assert 1<=len(tokens)==len(self._steps)<=2048 and all(s['next_token_ID']==v for s,v in zip(self._steps,tokens)) and all(v!=92542 for v in tokens[:-1]);assert tokens[-1]==92542 or len(tokens)==2048
   decoded=self.tok.batch_decode([tokens],skip_special_tokens=True)[0];assert raw==decoded.split('<|im_end|>')[0].strip() and self._bridge and history==[('<image>\n'+prompt,raw)]
   archive=self.evidence/'samples'/('%03d.npz'%r['index']);np.savez_compressed(archive,**{k:v.view(torch.uint8).numpy() for k,v in self._arrays.items()});detail={'index':r['index'],'original_prompt':r['original_prompt'],'actual_original_model_prompt':prompt,'CPU_pixel_values':tr(x),'original_grid':grid,'actual_tile_batch_size':len(x),'actual_language_sample_batch_size':1,'low_input_ids':self._low_ids.cpu().tolist(),'HD_input_ids':self._hd_ids.cpu().tolist(),'native_activation_call_counts':self._counts,'actual_visual_injection_and_original_normalization_verified':self._bridge,'steps':self._steps,'full_generation_output_ids':full,'generation_token_ids':tokens,'full_decoded_output_before_original_strip':decoded,'raw_output':raw,'truncated':len(tokens)>=2048,'error':None,'archive':str(archive),'archive_sha256':sha(archive),'tensor_records':{k:tr(v) for k,v in self._arrays.items()}};q=self.evidence/'samples'/('%03d.json'%r['index']);q.write_text(json.dumps(detail,ensure_ascii=False,indent=2));entry.update(source_image_sha256=sha(media_path),sample_execution=str(q),sample_execution_sha256=sha(q),generation_token_ids=tokens,generation_count=len(tokens),raw_output=raw,truncated=len(tokens)>=2048,empty=not raw,actual_visual_injection_verified=True)
   with self.progress.open('a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'sample_complete','index':r['index'],'generation_count':len(tokens),'error':None})+'\n')
   log.info('HoVLE NPU sample index=%d tiles=%d generated=%d raw=%r',r['index'],len(x),len(tokens),raw[:140]);return raw
  except Exception:entry['error']=traceback.format_exc();log.error(entry['error']);raise
  finally:
   entry['elapsed_seconds']=time.monotonic()-started;self.details[(r['original_prompt'],media_path)]=entry
   with self.trace.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
   with self.act_trace.open('a') as f:f.write(json.dumps({'index':r['index'],'device':str(self.device),'dtype':'torch.bfloat16','language_sample_batch_size':1,'tile_batch_size':len(self._pixels) if self._pixels is not None else None,'decoder_forwards':len(self._steps),'module_call_counts':self._counts,'error':entry['error']})+'\n')
   self._arrays={};self._pixels=self._vis=self._HD_hidden=self._image_features=self._last_HD_layer_output=self._first_HD_input=None
 def batch_predict(self,batch):
  groups=collections.OrderedDict()
  for i,(prompt,path) in enumerate(batch):
   with Image.open(path) as im:key=(im.width//64,im.height//64)
   groups.setdefault(key,[]).append((i,prompt,path))
  result=[None]*len(batch)
  for group in groups.values():
   for i,prompt,path in group:result[i]=self.predict(prompt,path)
  return result
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'No actual recorded execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

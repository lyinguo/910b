"""Observe the unmodified own Phi-3.5 vLLM model with complete numerical values.

Wrappers call each original bound method once and return its identical result.
This source is not a successful NPU run or numerical proof until actually tested.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,inspect,json,os,types
import numpy as np
import torch

MID='microsoft/Phi-3.5-vision-instruct'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def meta(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t))
def write(path,value):Path(path).write_bytes((json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode())

class OwnPhi35Capture:
 def __init__(self,worker,home):
  self.worker=worker;self.model=worker.model_runner.model;self.home=Path(home)
  self.active=False;self.original_methods=[];self.history={};self.runtime_seen={};self.rotaries_seen=set()
  assert type(self.model).__name__=='Phi3VForCausalLM'
  assert len(self.model.language_model.model.layers)==32 and self.model.config.hidden_size==3072
  self.install()

 def save(self,t,label):
  assert torch.is_tensor(t)
  a=t.detach().cpu().contiguous();m=meta(a)
  data=a.view(torch.uint16).numpy() if a.dtype==torch.bfloat16 else a.numpy()
  key=(m['dtype'],tuple(m['shape']),m['SHA256']);self.occurrences+=1;self.logical_bytes+=data.nbytes
  if key in self.same_values:
   name=self.same_values[key];prior=self.arrays[name]
   assert prior.dtype==data.dtype and prior.shape==data.shape and np.array_equal(prior.view(np.uint8).reshape(-1),data.view(np.uint8).reshape(-1))
   self.bitwise_reuses+=1
  else:
   name='a'+str(self.seq).zfill(8)+'_'+label.replace('.','_');self.seq+=1
   self.arrays[name]=data.copy();self.same_values[key]=name
  return dict(array=name,**m)

 def wrap(self,obj,name,before=None,after=None):
  original=getattr(obj,name);self.original_methods.append((obj,name,original))
  def observed(inst,*args,**kwargs):
   context=before(args,kwargs) if self.active and before else None
   result=original(*args,**kwargs)
   if self.active and after:after(args,kwargs,result,context)
   return result
  setattr(obj,name,types.MethodType(observed,obj))
  return original

 def new_frame(self,args,kwargs):
  data=inspect.signature(self.original_forward).bind_partial(*args,**kwargs).arguments
  self.frame=dict(index=len(self.frames),text_qkv=[],cache=[],rotary=[],head=None,inputs={})
  for name in ['input_ids','positions','inputs_embeds']:
   if torch.is_tensor(data.get(name)):self.frame['inputs'][name]=self.save(data[name],name)
  batch=self.worker.model_runner.input_batch;ids=[x for x in batch.req_ids if x is not None]
  self.frame['active_request_ids']=ids
  self.frame['request_state_before_forward']={rid:dict(prompt_token_ids=list(self.worker.model_runner.requests[rid].prompt_token_ids),output_token_ids_before_forward=list(self.worker.model_runner.requests[rid].output_token_ids)) for rid in ids}
  return self.frame

 def finish_frame(self,args,kwargs,result,context):
  assert context is self.frame
  assert len(self.frame['text_qkv'])==len(self.frame['cache'])==len(self.frame['rotary'])==32,('Full32 coverage',len(self.frame['text_qkv']),len(self.frame['cache']),len(self.frame['rotary']))
  self.frames.append(self.frame)

 def before_cache(self,name,impl,args,kwargs):
  d=inspect.signature(self.backend_originals[name]).bind_partial(*args,**kwargs).arguments
  k=d['key'];v=d['value'];m=d['attn_metadata'];n=int(m.num_actual_tokens)
  slots=m.slot_mapping[:n].detach().cpu().long()
  assert len(slots)==n and slots.ndim==1 and (slots>=0).all()
  assert k.dtype==v.dtype==torch.bfloat16 and k.shape[0]>=n and k.shape[-1]==v.shape[-1]==96
  entry=dict(layer=name,num_actual_tokens=n,slots=self.save(slots,'slots'),incoming_K=self.save(k[:n].reshape(n,32,96),'incoming_K'),incoming_V=self.save(v[:n].reshape(n,32,96),'incoming_V'),metadata={})
  for field in ['query_start_loc','seq_lens','seq_lens_cpu','query_start_loc_cpu','block_table','block_tables','slot_mapping','attn_mask','attention_mask','causal_mask','sliding_window_mask','causal','attn_state']:
   x=getattr(m,field,None)
   if torch.is_tensor(x):entry['metadata'][field]=self.save(x,'metadata_'+field)
   elif isinstance(x,(list,tuple)) and all(isinstance(a,int) for a in x):entry['metadata'][field]=list(x)
   elif x is None or isinstance(x,(bool,int,float,str)):entry['metadata'][field]=x
   else:entry['metadata'][field]=dict(type=type(x).__module__+'.'+type(x).__name__,repr=repr(x))
  return entry,slots,k[:n].reshape(n,32,96).detach().cpu(),v[:n].reshape(n,32,96).detach().cpu(),m

 def after_cache(self,name,impl,args,kwargs,result,context):
  entry,slots,in_k,in_v,m=context;kc,vc=impl.key_cache,impl.value_cache
  assert kc is not None and vc is not None and kc.ndim==vc.ndim==4 and kc.shape==vc.shape and kc.shape[-1]==96
  bnsd=bool(impl.use_bnsd_kv_cache)
  if bnsd:
   assert kc.shape[1]==32;block=kc.shape[2]
   def selected(cache,ss):
    ss=ss.to(cache.device);return cache[ss//block,:,ss%block,:].detach().cpu()
  else:
   assert kc.shape[2]==32;block=kc.shape[1]
   def selected(cache,ss):
    ss=ss.to(cache.device);return cache[ss//block,ss%block,:,:].detach().cpu()
  actual_k=selected(kc,slots);actual_v=selected(vc,slots)
  assert torch.equal(actual_k,in_k) and torch.equal(actual_v,in_v),'Own actual physical KV scatter mismatch'
  previous=self.history.setdefault(name,{})
  for j,s in enumerate(slots.tolist()):previous[s]=(actual_k[j].clone(),actual_v[j].clone())
  table=getattr(m,'block_tables',None)
  if table is None:table=getattr(m,'block_table',None)
  assert torch.is_tensor(table)
  lengths=getattr(m,'seq_lens',None)
  if lengths is None:lengths=getattr(m,'seq_lens_cpu',None)
  starts=getattr(m,'query_start_loc',None)
  if starts is None:starts=getattr(m,'query_start_loc_cpu',None)
  lengths=lengths.detach().cpu().tolist() if torch.is_tensor(lengths) else list(lengths)
  starts=starts.detach().cpu().tolist() if torch.is_tensor(starts) else list(starts)
  ids=self.frame['active_request_ids'];assert len(lengths)>=len(ids) and len(starts)>=len(ids)+1
  tables=table.detach().cpu().long();active=[]
  for j,rid in enumerate(ids):
   length=int(lengths[j]);q=int(starts[j+1]-starts[j]);assert 0<q<=length
   pos=torch.arange(length,dtype=torch.int64);physical=tables[j,pos//block]*block+pos%block
   order=physical.tolist();assert all(s in previous for s in order),('Uncaptured active native prefix',name,rid)
   ak=selected(kc,physical);av=selected(vc,physical)
   assert torch.equal(ak,torch.stack([previous[s][0] for s in order])) and torch.equal(av,torch.stack([previous[s][1] for s in order])),'Active physical native cache prefix changed'
   active.append(dict(request_id=rid,seq_len=length,query_len=q,query_range=[int(starts[j]),int(starts[j+1])],logical_position_to_physical_slots=order,committed_K=meta(ak),committed_V=meta(av),all_active_prefix_values_bitwise_reconstructed=True))
  entry.update(actual_cache_layout='BNSD' if bnsd else 'BSND',cache_shape=list(kc.shape),block_size=block,actual_current_slot_K=self.save(actual_k,'actual_slot_K'),actual_current_slot_V=self.save(actual_v,'actual_slot_V'),active_committed_cache=active,prior_groups_are_preserved_for_actual_native_prefix_cache=True)
  self.frame['cache'].append(entry)

 def install(self):
  self.original_forward=self.wrap(self.model,'forward',self.new_frame,self.finish_frame)
  def head_before(args,kwargs):return self.save(args[0] if args else kwargs['hidden_states'],'head_hidden')
  def head_after(args,kwargs,result,context):
   assert self.frames and result is not None and result.ndim==2 and result.shape[-1]==32064
   assert result.dtype in [torch.float32,torch.bfloat16] and torch.isfinite(result).all()
   assert self.frames[-1]['head'] is None
   lp=self.model.language_model.logits_processor
   self.frames[-1]['head']=dict(hidden=context,logits=self.save(result,'all32064_logits'),logits_processor=dict(org_vocab_size=lp.org_vocab_size,scale=lp.scale,soft_cap=lp.soft_cap))
  self.wrap(self.model,'compute_logits',head_before,head_after)
  def sampler_before(args,kwargs):
   assert self.frames and self.frames[-1]['head'] is not None
   logits=args[0] if args else kwargs['logits'];sampling=args[1] if len(args)>1 else kwargs['sampling_metadata']
   def serial(x):
    if torch.is_tensor(x):return self.save(x,'sampling_metadata')
    if x is None or isinstance(x,(str,int,float,bool)):return x
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return dict(type=type(x).__module__+'.'+type(x).__name__,repr=repr(x))
   fields={k:serial(v) for k,v in vars(sampling).items()};assert fields.get('all_greedy') is True
   self.frames[-1]['sampling_metadata']=fields;return logits
  def sampler_after(args,kwargs,result,context):
   assert torch.is_tensor(result.sampled_token_ids)
   self.frames[-1]['sampler']=dict(input_logits=self.save(context,'native_sampler_input_logits'),sampled_token_ids=self.save(result.sampled_token_ids,'actual_native_sampled_ids'))
  self.wrap(self.worker.model_runner.sampler,'forward',sampler_before,sampler_after)
  def embed_before(args,kwargs):
   d=inspect.signature(self.original_embed_input_ids).bind_partial(*args,**kwargs).arguments
   entry=dict(input_ids=self.save(d['input_ids'],'native_embedding_input_ids'),is_multimodal=None,multimodal_embeddings=[])
   if d.get('is_multimodal') is not None:entry['is_multimodal']=self.save(d['is_multimodal'],'native_image_position_mask')
   mm=d.get('multimodal_embeddings')
   if mm is not None:entry['multimodal_embeddings']=[self.save(t,'native_injected_image_features') for t in mm]
   return entry
  def embed_after(args,kwargs,result,context):
   context['output']=self.save(result,'native_actual_merged_embeddings');self.embeddings.append(context)
  self.original_embed_input_ids=self.wrap(self.model,'embed_input_ids',embed_before,embed_after)
  def visual_before(args,kwargs):
   pixels=kwargs.get('pixel_values');sizes=kwargs.get('image_sizes')
   if pixels is None:return None
   assert torch.is_tensor(pixels) and torch.is_tensor(sizes)
   return dict(pixels=self.save(pixels,'own_native_pixels'),image_sizes=self.save(sizes,'own_HD_sizes'))
  def visual_after(args,kwargs,result,context):
   if context is None:return
   assert isinstance(result,(tuple,list)) and all(torch.is_tensor(v) and v.device.type=='npu' for v in result)
   context['projected_features']=[self.save(v,'own_native_HD_projected') for v in result];self.vision.append(context)
  self.wrap(self.model,'embed_multimodal',visual_before,visual_after)
  def projection_before(args,kwargs):return self.save(args[0] if args else kwargs['input'],'HD_projection_input')
  def projection_after(args,kwargs,result,context):
   assert torch.is_tensor(result) and result.device.type=='npu'
   self.projectors.append(dict(input=context,output=self.save(result,'HD_projection_output')))
  self.wrap(self.model.vision_embed_tokens.img_projection,'forward',projection_before,projection_after)
  def clip_before(args,kwargs):return self.save(args[0] if args else kwargs['img'],'CLIP_flattened_crops')
  def clip_after(args,kwargs,result,context):
   assert torch.is_tensor(result) and result.shape[-2:]==(576,1024)
   self.clip.append(dict(input=context,output=self.save(result,'own_CLIP_minus2_features')))
  self.wrap(self.model.vision_embed_tokens,'get_img_features',clip_before,clip_after)
  self.backend_originals={}
  for i,layer in enumerate(self.model.language_model.model.layers):
   name='language_model.model.layers.'+str(i);attention=layer.self_attn
   assert attention.q_size==attention.kv_size==3072 and attention.qkv_proj.weight.dtype==torch.bfloat16
   def attn_before(args,kwargs,name=name):self.current_layer=name
   self.wrap(attention,'forward',attn_before,None)
   def qkv_before(args,kwargs):return self.save(args[0] if args else kwargs['input_'],'own_qkv_input')
   def qkv_after(args,kwargs,result,context,name=name):
    out,bias=result;assert bias is None and out.shape[-1]==9216 and out.dtype==torch.bfloat16
    self.frame['text_qkv'].append(dict(layer=name,input=context,packed_output=self.save(out,'own_QKV9216')))
   self.wrap(attention.qkv_proj,'forward',qkv_before,qkv_after)
   rope=attention.rotary_emb
   assert rope.head_size==rope.rotary_dim==96 and hasattr(rope,'long_short_cos_sin_cache')
   if id(rope) not in self.rotaries_seen:
    self.rotaries_seen.add(id(rope));original=rope.forward
    def rope_before(args,kwargs,rope=rope,original=original):
     d=inspect.signature(original).bind_partial(*args,**kwargs).arguments
     entry=dict(layer=self.current_layer,positions=self.save(d['positions'],'positions'),query_before=self.save(d['query'],'rotary_Q_before'),key_before=self.save(d['key'],'rotary_K_before'),offsets=None if d.get('offsets') is None else self.save(d['offsets'],'rope_offsets'),table=self.save(rope.long_short_cos_sin_cache,'own_native_intrinsic_LongRoPE_table'),head_size=rope.head_size,rotary_dim=rope.rotary_dim,use_long_rope=bool(rope.use_long_rope),original_max_position_embeddings=rope.original_max_position_embeddings,max_position_embeddings=rope.max_position_embeddings,short_mscale=rope.short_mscale,long_mscale=rope.long_mscale)
     return entry
    def rope_after(args,kwargs,result,context):
     q,k=result;context.update(query_after=self.save(q,'rotary_Q_after'),key_after=self.save(k,'rotary_K_after'));self.frame['rotary'].append(context)
    self.wrap(rope,'forward',rope_before,rope_after)
   impl=attention.attn.impl
   assert type(impl).__name__=='AscendAttentionBackendImpl' and impl.kv_sharing_target_layer_name is None and not impl.pcp_enabled
   self.backend_originals[name]=impl.reshape_and_cache
   def cache_before(args,kwargs,name=name,impl=impl):return self.before_cache(name,impl,args,kwargs)
   def cache_after(args,kwargs,result,context,name=name,impl=impl):return self.after_cache(name,impl,args,kwargs,result,context)
   self.wrap(impl,'reshape_and_cache',cache_before,cache_after)

 def begin(self,group):
  assert not self.active;self.group=group;self.frames=[];self.vision=[];self.projectors=[];self.clip=[];self.embeddings=[];self.arrays={};self.same_values={};self.occurrences=0;self.logical_bytes=0;self.bitwise_reuses=0;self.seq=0;self.active=True

 def finish(self):
  assert self.active;self.active=False
  assert self.frames and all(x['head'] is not None and x.get('sampler') is not None for x in self.frames)
  dest=self.home/('full_native_group_'+str(self.group).zfill(2)+'.npz');assert not dest.exists()
  np.savez_compressed(dest,**self.arrays)
  runtime=set()
  for module in list(sys.modules.values()):
   name=getattr(module,'__file__',None)
   if name and Path(name).is_file() and Path(name).suffix in ['.py','.so']:runtime.add(Path(name).resolve())
  for line in Path('/proc/self/maps').read_text().splitlines():
   name=line.split()[-1]
   if name.startswith('/') and ('.so' in name or 'python3' in name) and Path(name).is_file():runtime.add(Path(name).resolve())
  libraries={}
  for path in sorted(runtime):
   st=path.stat();signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns);prior=self.runtime_seen.get(str(path))
   if prior is None or prior[0]!=signature:self.runtime_seen[str(path)]=(signature,sha(path))
   libraries[str(path)]=self.runtime_seen[str(path)][1]
  storage=dict(all_actual_tensor_occurrences=self.occurrences,physical_unique_arrays=len(self.arrays),logical_unshared_raw_nbytes=self.logical_bytes,physical_unique_raw_nbytes=sum(x.nbytes for x in self.arrays.values()),bitwise_duplicate_occurrences=self.bitwise_reuses,every_duplicate_compared_bitwise_to_actual_current_NPU_value=True,no_tensor_or_occurrence_or_numeric_check_dropped=True)
  record=dict(passed_capture=True,model_id=MID,group_index=self.group,actual_worker_PID=os.getpid(),actual_native_forward_calls=len(self.frames),full_all32_layer_KV_cache_scatter_and_active_prefix_values_checked=True,full32064_head_all_frames=True,model_math_implementation_unmodified=True,frames=self.frames,vision=self.vision,vision_projectors=self.projectors,CLIP_features=self.clip,input_embeddings=self.embeddings,archive=str(dest),archive_SHA256=sha(dest),archive_bytes=dest.stat().st_size,lossless_shared_tensor_storage=storage,all_actual_worker_imported_sources_and_mapped_shared_libraries_SHA256=libraries,actual_worker_source=__file__,actual_worker_source_SHA256=sha(__file__))
  out=dest.with_suffix('.json');write(out,record);self.arrays={};self.same_values={}
  return dict(path=str(out),SHA256=sha(out),archive=str(dest),archive_SHA256=sha(dest),actual_native_forward_calls=len(self.frames))

def inspect_own_sources(worker,root,headers,home):
 from safetensors import safe_open
 model=worker.model_runner.model;parameters=dict(model.named_parameters(remove_duplicate=False));records={};unused=[]
 assert len(headers)==592
 for shard in sorted({v['shard'] for v in headers.values()}):
  with safe_open(str(Path(root)/shard),framework='pt',device='cpu') as reader:
   assert set(reader.keys())=={k for k,v in headers.items() if v['shard']==shard}
   for source in reader.keys():
    raw=reader.get_tensor(source).contiguous();assert raw.dtype==torch.bfloat16 and list(raw.shape)==headers[source]['shape']
    name=source
    for old,new in [('model.vision_embed_tokens.wte','embed_tokens'),('model.vision_embed_tokens.','vision_embed_tokens.'),('lm_head.','language_model.lm_head.'),('model.','language_model.model.')]:
     if name.startswith(old):name=new+name[len(old):];break
    offset=None
    if source.startswith('model.vision_embed_tokens.'):
     for part,label in [('.q_proj.','q'),('.k_proj.','k'),('.v_proj.','v')]:
      if part in source:
       q=headers[source.replace(part,'.q_proj.')]['shape'][0];k=headers[source.replace(part,'.k_proj.')]['shape'][0]
       offset={'q':0,'k':q,'v':q+k}[label];name=name.replace(part,'.qkv_proj.');break
    candidates=[name,name.replace('.img_processor.vision_model.','.img_processor.')]
    matches=list(dict.fromkeys(n for n in candidates if n in parameters))
    if not matches:
     assert source.startswith('model.vision_embed_tokens.img_processor.vision_model.') and ('.encoder.layers.23.' in source or '.post_layernorm.' in source),('Own active source missing',source,candidates)
     records[source]=dict(source_file=shard,source=meta(raw),original_native_unused=True,reason='Original native CLIP selects hidden_states[-2]:23 layers, no pooled post_layernorm');unused.append(source);continue
    assert len(matches)==1;target=parameters[matches[0]]
    assert target.device.type=='npu' and target.dtype==raw.dtype
    if offset is not None:target=target[offset:offset+raw.shape[0]]
    if target.shape!=raw.shape:
     assert matches[0].endswith(('embed_tokens.weight','lm_head.weight')) and target.shape[1:]==raw.shape[1:] and target.shape[0]>=raw.shape[0];target=target[:raw.shape[0]]
    assert target.shape==raw.shape and tensor_sha(target)==tensor_sha(raw),source
    records[source]=dict(source_file=shard,source=meta(raw),native_parameter=matches[0],packed_row_offset=offset,actual_NPU=meta(target),actual_device=str(target.device),original_native_unused=False)
 assert set(records)==set(headers) and len(unused)==18 and len(records)-len(unused)==574
 assert model.embed_tokens.weight is model.language_model.model.embed_tokens.weight
 result=dict(passed=True,model_id=MID,own_source_count=592,actual_native_source_count=574,original_native_unused_count=18,all_own_sources=records,original_native_unused_sources=unused,original_text_image_embedding_alias_identity=True,no_foreign_checkpoint_used=True)
 out=Path(home)/'own592_source574_actual_NPU18_original_unused_mapping.json';write(out,result)
 return dict(path=str(out),SHA256=sha(out),own_source_count=592,actual_native_source_count=574,original_native_unused_count=18)

def install_capture(worker,root,headers,home):
 assert not hasattr(worker,'_own_phi35_full_capture')
 inspection=inspect_own_sources(worker,root,headers,home);worker._own_phi35_full_capture=OwnPhi35Capture(worker,home)
 return dict(installed=True,worker_PID=os.getpid(),own_source_inspection=inspection,no_evaluated_request_started=True)

class OwnPhi35FullValueWorkerExtension:
 def own_phi35_install_capture(self,root,headers,home):
  import shutil
  assert shutil.disk_usage(home).free>int(4.5*1024**3)
  return install_capture(self,root,headers,home)
 def own_phi35_begin_capture(self,group,count):
  import shutil
  assert 1<=count<=10 and shutil.disk_usage(self._own_phi35_full_capture.home).free>int(4.5*1024**3)+count*2*1024**3
  self._own_phi35_full_capture.begin(group);return dict(started=True,group_index=group)
 def own_phi35_finish_capture(self):
  import shutil
  observer=self._own_phi35_full_capture;exact=sum(x.nbytes for x in observer.arrays.values())
  assert shutil.disk_usage(observer.home).free>int(4.5*1024**3)+exact+1024**2
  return observer.finish()

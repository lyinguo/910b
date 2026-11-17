"""Read-only observers for the own original Pali10pt vLLM worker.

No model/attention/sampler implementation is replaced. Wrappers invoke the
original bound method once and return the identical result. All saved arrays
are actual worker values, not CPU model predictions. Untested on NPU: this
source alone is neither ready for dispatch nor a numerical acceptance.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,inspect,types,os,traceback
import numpy as np
import torch

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def jwrite(p,v):Path(p).write_bytes(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2).encode())
def meta(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t))

class OwnWorkerCapture:
 def __init__(self,worker,root,source_header,home):
  self.worker=worker;self.model=worker.model_runner.model;self.root=Path(root);self.home=Path(home);self.header=source_header;self.active=False;self.handles=[];self.original_methods=[];self.history={};self.frames=[];self.arrays={};self.seq=0;self.runtime_seen={}
  assert type(self.model).__name__=='PaliGemmaForConditionalGeneration'
  assert len(self.model.language_model.model.layers)==42
  self.install()
 def save(self,t,label):
  assert torch.is_tensor(t);a=t.detach().cpu().contiguous();name='a'+str(self.seq).zfill(8)+'_'+label.replace('.','_');self.seq+=1
  data=a.view(torch.uint16).numpy().copy() if a.dtype==torch.bfloat16 else a.numpy().copy();self.arrays[name]=data;return dict(array=name,**meta(a))
 def wrap(self,obj,name,before=None,after=None):
  original=getattr(obj,name);self.original_methods.append((obj,name,original))
  def observed(inst,*args,**kwargs):
   context=before(args,kwargs) if self.active and before else None
   result=original(*args,**kwargs)
   if self.active and after:after(args,kwargs,result,context)
   return result
  setattr(obj,name,types.MethodType(observed,obj))
 def new_frame(self,args,kwargs):
  data=inspect.signature(self.original_forward).bind_partial(*args,**kwargs).arguments
  self.frame=dict(index=len(self.frames),text_qkv=[],cache=[],head=None,inputs={})
  for name in ['input_ids','positions','inputs_embeds']:
   if torch.is_tensor(data.get(name)):self.frame['inputs'][name]=self.save(data[name],name)
  batch=self.worker.model_runner.input_batch;ids=[x for x in batch.req_ids if x is not None];self.frame['active_request_ids']=ids
  reqs={}
  for rid in ids:
   request=self.worker.model_runner.requests[rid];reqs[rid]=dict(prompt_token_ids=list(request.prompt_token_ids),output_token_ids_before_forward=list(request.output_token_ids))
  self.frame['request_state_before_forward']=reqs;return self.frame
 def finish_frame(self,args,kwargs,result,context):
  assert context is self.frame and len(self.frame['text_qkv'])==42 and len(self.frame['cache'])==42,('Incomplete observer coverage',len(self.frame['text_qkv']),len(self.frame['cache']))
  self.frames.append(self.frame)
 def backend_cache(self,name,impl,args,kwargs):
  # Bind the real installed Ascend reshape_and_cache signature. Its source
  # passes the exact tensors below to DeviceOperator.reshape_and_cache.
  data=inspect.signature(self.backend_originals[name]).bind_partial(*args,**kwargs).arguments
  k=data['key'];v=data['value'];m=data['attn_metadata'];n=int(m.num_actual_tokens);slots=m.slot_mapping[:n].detach().cpu().to(torch.int64);assert slots.ndim==1 and len(slots)==n and (slots>=0).all()
  assert k.dtype==v.dtype==torch.bfloat16 and k.shape[0]>=n and k.shape[-1]==v.shape[-1]==256
  c=dict(layer=name,slots=self.save(slots,'slots'),incoming_K=self.save(k[:n].reshape(n,8,256),'incoming_K'),incoming_V=self.save(v[:n].reshape(n,8,256),'incoming_V'),num_actual_tokens=n,metadata={})
  for field in ['query_start_loc','seq_lens','seq_lens_cpu','query_start_loc_cpu','block_table','block_tables','slot_mapping']:
   t=getattr(m,field,None)
   if torch.is_tensor(t):c['metadata'][field]=self.save(t,'metadata_'+field)
   elif isinstance(t,(list,tuple)) and all(isinstance(x,int) for x in t):c['metadata'][field]=list(t)
  return c,slots,k[:n].reshape(n,8,256).detach().cpu(),v[:n].reshape(n,8,256).detach().cpu()
 def after_cache(self,name,impl,args,kwargs,result,context):
  c,slots,k,v=context;kc,vc=impl.key_cache,impl.value_cache;assert kc is not None and vc is not None
  assert kc.ndim==vc.ndim==4 and kc.shape==vc.shape and kc.shape[-1]==256
  bnsd=bool(impl.use_bnsd_kv_cache)
  if bnsd:
   assert kc.shape[1]==8;block=kc.shape[2]
   def selected(cache,ss):return cache[ss.to(cache.device)//block,:,ss.to(cache.device)%block,:].detach().cpu()
  else:
   assert kc.shape[2]==8;block=kc.shape[1]
   def selected(cache,ss):return cache[ss.to(cache.device)//block,ss.to(cache.device)%block,:,:].detach().cpu()
  observed_k=selected(kc,slots);observed_v=selected(vc,slots)
  assert torch.equal(observed_k,k) and torch.equal(observed_v,v),'Actual native paged cache scatter must be bitwise equal to incoming K/V'
  old=self.history.setdefault(name,{})
  newslots=set(slots.tolist());previous=sorted(set(old)-newslots)
  if previous:
   ss=torch.tensor(previous,dtype=torch.int64);pk=selected(kc,ss);pv=selected(vc,ss)
   assert torch.equal(pk,torch.stack([old[s][0] for s in previous])) and torch.equal(pv,torch.stack([old[s][1] for s in previous])),'Previously committed native KV slots changed'
  for j,slot in enumerate(slots.tolist()):old[slot]=(observed_k[j].clone(),observed_v[j].clone())
  allslots=sorted(old);whole_k=torch.stack([old[s][0] for s in allslots]);whole_v=torch.stack([old[s][1] for s in allslots])
  c.update(actual_cache_layout='BNSD' if bnsd else 'BSND',cache_shape=list(kc.shape),actual_current_slot_K=self.save(observed_k,'actual_slot_K'),actual_current_slot_V=self.save(observed_v,'actual_slot_V'),all_previous_slots_bitwise_preserved=True,committed_slot_count=len(old),reconstructed_committed_slot_order=allslots,committed_K=meta(whole_k),committed_V=meta(whole_v));self.frame['cache'].append(c)
 def install(self):
  self.original_forward=self.model.forward
  self.wrap(self.model,'forward',self.new_frame,self.finish_frame)
  def logits_before(args,kwargs):return self.save(args[0] if args else kwargs['hidden_states'],'head_hidden')
  def logits_after(args,kwargs,result,context):
   assert self.frames and result is not None and result.ndim==2 and result.shape[-1]==257216 and result.dtype==torch.float32
   assert self.frames[-1]['head'] is None;self.frames[-1]['head']=dict(hidden=context,logits=self.save(result,'full257216_native_logits'))
  self.wrap(self.model,'compute_logits',logits_before,logits_after)
  def sample_before(args,kwargs):
   assert self.frames and self.frames[-1]['head'] is not None
   logits=args[0] if args else kwargs['logits'];sampling=args[1] if len(args)>1 else kwargs['sampling_metadata']
   def serial(v):
    if torch.is_tensor(v):return self.save(v,'sampling_metadata')
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return dict(type=type(v).__module__+'.'+type(v).__name__,repr=repr(v))
   fields={k:serial(v) for k,v in vars(sampling).items()};assert fields.get('all_greedy') is True
   self.frames[-1]['sampling_metadata']=fields;return logits
  def sample_after(args,kwargs,result,context):
   ids=result.sampled_token_ids;assert torch.is_tensor(ids) and ids.dtype in [torch.int32,torch.int64]
   self.frames[-1]['sampler']=dict(processed_logits=self.save(context,'native_sampler_processed_logits'),sampled_token_ids=self.save(ids,'native_sampler_selected_ids'))
  self.wrap(self.worker.model_runner.sampler,'forward',sample_before,sample_after)
  def visual_before(args,kwargs):
   pixels=kwargs.get('pixel_values');return None if pixels is None else self.save(pixels,'own_actual_vision_pixels')
  def visual_after(args,kwargs,result,context):
   if context is None:return
   assert torch.is_tensor(result);self.vision.append(dict(pixels=context,projected_features=self.save(result,'own_actual_vision_projection')))
  self.wrap(self.model,'embed_multimodal',visual_before,visual_after)
  self.backend_originals={}
  for i,layer in enumerate(self.model.language_model.model.layers):
   name='language_model.model.layers.'+str(i);attention=layer.self_attn
   assert attention.qkv_proj.weight.dtype==torch.bfloat16 and attention.q_size==4096 and attention.kv_size==2048
   def before(args,kwargs,name=name):return self.save(args[0] if args else kwargs['input_'],'own_qkv_input_'+name)
   def after(args,kwargs,result,context,name=name):
    output,bias=result;assert bias is None and output.dtype==torch.bfloat16 and output.shape[-1]==8192
    self.frame['text_qkv'].append(dict(layer=name,input=context,packed_output=self.save(output,'own_qkv_packed_'+name)))
   self.wrap(attention.qkv_proj,'forward',before,after)
   # get_rope() may return a shared cached module. Observe the owning
   # attention's positions once, rather than nesting wrappers on that shared
   # rotary object once per layer. Post-RoPE K is captured at the cache call.
   original_attention=attention.forward
   def apos(args,kwargs,original_attention=original_attention,name=name,attention=attention):
    values=inspect.signature(original_attention).bind_partial(*args,**kwargs).arguments
    rotary=attention.rotary_emb
    self.frame.setdefault('rotary',[]).append(dict(layer=name,positions=self.save(values['positions'],'rope_positions'),native_rotary_class=type(rotary).__module__+'.'+type(rotary).__name__,native_rotary_dim=int(rotary.rotary_dim),native_head_size=int(rotary.head_size),native_is_neox_style=bool(rotary.is_neox_style),native_cos_sin_cache=meta(rotary.cos_sin_cache)))
   self.wrap(attention,'forward',apos,None)
   impl=attention.attn.impl;assert type(impl).__name__=='AscendAttentionBackendImpl' and impl.kv_sharing_target_layer_name is None and not impl.pcp_enabled
   self.backend_originals[name]=impl.reshape_and_cache
   def pre(args,kwargs,name=name,impl=impl):return self.backend_cache(name,impl,args,kwargs)
   def post(args,kwargs,result,context,name=name,impl=impl):return self.after_cache(name,impl,args,kwargs,result,context)
   self.wrap(impl,'reshape_and_cache',pre,post)
 def begin(self,group):
  assert not self.active;self.group=group;self.history={};self.frames=[];self.arrays={};self.vision=[];self.seq=0;self.active=True
 def finish(self):
  assert self.active;self.active=False;assert self.frames and all(f['head'] is not None and f.get('sampler') is not None for f in self.frames)
  dest=self.home/('full_native_group_'+str(self.group).zfill(2)+'.npz');assert not dest.exists();np.savez_compressed(dest,**self.arrays)
  runtime=set()
  for module in list(sys.modules.values()):
   name=getattr(module,'__file__',None)
   if name:
    p=Path(name)
    if p.is_file() and p.suffix in ['.py','.so']:runtime.add(p.resolve())
  for line in Path('/proc/self/maps').read_text().splitlines():
   f=line.split()[-1]
   if f.startswith('/') and ('.so' in f or 'python3' in f) and Path(f).is_file():runtime.add(Path(f).resolve())
  libraries={}
  for p in sorted(runtime):
   st=p.stat();sig=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns);previous=self.runtime_seen.get(str(p))
   if previous is None or previous[0]!=sig:self.runtime_seen[str(p)]=(sig,sha(p))
   libraries[str(p)]=self.runtime_seen[str(p)][1]
  manifest=dict(passed_capture=True,group_index=self.group,actual_worker_PID=os.getpid(),actual_native_forward_calls=len(self.frames),full_all42_layer_KV_cache_scatter_and_previous_slot_values_checked=True,full_257216_logits_all_frames=True,model_math_implementation_unmodified=True,original_engine_backend_dtype_and_sampling_unmodified=True,frames=self.frames,vision=self.vision,archive=str(dest),archive_SHA256=sha(dest),archive_bytes=dest.stat().st_size,all_actual_worker_imported_sources_and_mapped_shared_libraries_SHA256=libraries,actual_worker_source=__file__,actual_worker_source_SHA256=sha(__file__))
  f=dest.with_suffix('.json');jwrite(f,manifest);self.arrays={};self.history={};return dict(path=str(f),SHA256=sha(f),archive=str(dest),archive_SHA256=sha(dest),actual_native_forward_calls=len(self.frames))

def inspect_own_sources(worker,root,headers,home):
 from safetensors import safe_open
 model=worker.model_runner.model;parameters=dict(model.named_parameters(remove_duplicate=False));records={}
 for shard in sorted({r['shard'] for r in headers.values()}):
  with safe_open(str(Path(root)/shard),framework='pt',device='cpu') as f:
   assert set(f.keys())=={n for n,r in headers.items() if r['shard']==shard}
   for source in f.keys():
    raw=f.get_tensor(source).contiguous();record=headers[source];assert raw.dtype==torch.bfloat16 and list(raw.shape)==record['shape']
    name=source
    for old,new in [('model.language_model.','language_model.model.'),('model.vision_tower.','vision_tower.'),('model.multi_modal_projector.','multi_modal_projector.'),('lm_head.','language_model.lm_head.')]:
     if name.startswith(old):name=new+name[len(old):];break
    offset=None
    for part,label in [('.q_proj.','q'),('.k_proj.','k'),('.v_proj.','v')]:
     if part in source:
      q=headers[source.replace(part,'.q_proj.')]['shape'][0];k=headers[source.replace(part,'.k_proj.')]['shape'][0];offset={'q':0,'k':q,'v':q+k}[label];name=name.replace(part,'.qkv_proj.');break
    for part,index in [('.gate_proj.',0),('.up_proj.',1)]:
     if part in source:name=name.replace(part,'.gate_up_proj.');offset=index*raw.shape[0]
    assert name in parameters,('Own source missing from real engine',source,name)
    target=parameters[name];assert target.device.type=='npu' and target.dtype==raw.dtype
    if offset is not None:target=target[offset:offset+raw.shape[0]]
    elif target.shape!=raw.shape:
     assert name.endswith('embed_tokens.weight') or name.endswith('lm_head.weight');assert target.shape[1:]==raw.shape[1:] and target.shape[0]>=raw.shape[0];target=target[:raw.shape[0]]
    assert target.shape==raw.shape and tensor_sha(target)==tensor_sha(raw),source
    records[source]=dict(source_file=shard,source=meta(raw),native_parameter=name,packed_row_offset=offset,actual_NPU=meta(target),actual_device=str(target.device))
 assert set(records)==set(headers)
 dest=Path(home)/'own_all_source_cast_actual_NPU_packed_mapping.json';jwrite(dest,dict(passed=True,model_id='google/paligemma2-10b-pt-448',own_source_count=len(records),all_own_sources=records,source_count_not_claimed_as_executed_module_count=True,no_foreign_checkpoint_used=True))
 return dict(path=str(dest),SHA256=sha(dest),own_source_count=len(records))
def install_capture(worker,root,headers,home):
 assert not hasattr(worker,'_own_pali10pt_full_capture');source=inspect_own_sources(worker,root,headers,home);worker._own_pali10pt_full_capture=OwnWorkerCapture(worker,root,headers,home)
 return dict(installed=True,worker_PID=os.getpid(),native_model=type(worker.model_runner.model).__name__,own_source_inspection=source,no_evaluated_request_started=True)
def begin_capture(worker,group):worker._own_pali10pt_full_capture.begin(group);return dict(started=True,group=group)
def finish_capture(worker):return worker._own_pali10pt_full_capture.finish()

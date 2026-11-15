"""Own MiniCPM-o-2_6 native BF16 greedy inference, complete NPU source/head/KV evidence."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,hashlib,inspect,logging,subprocess,traceback,shutil,zipfile,re
import torch,torch_npu,numpy as np
from safetensors import safe_open
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.minicpmo26_own_frontend_bf16_v2_unpadded_vllm import OwnMiniCPMO26InputsV2,tensor_sha,flatten_tensors
from universal_eval.model.minicpmo26_own_original_20261116.configuration_minicpm import MiniCPMOConfig
from universal_eval.model.minicpmo26_own_original_20261116.modeling_image_branch import MiniCPMOImageBranch

MID='openbmb/MiniCPM-o-2_6';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array(t):
 t=t.detach().cpu().contiguous();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def save_npz(p,values):
 with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for k,v in values.items():
   with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(v),allow_pickle=False)

class MiniCPMO26OwnNativeBF16V2(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
  assert protocol=={'model_id':MID} and npu_devices==[4]
  self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**kw):raise RuntimeError('Only own existing weights are authorized')
 def load(self,model_path,**kwargs):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=json.loads(self.protocol_file.read_bytes());self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,r in self.p['own_weight_files'].items():assert sha(self.W/n)==r['SHA256'],n
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert info.returncode==0 and 'No running processes found in NPU 4' in info.stdout
  self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  self.inputs=OwnMiniCPMO26InputsV2(self.W);self.tokenizer=self.inputs.tokenizer;self.stops=self.p['original_stop_ids'];assert self.stops and 151645 in self.stops
  cfg=MiniCPMOConfig(**json.loads((self.W/'config.json').read_bytes()));cfg._attn_implementation=cfg.vision_config._attn_implementation='eager'
  assert cfg.num_hidden_layers==28 and cfg.num_key_value_heads==4 and cfg.hidden_size==3584 and cfg.hidden_size//cfg.num_attention_heads==128 and cfg.eos_token_id==151645 and cfg.query_num==64 and not cfg.tie_word_embeddings
  with torch.device('meta'):self.model=MiniCPMOImageBranch(cfg)
  expected=json.loads(Path(self.p['own_source_manifest']).read_bytes());index=json.loads((self.W/'model.safetensors.index.json').read_bytes())['weight_map'];params=dict(self.model.named_parameters(remove_duplicate=False));assert set(params)==set(expected) and set(params)|set(self.p['unused_own_source_tensor_names'])==set(index) and not set(params)&set(self.p['unused_own_source_tensor_names'])
  weights={}
  for filename in sorted(set(index.values())):
   with safe_open(str(self.W/filename),framework='pt',device='cpu') as f:
    for key in f.keys():
     if key not in expected:continue
     t=f.get_tensor(key);assert t.dtype==torch.bfloat16 and tensor_sha(t)==expected[key]['source_SHA256'] and list(t.shape)==expected[key]['shape'];weights[key]=t.to(self.device)
  self.model.load_state_dict(weights,strict=True,assign=True);del weights
  rotary=Qwen2RotaryEmbedding(cfg,device='cpu');assert rotary.rope_type=='default';buffers={}
  for name,t in list(self.model.named_buffers(remove_duplicate=False)):
   if name.endswith(('.inv_freq','.original_inv_freq')):v=rotary.inv_freq.detach().cpu()
   else:
    assert not t.is_meta,('unrestored native buffer',name,t.shape);v=t.detach().cpu()
   assert list(v.shape)==list(t.shape);parent,_,key=name.rpartition('.');module=self.model.get_submodule(parent);setattr(module,key,v.to(self.device));buffers[name]=dict(shape=list(v.shape),dtype=str(v.dtype),CPU_analytic_SHA256=tensor_sha(v),actual_NPU_SHA256=tensor_sha(getattr(module,key)))
  self.model.llm.model.rotary_emb.original_inv_freq=self.model.llm.model.rotary_emb.inv_freq
  self.model.eval();assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters())
  now=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('MiniCPMO26_own_BF16_'+now+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  actual={n:dict(v,actual_NPU_SHA256=tensor_sha(dict(self.model.named_parameters(remove_duplicate=False))[n]),actual_NPU_device=str(self.device)) for n,v in expected.items()};assert all(v['actual_NPU_SHA256']==v['source_SHA256']==v['cast_SHA256'] for v in actual.values())
  self.weight_proof=self.home/'all_own_source_actual_NPU.json';write(self.weight_proof,dict(weights=actual,buffers=buffers,own_tied_head=False))
  self.front=json.loads(Path(self.p['own_frontend200']).read_bytes());assert len(self.front)==200
  self.index=0;self.report_index=0;self.details=[];self.handles=[];self.active=False;self.failed=False;self.profile_done=False;self.profiler=None;self.batches=[];self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json';self._install()
  libs={}
  import transformers,transformers.masking_utils,transformers.cache_utils,transformers.models.qwen2.modeling_qwen2
  for mod in [torch,torch_npu,transformers,transformers.masking_utils,transformers.cache_utils,transformers.models.qwen2.modeling_qwen2]:
   path=inspect.getfile(mod);libs[path]=sha(path)
  self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime='Own original Navit Siglip27 and own Resampler + own Qwen2_28',runtime_dtype='BF16',max_new_tokens=4096,actual_outer_batch_size=10,actual_inner_batch_size=1,attention='own native eager and causal native Qwen2; saved vLLM eager BF16 reconstruction boundary disclosed',use_cache=True,head_size=151700,tokenizer_size=len(self.tokenizer),protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,evidence_root=str(self.home),weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),runtime_library_source_SHA256=libs,coordinate_computation='Original CPU integer image bounds/tgt_sizes and CPU F32 analytic position construction; no learned CPU layer',historical_source_boundary=self.p['historical_source_boundary'])
 def predict(self,prompt,media_path=None):
  assert not self.failed,'Prior failure is preserved; no batch fallback NPU replay'
  i=self.index;self.frames=[];self.active=False
  try:
   ref=self.front[i];assert prompt==ref['actual_model_prompt'] and sha(media_path)==ref['media_file_SHA256'];text,v=self.inputs.prepare(prompt,media_path);assert text==ref['text']
   for n,t in flatten_tensors(v).items():assert tensor_sha(t)==ref['tensor_metadata'][n]['SHA256']
   reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
   self.arrays={'input_ids':v['input_ids'].numpy(),'original_own_HF_processor_mask_diagnostic':v['attention_mask'].numpy()}
   for j,t in enumerate(v['pixel_values'][0]):self.arrays['own_pixels_F32_%d'%j]=array(t);self.arrays['own_pixels_BF16_%d'%j]=array(t.to(torch.bfloat16))
   self.logits=[];self.head_inputs=[];self.deltas=[];self.activations={};self.counts={};self.previous_pairs=[];self.active=True
   data=dict(input_ids=v['input_ids'].to(self.device,dtype=torch.int64),pixel_values=[[t.to(self.device,dtype=torch.bfloat16) for t in v['pixel_values'][0]]],image_bound=v['image_bound'],tgt_sizes=v['tgt_sizes'])
   if not self.profile_done:
    self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.profiler.start()
   tokens=[];cache=None;N=len(ref['input_ids'])
   with torch.inference_mode():
    embeddings,_=self.model.get_vllm_embedding(data)
    for step in range(4096):
     length=N+step;pos=torch.arange(N,device=self.device).unsqueeze(0) if step==0 else torch.tensor([[length-1]],device=self.device)
     kw=dict(position_ids=pos,attention_mask=torch.ones(1,length,dtype=torch.int64,device=self.device),use_cache=True,logits_to_keep=1,return_dict=True)
     if step:kw.update(input_ids=torch.tensor([[tokens[-1]]],device=self.device),past_key_values=cache)
     else:kw.update(inputs_embeds=embeddings)
     out=self.model.llm(**kw);cache=out.past_key_values;token=int(out.logits[0,-1].argmax().item());tokens.append(token)
     if token in self.stops:break
   self.active=False;assert tokens==[f['argmax'] for f in self.frames]
   raw=self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
   self.arrays['all_original_full_head_logits']=np.stack(self.logits);self.arrays['all_head_inputs']=np.stack(self.head_inputs);self.arrays['native_cache_decode_delta']=np.concatenate(self.deltas,axis=3) if self.deltas else np.empty((28,2,4,0,128),dtype=np.uint16)
   final_cache=[[tensor_sha(t) for t in pair] for pair in self.previous_pairs];assert len(self.activations)>=57
   size=sum(t.nbytes for t in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
   ap=self.home/('sample_%05d.npz'%i);save_npz(ap,self.arrays);rp=self.home/('sample_%05d.json'%i)
   rec=dict(index=i,actual_model_prompt=prompt,query=text,media_path=str(media_path),media_SHA256=ref['media_file_SHA256'],input_ids=ref['input_ids'],generated_tokens=tokens,raw_output=raw,reference=ref['reference'],error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,original_budget_hit=len(tokens)==4096 and tokens[-1] not in self.stops,stop_ids=self.stops,compressed_length=self.compressed_length,final_cache_SHA256=final_cache,archive=str(ap),archive_SHA256=sha(ap),archive_bytes=ap.stat().st_size,lossless_uncompressed_bytes=size);write(rp,rec)
   detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(tokens),generated_tokens=tokens,raw_output=raw,error=None)
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.details.append(detail);self.index+=1;self.arrays={};self.logits=[];self.head_inputs=[];self.deltas=[];self.previous_pairs=[]
   LOG.info('MiniCPM-o-2_6 own sample %d, %d genuine forwards',i+1,len(tokens));return raw
  except BaseException:
   self.active=False;self.failed=True
   if self.profiler is not None:self.profiler.stop();self.profiler=None
   write(self.home/('sample_%05d_failed.json'%i),dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
 def batch_predict(self,batch):
  assert 1<=len(batch)<=10
  rec=dict(indices=list(range(self.index,self.index+len(batch))),status='running');self.batches.append(rec)
  try:result=[self.predict(p,m) for p,m in batch];rec['status']='completed';return result
  except BaseException:rec['status']='failed';raise
 def get_sample_execution(self,prompt,media_path):
  i=self.report_index;self.report_index+=1
  if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),actual_outer_batches=self.batches,first_prefill_profiler_SHA256=sha(self.profile),first_real_vision_prefill_profiler_SHA256={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()}))
  return self.details[i]
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  for h in getattr(self,'handles',[]):h.remove()
  self.model=None;self.previous_pairs=[];torch.npu.empty_cache()

 def _record(self,name,value,retain=True):
             assert torch.is_tensor(value) and value.device==self.device,(name,getattr(value,'device',None))
             v=value.detach().cpu().contiguous();raw=v.reshape(-1).view(torch.uint8).numpy().tobytes();meta=dict(shape=list(v.shape),dtype=str(v.dtype),SHA256=hashlib.sha256(raw).hexdigest())
             if retain:
                 assert name not in self.arrays,name
                 self.arrays[name]=v.view(torch.uint16).numpy() if v.dtype==torch.bfloat16 else v.numpy()
             return meta
 def _install(self):
             lm=self.model.llm
             def lm_pre(m,args,kw):
                 if not self.active:return
                 assert not args
                 n=len(self.frames);self.frame=dict(step=n,attention=[],input_ids=None,position_ids=None);self.tensor_cache={}
                 for key in ['input_ids','position_ids','cache_position','attention_mask']:
                     val=kw.get(key)
                     if torch.is_tensor(val):
                         self.frame[key]=self._record('f%d_%s'%(n+1,key),val)
                         if key in ['input_ids','position_ids','cache_position']:self.frame[key+'_values']=val.detach().cpu().tolist()
                 if n==0:
                     self.activations['prefill_inputs_embeds']=self._record('prefill_inputs_embeds',kw['inputs_embeds']);self.compressed_length=kw['inputs_embeds'].shape[1]
             def head_hook(m,args,out):
                 if self.active:
                     row=out[0,-1];assert row.shape==(151700,) and row.dtype==torch.bfloat16 and row.device==self.device
                     self.logits.append(array(row));self.head_inputs.append(array(args[0][0,-1]));self.frame['head_SHA256']=tensor_sha(row);self.frame['head_shape']=list(out.shape);self.frame['head_dtype']=str(row.dtype);self.frame['argmax']=int(row.argmax().item())
             def lm_post(m,args,kw,out):
                 if not self.active:return
                 n=len(self.frames);length=self.compressed_length+n;cache=out.past_key_values
                 assert cache is not None and len(cache.layers)==28
                 pieces=[];newpairs=[]
                 for i,layer in enumerate(cache.layers):
                     k,v=layer.keys,layer.values
                     assert k.shape==v.shape==(1,4,length,128) and k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
                     if n:
                         pk,pv=self.previous_pairs[i]
                         assert torch.equal(k[:,:,:-1],pk) and torch.equal(v[:,:,:-1],pv),(n,i,'native cache prefix changed')
                         pieces.extend((k[0,:,-1:],v[0,:,-1:]))
                     else:pieces.extend((k[0],v[0]))
                     newpairs.append((k,v))
                 packed=torch.stack(pieces).reshape(28,2,4,-1,128)
                 if n:self.deltas.append(array(packed))
                 else:self.arrays['native_cache_prefill']=array(packed)
                 self.previous_pairs=newpairs;self.frame['cache_seq_len']=length;self.frame['unchanged_prefix_exact_checked_layers']=28 if n else 0;self.frames.append(self.frame)
                 if self.profiler is not None and n==0:
                     self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None;self.profile_done=True
             self.handles.extend([lm.register_forward_pre_hook(lm_pre,with_kwargs=True),lm.lm_head.register_forward_hook(head_hook),lm.register_forward_hook(lm_post,with_kwargs=True)])
             for name,module in self.model.named_modules():
                 if re.fullmatch(r'(vpm.encoder.layers|llm.model.layers)\.\d+',name):
                     def stage_hook(m,args,out,name=name):
                         if not self.active:return
                         self.counts[name]=self.counts.get(name,0)+1
                         if self.frames:return
                         value=out[0] if isinstance(out,(tuple,list)) else out
                         key='stage_'+re.sub('[^a-zA-Z0-9_]','_',name);self.activations[key]=self._record(key,value,retain=self.index<3)
                     self.handles.append(module.register_forward_hook(stage_hook))
             for i,layer in enumerate(lm.model.layers):
                 def attn_pre(m,args,kw,i=i):
                     if not self.active:return
                     obs={'layer':i}
                     for key in ['attention_mask','position_ids','position_embeddings']:
                         values=kw.get(key)
                         if values is None:obs[key]=None;continue
                         values=values if isinstance(values,(tuple,list)) else [values];refs=[]
                         for j,value in enumerate(values):
                             assert torch.is_tensor(value),(key,type(value))
                             ident=(id(value),value.data_ptr(),tuple(value.shape),str(value.dtype))
                             if ident not in self.tensor_cache:
                                 name='f%d_attn_%s_%d_%d'%(len(self.frames)+1,key,i,j);meta=self._record(name,value);self.tensor_cache[ident]=dict(array=name,**meta)
                             refs.append(self.tensor_cache[ident])
                         obs[key]=refs
                     self.frame['attention'].append(obs)
                 self.handles.append(layer.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
             original=self.model.get_vllm_embedding
             def observe_own_visual(data):
                 result,visual=original(data)
                 if self.active:self.activations['own_visual_features']=self._record('own_visual_features',torch.cat(visual,dim=0))
                 return result,visual
             self.model.get_vllm_embedding=observe_own_visual

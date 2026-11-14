"""Own SmolVLM2-2.2B original HF FP16/4096 image evaluation on Ascend."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from functools import wraps
import hashlib,json,logging,os,shutil,traceback,zipfile
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from transformers import AutoModelForImageTextToText,AutoTokenizer
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.smol22_original_own_image_inputs_2109_20261111 import Smol22OwnImageInputs

MID='HuggingFaceTB/SmolVLM2-2.2B-Instruct';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def ts(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def array(t):return t.detach().cpu().contiguous().numpy()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def npz(p,values):
 with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for name,value in values.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(value),allow_pickle=False)

class Smol22OriginalHFNative(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):
  self.pfile=Path(protocol_file);self.psha=protocol_sha256;self.mid=protocol['model_id'];self.devices=npu_devices
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**kw):raise RuntimeError('Original own local weights required')
 def load(self,model_path,**kw):
  assert self.mid==MID and sha(self.pfile)==self.psha
  self.p=js(self.pfile);self.W=Path(model_path);assert str(self.W)==self.p['weight_path'] and len(self.devices)==1
  self.device=torch.device('npu:'+str(self.devices[0]));torch.npu.set_device(self.device);torch.set_num_threads(4)
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
  for kind in ('matmul','conv'):getattr(torch.npu,kind).allow_hf32=False
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,r in self.p['own_weight_files'].items():assert sha(self.W/n)==r['SHA256'],n
  for p,h in self.p['static_evidence_SHA256'].items():assert sha(p)==h,p
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
  self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('Smol22_original_HF_FP16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  # This image task does not instantiate the video processor, whose optional num2words is absent.
  # The independent own image frontend was checked byte-for-byte on all200 actual AA inputs.
  self.front=Smol22OwnImageInputs(self.W)
  self.tokenizer=AutoTokenizer.from_pretrained(str(self.W),local_files_only=True)
  self.model=AutoModelForImageTextToText.from_pretrained(str(self.W),dtype=torch.float16,local_files_only=True).eval().to(self.device)
  assert type(self.model).__name__=='SmolVLMForConditionalGeneration'
  c=self.model.config;assert c.text_config.num_hidden_layers==24 and c.vision_config.num_hidden_layers==27 and c.text_config.vocab_size==49280 and c.scale_factor==3
  assert c.text_config.num_attention_heads==c.text_config.num_key_value_heads==32 and c.text_config.head_dim==64
  assert {c._attn_implementation,c.text_config._attn_implementation,c.vision_config._attn_implementation}=={'sdpa'}
  assert self.model.generation_config.eos_token_id==49279 and self.model.generation_config.pad_token_id==2
  source=js(self.p['actual_AA_source_manifest']);expected={x['name']:x for x in source}
  params=dict(self.model.named_parameters(remove_duplicate=False));assert len(params)==len(expected)==657 and set(params)==set(expected)
  loaded={}
  for filename in self.p['own_weight_files']:
   with safe_open(self.W/filename,framework='pt',device='cpu') as f:
    for name in f.keys():
     value=f.get_tensor(name);ref=expected[name];actual=params[name]
     assert list(value.shape)==ref['shape']==list(actual.shape) and str(value.dtype)==ref['source_dtype']
     assert ts(value)==ref['source_SHA256'] and ts(value.half())==ref['cast_SHA256']==ref['actual_CUDA_SHA256'],name
     assert actual.device==self.device and actual.dtype==torch.float16 and ts(actual)==ref['cast_SHA256'],name
     loaded[name]=dict(ref,actual_NPU_SHA256=ts(actual),actual_NPU_device=str(actual.device))
  aliases={}
  for n,t in params.items():aliases.setdefault(t.data_ptr(),[]).append(n)
  self.weight_file=self.home/'own657_source_cast_actual_NPU.json';write(self.weight_file,dict(weights=loaded,physical_unique=len(aliases),aliases=[x for x in aliases.values() if len(x)>1]))
  self.refs=js(self.p['independent_R_CPU200'])['records'];assert len(self.refs)==200
  self.index=0;self.report_index=0;self.details=[];self.handles=[];self.failed=False;self.profile_done=False
  self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json'
  self._install()
  self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime_dtype='FP16',runtime='own SmolVLM2-2.2B original HF greedy generate',actual_outer_batch_size=10,actual_inner_batch_size=1,max_new_tokens=4096,attention='original HF sdpa for vision and text',use_cache='original generate resolved DynamicCache True; raw root config False retained',evidence_root=str(self.home),protocol_file=str(self.pfile),protocol_SHA256=self.psha,weight_evidence=str(self.weight_file),weight_evidence_SHA256=sha(self.weight_file),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),profiler_scope='first original vision and decoder prefill only',numeric_retention_scope='all samples full input/pixels/masks/generated tokens/full49280 head rows and every24KV SHA/shape/dtype; first3 complete KV prefill+deltas and all51 prefill layer values; later layer and vision SHA/shape/dtype/count. No claim later KV arrays reconstructable.',original_optional_video_dependency_not_needed_for_image_task=True)
  LOG.info('Smol22 own657 FP16 loaded on %s, original HF SDPA/4096/cache default',self.device)
 def _record(self,name,t,retain):
  assert t.device==self.device
  meta=dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=ts(t))
  if retain and name not in self.arrays:self.arrays[name]=array(t)
  return meta
 def _install(self):
  def pre(m,args,kw):
   assert not args
   self.current=dict(step=len(self.frames),input_ids=kw['input_ids'].detach().cpu().tolist(),input_shapes={},input_SHA256={},use_cache=kw.get('use_cache'),cache_class=type(kw.get('past_key_values')).__name__,attention=[])
   for k,v in kw.items():
    if torch.is_tensor(v):self.current['input_shapes'][k]=list(v.shape);self.current['input_SHA256'][k]=ts(v)
   self.current['positions']=kw['position_ids'].detach().cpu().tolist()
   assert self.current['use_cache'] is True and self.current['cache_class']=='DynamicCache'
  def post(m,args,kw,out):
   row=out.logits[0,-1];assert row.dtype==torch.float16 and row.shape==(49280,)
   self.heads.append(array(row));self.current.update(full_logits_shape=list(out.logits.shape),logits_dtype=str(out.logits.dtype),argmax=int(row.argmax().item()),last_full_head_SHA256=ts(row))
   cache=out.past_key_values;assert len(cache.layers)==24
   meta=[];blocks=[]
   for layer in cache.layers:
    k,v=layer.keys,layer.values;assert k.shape==v.shape and k.shape[:2]==(1,32) and k.shape[-1]==64 and k.dtype==v.dtype==torch.float16
    meta.append([dict(shape=list(x.shape),dtype=str(x.dtype),SHA256=ts(x)) for x in (k,v)])
    if self.index<3:blocks.extend((k[0] if not self.frames else k[0,:,-1:,:],v[0] if not self.frames else v[0,:,-1:,:]))
   self.current['cache']=meta
   if self.index<3:
    block=torch.stack(blocks).reshape(24,2,32,-1,64)
    if not self.frames:self.arrays['native_cache_prefill']=array(block)
    else:self.deltas.append(array(block))
   self.frames.append(self.current)
  self.handles.extend((self.model.register_forward_pre_hook(pre,with_kwargs=True),self.model.register_forward_hook(post,with_kwargs=True)))
  def observation(name):
   def hook(m,args,out):
    self.counts[name]=self.counts.get(name,0)+1
    if self.frames:return
    if not torch.is_tensor(out):out=out[0] if isinstance(out,(tuple,list)) else out.last_hidden_state
    self.activations[name]=self._record(name,out,self.index<3)
   return hook
  core=self.model.model
  self.handles.append(core.vision_model.register_forward_hook(observation('vision_final')))
  self.handles.append(core.connector.register_forward_hook(observation('connector')))
  def merged_pre(m,args,kw):
   if not self.frames:self.activations['merged_embeddings']=self._record('merged_embeddings',kw['inputs_embeds'],self.index<3)
  self.handles.append(core.text_model.register_forward_pre_hook(merged_pre,with_kwargs=True))
  for i,layer in enumerate(core.vision_model.encoder.layers):self.handles.append(layer.register_forward_hook(observation('vision_layer_%02d'%i)))
  for i,layer in enumerate(core.text_model.layers):
   self.handles.append(layer.register_forward_hook(observation('text_layer_%02d'%i)))
   def ampre(m,args,kw,i=i):
    mask=kw.get('attention_mask');record=dict(layer=i,mask=None,implicit_SDPA_causal=mask is None,positions=kw['position_ids'].detach().cpu().tolist() if torch.is_tensor(kw.get('position_ids')) else None)
    if mask is not None:
     name='causal_mask_'+ts(mask);record['mask']=name;self._record(name,mask,True)
    self.current['attention'].append(record)
   self.handles.append(layer.self_attn.register_forward_pre_hook(ampre,with_kwargs=True))
 def predict(self,prompt,media_path=None):
  i=self.index;ref=self.refs[i];assert prompt==ref['actual_model_prompt'] and Path(media_path).resolve()==Path(ref['R_media']).resolve() and sha(media_path)==ref['media_SHA256']
  values,text=self.front.prepare(prompt,media_path)
  assert text['actual_text']==ref['actual_text'] and text['grid']==ref['grid']
  for k,v in values.items():
   meta=ref['inputs'][k];assert list(v.shape)==meta['shape'] and hashlib.sha256(v.tobytes()).hexdigest()==meta['SHA256'],k
  assert values['input_ids'][0].tolist()==ref['input_ids']
  self.arrays=dict(values);self.frames=[];self.heads=[];self.deltas=[];self.activations={};self.counts={}
  assert shutil.disk_usage(self.home).free>3*1024**3+sum(v.nbytes for v in self.arrays.values())*4,'Original evidence reserve'
  inputs={k:torch.from_numpy(v.copy()).to(self.device) for k,v in values.items()}
  original=self.model.forward
  first=not self.profile_done
  if first:
   @wraps(original)
   def one_profile(*a,**kw):
    if self.profile_done:return original(*a,**kw)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:out=original(*a,**kw)
    prof.export_chrome_trace(str(self.profile));self.profile_done=True;return out
   self.model.forward=one_profile
  try:
   with torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=4096,do_sample=False,temperature=0.0)
   N=values['input_ids'].shape[1];gen=output[0,N:].detach().cpu().tolist();raw=self.tokenizer.decode(gen,skip_special_tokens=True).strip()
   assert gen and len(gen)<=4096 and len(self.frames)==len(gen)
   self.arrays['all_original_full_head_logits']=np.stack(self.heads)
   if i<3:self.arrays['native_cache_decode_delta']=np.concatenate(self.deltas,axis=3) if self.deltas else np.empty((24,2,32,0,64),dtype=np.float16)
   archive=self.home/('sample_%05d.npz'%i);npz(archive,self.arrays)
   record=dict(index=i,actual_model_prompt=prompt,actual_text=text['actual_text'],expanded_text=text['expanded_text'],grid=text['grid'],media_path=str(media_path),media_SHA256=ref['media_SHA256'],input_ids=values['input_ids'][0].tolist(),generated_tokens=gen,raw_output=raw,error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,original_budget_hit=len(gen)==4096,stop_ids=[49279],archive=str(archive),archive_SHA256=sha(archive),full_KV_and_layer_arrays_retained=i<3)
   rp=self.home/('sample_%05d.json'%i);write(rp,record)
   detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=gen,raw_output=raw,error=None)
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.details.append(detail);self.index+=1;self.arrays={};self.deltas=[];self.heads=[]
   LOG.info('Smol22 actual sample %d completed, %d native forwards, original HF FP16 SDPA',i,len(self.frames))
   return raw
  except Exception:
   self.failed=True;write(self.home/('sample_%05d_failed.json'%i),dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
  finally:
   if first:self.model.forward=original
 def batch_predict(self,batch):
  assert len(batch)<=10
  return [self.predict(prompt,media) for prompt,media in batch]
 def get_sample_execution(self,prompt,media_path):
  i=self.report_index;self.report_index+=1
  if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),first_prefill_profiler_SHA256=sha(self.profile)))
  return self.details[i]
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  for h in getattr(self,'handles',[]):h.remove()
  self.model=None;torch.npu.empty_cache()

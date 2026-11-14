"""Own Granite MLX original HF BF16/512 serial captioning on Ascend.

Raw MLX-named BF16 tensors are mapped/reordered without changing values.
This is disclosed reference BF16 computation, not native low-bit acceleration.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps
import hashlib,json,logging,os,shutil,traceback,zipfile
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from transformers import AutoModelForImageTextToText,Idefics3Config,GenerationConfig
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.granite_mlx_own_HF_frontend_v1_pil104_0239_20261112 import OwnGraniteMLXHFInputs
from universal_eval.model.adapters.granite_mlx_exact_mixed_BF16_buckets_v2_0239_20261112 import install_own_embedding_buckets
MID='ibm-granite/granite-docling-258M-mlx';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def ts(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def array(t):
 x=t.detach().cpu().contiguous()
 return x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def npz(p,values):
 with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for name,value in values.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(value),allow_pickle=False)

class GraniteMLXOriginalHFV3(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):self.pfile=Path(protocol_file);self.psha=protocol_sha256;self.mid=protocol['model_id'];self.devices=npu_devices
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**kw):raise RuntimeError('Original own local weights only')
 def load(self,model_path,**kw):
  assert self.mid==MID and sha(self.pfile)==self.psha;self.p=js(self.pfile);self.W=Path(model_path);assert str(self.W)==self.p['weight_path'] and self.devices==[4]
  self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4)
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
  for k in ('matmul','conv'):getattr(torch.npu,k).allow_hf32=False
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  assert sha(self.W/'model.safetensors')==self.p['own_weight_SHA256']
  for path,h in self.p['static_evidence_SHA256'].items():assert sha(path)==h,path
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('GraniteMLX_original_HF_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  self.front=OwnGraniteMLXHFInputs(self.W);aa_meta=js(self.p['actual_AA_metadata']);aa=js(self.p['actual_AA_source_manifest']);expected=aa['logical_parameters'];mapping=js(self.p['own_parameter_mapping'])
  cfg=Idefics3Config.from_dict(aa_meta['config']);self.model=AutoModelForImageTextToText.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
  params=dict(self.model.named_parameters(remove_duplicate=False));assert set(params)==set(expected)
  assert self.model.lm_head.weight is self.model.model.text_model.embed_tokens.weight,'Original A100 tied-head physical alias'
  mapped={}
  with safe_open(self.W/'model.safetensors',framework='pt',device='cpu') as f:
   assert set(f.keys())=={m['source_name'] for m in mapping}
   for m in mapping:
    value=f.get_tensor(m['source_name']);target=m['actual_AA_parameter']
    if m['permutation']:value=value.permute(*m['permutation']).contiguous()
    assert list(value.shape)==expected[target]['shape'] and ts(value.bfloat16())==m['BF16_SHA256']==expected[target]['SHA256'],target
    with torch.no_grad():params[target].copy_(value.bfloat16())
    mapped[target]=m
  assert set(mapped)==set(params) and len(params)==471
  self.model.generation_config=GenerationConfig.from_dict(aa_meta['generation_config']);assert self.model.generation_config.use_cache is False
  assert self.model.generation_config.eos_token_id==100257 and self.model.generation_config.pad_token_id==100257
  assert {cfg._attn_implementation,cfg.text_config._attn_implementation,cfg.vision_config._attn_implementation}=={'sdpa'}
  assert cfg.text_config.num_hidden_layers==30 and cfg.vision_config.num_hidden_layers==12 and cfg.text_config.vocab_size==100352 and cfg.scale_factor==4
  self.model=self.model.to(self.device);self.bucket_implementation=install_own_embedding_buckets(self.model);params=dict(self.model.named_parameters(remove_duplicate=False));loaded={}
  for name,t in params.items():
   assert t.dtype==torch.bfloat16 and t.device==self.device and ts(t)==expected[name]['SHA256'],name
   loaded[name]=dict(source_mapping=mapped[name],actual_AA_SHA256=expected[name]['SHA256'],actual_NPU_SHA256=ts(t),shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
  self.weight_file=self.home/'own471_source_actual_AA_NPU.json';write(self.weight_file,dict(logical_parameters=loaded,physical_unique=len({t.data_ptr() for t in params.values()}),head_tied_to_own_embedding=True,checkpoint_layout_conversion_only=True,original_extra_head_same_own_embedding=True))
  self.refs=js(self.p['independent_R_CPU200']);assert len(self.refs)==200
  self.index=0;self.report_index=0;self.details=[];self.handles=[];self.failed=False;self.profile_done=False;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json';self._install()
  self.lineage=dict(hardware='Ascend910B',devices=self.devices,runtime_dtype='BF16',runtime='own Granite MLX original HF serial caption greedy generate',actual_outer_batch_size=1,actual_inner_batch_size=1,max_new_tokens=512,attention='original actual HF SDPA vision and text',use_cache=False,original_AA_task_label='image-to-text',canonical_workbook_task_label='image-text-to-text',actual_task_semantics='original 1459 short-caption data and ROUGE-L, no extraction',evidence_root=str(self.home),protocol_file=str(self.pfile),protocol_SHA256=self.psha,weight_evidence=str(self.weight_file),weight_evidence_SHA256=sha(self.weight_file),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),profiler_scope='first original vision and decoder prefill only',numeric_retention_scope='all samples full input tensors/pixels/token/raw/ref/errors and full100352 BF16 head each actual forward; all positions/mask/cache metadata; first3 full42layer and connector prefill values, later layer metadata only. No KV values exist because original use_cache=False.',BF16_position_bucket_implementation=self.bucket_implementation,weight_disclosure='own BF16 tensor layout conversion and reference BF16 execution, no native quantized or MLX acceleration claim')
  LOG.info('Granite MLX own471/470physical BF16 loaded on NPU4; original HF/512/serial1/cacheFalse')
 def _record(self,name,t,retain):
  assert t.device==self.device;meta=dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=ts(t))
  if retain and name not in self.arrays:self.arrays[name]=array(t)
  return meta
 def _install(self):
  def pre(m,args,kw):
   assert not args;ids=kw['input_ids'];pos=kw.get('position_ids');self.current=dict(step=len(self.frames),input_ids=ids.detach().cpu().tolist(),positions=None if pos is None else pos.detach().cpu().tolist(),input_shapes={k:list(v.shape) for k,v in kw.items() if torch.is_tensor(v)},input_SHA256={k:ts(v) for k,v in kw.items() if torch.is_tensor(v)},use_cache=kw.get('use_cache'),cache_class=None if kw.get('past_key_values') is None else type(kw['past_key_values']).__name__,attention=[])
  def post(m,args,kw,out):
   row=out.logits[0,-1];assert row.dtype==torch.bfloat16 and row.shape==(100352,) and out.past_key_values is None
   self.heads.append(array(row));self.current.update(full_logits_shape=list(out.logits.shape),logits_dtype=str(row.dtype),last_full_head_SHA256=ts(row),argmax=int(row.argmax()),output_cache_is_None=True);self.frames.append(self.current)
  self.handles.extend((self.model.register_forward_pre_hook(pre,with_kwargs=True),self.model.register_forward_hook(post,with_kwargs=True)))
  def observe(name):
   def hook(m,args,out):
    self.counts[name]=self.counts.get(name,0)+1
    if self.frames:return
    value=out if torch.is_tensor(out) else (out[0] if isinstance(out,(tuple,list)) else out.last_hidden_state)
    self.activations[name]=self._record(name,value,self.index<3)
   return hook
  core=self.model.model;self.handles.extend((core.vision_model.register_forward_hook(observe('vision_final')),core.connector.register_forward_hook(observe('connector'))))
  def text_pre(m,args,kw):
   if not self.frames:self.activations['merged_embeddings']=self._record('merged_embeddings',kw['inputs_embeds'],self.index<3)
  self.handles.append(core.text_model.register_forward_pre_hook(text_pre,with_kwargs=True))
  for i,layer in enumerate(core.vision_model.encoder.layers):self.handles.append(layer.register_forward_hook(observe('vision_layer_%02d'%i)))
  for i,layer in enumerate(core.text_model.layers):
   self.handles.append(layer.register_forward_hook(observe('text_layer_%02d'%i)))
   def attention_pre(m,args,kw,i=i):
    mask=kw.get('attention_mask');record=dict(layer=i,mask=None,implicit_SDPA_causal=mask is None)
    if mask is not None:
     name='causal_mask_'+ts(mask);record['mask']=name;record['meta']=self._record(name,mask,self.index<3)
    self.current['attention'].append(record)
   self.handles.append(layer.self_attn.register_forward_pre_hook(attention_pre,with_kwargs=True))
 def predict(self,prompt,media_path=None):
  if self.failed:raise RuntimeError('Previous genuine NPU failure; no repeated NN on remaining fixed samples: '+self.failure)
  i=self.index;ref=self.refs[i];assert prompt==ref['prompt'] and Path(media_path).resolve()==Path(ref['R_media']).resolve() and sha(media_path)==ref['media_SHA256']
  inputs,text=self.front.prepare(prompt,media_path);assert text==ref['actual_text']
  for k,t in inputs.items():
   meta=ref['inputs'][k];assert list(t.shape)==meta['shape'] and str(t.dtype)==meta['dtype'] and ts(t)==meta['SHA256'],k
  self.arrays={k:array(v) for k,v in inputs.items()};self.frames=[];self.heads=[];self.activations={};self.counts={}
  assert shutil.disk_usage(self.home).free>4*1024**3+sum(v.nbytes for v in self.arrays.values())*6,'Original lossless evidence reserve'
  values={k:v.to(self.device) for k,v in inputs.items()};original=self.model.forward;first=not self.profile_done
  if first:
   @wraps(original)
   def one_profile(*a,**kw):
    if self.profile_done:return original(*a,**kw)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:out=original(*a,**kw)
    prof.export_chrome_trace(str(self.profile));self.profile_done=True;return out
   self.model.forward=one_profile
  try:
   with torch.inference_mode():output=self.model.generate(**values,max_new_tokens=512,do_sample=False,temperature=0.0)
   N=inputs['input_ids'].shape[1];gen=output[0,N:].detach().cpu().tolist();raw=self.front.processor.batch_decode(output[:,N:],skip_special_tokens=True)[0].strip();assert gen and len(gen)<=512 and len(self.frames)==len(gen)
   self.arrays['all_original_full_head_logits_BF16_u16']=np.stack(self.heads);archive=self.home/('sample_%05d.npz'%i);npz(archive,self.arrays)
   record=dict(index=i,actual_model_prompt=prompt,actual_text=text,media_path=str(media_path),media_SHA256=ref['media_SHA256'],input_ids=inputs['input_ids'][0].tolist(),generated_tokens=gen,raw_output=raw,error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,original_budget_hit=len(gen)==512,stop_ids=[100257],archive=str(archive),archive_SHA256=sha(archive),full42_layer_prefill_values_retained=i<3,use_cache=False)
   rp=self.home/('sample_%05d.json'%i);write(rp,record);detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(self.frames),generated_tokens=gen,raw_output=raw,error=None)
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.details.append(detail);self.index+=1;self.arrays={};self.heads=[];LOG.info('Granite MLX actual sample %d committed, %d BF16 native forwards',i,len(self.frames));return raw
  except Exception:
   self.failed=True;self.failure=traceback.format_exc();write(self.home/('sample_%05d_failed.json'%i),dict(index=i,actual_native_forwards=len(self.frames),error=self.failure));raise
  finally:
   if first:self.model.forward=original
 def batch_predict(self,batch):
  assert len(batch)==1;return [self.predict(*batch[0])]
 def get_sample_execution(self,prompt,media_path):
  i=self.report_index;self.report_index+=1
  if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),first_prefill_profiler_SHA256=sha(self.profile) if self.profile.exists() else None))
  if i<len(self.details):return self.details[i]
  assert self.failed
  error=dict(index=i,actual_model_prompt=prompt,media_path=str(media_path),actual_native_forwards=0,raw_output=None,generated_tokens=[],error=self.failure,blocked_after_first_real_NPU_error=True)
  p=self.home/('sample_%05d_blocked_execution.json'%i);write(p,error)
  return dict(error,actual_execution_file=str(p),actual_execution_SHA256=sha(p))
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  for h in getattr(self,'handles',[]):h.remove()
  self.model=None;torch.npu.empty_cache()

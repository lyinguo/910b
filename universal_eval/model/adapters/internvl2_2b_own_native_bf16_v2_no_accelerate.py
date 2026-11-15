"""Own InternVL2 native BF16/legacy24KV full NPU trace. No CPU learned forward."""
import sys,os,json,hashlib,zipfile,logging,shutil,traceback,inspect,importlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from universal_eval.model.interface import IModel
MID='OpenGVLab/InternVL2-2B';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def arr(t):
 t=t.detach().cpu().contiguous();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=digest(t))
def put(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def save(p,values):
 with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n,v in values.items():
   with z.open(n+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(v),allow_pickle=False)
class InternVL2OwnNative(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices;assert protocol['model_id']==MID and npu_devices==[4]
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**k):raise RuntimeError('Own original local checkpoint required')
 def load(self,model_path,**kw):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=js(self.protocol_file);self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,r in self.p['own_weight_files'].items():assert sha(self.W/n)==r['SHA256'] and (self.W/n).stat().st_size==r['bytes']
  self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4)
  for n in ['matmul','conv']:getattr(torch.npu,n).allow_hf32=False
  from universal_eval.model.internvl2_own_native_v2_meta_cpu_20261115.configuration_internvl_chat import InternVLChatConfig
  from universal_eval.model.internvl2_own_native_v2_meta_cpu_20261115.modeling_internvl_chat import InternVLChatModel
  from universal_eval.model.internvl2_own_native_v2_meta_cpu_20261115.tokenization_internlm2 import InternLM2Tokenizer
  from vllm.transformers_utils.processors.internvl import InternVLImageProcessor,InternVLProcessor
  cfg=InternVLChatConfig(**js(self.W/'config.json'));assert cfg.llm_config.rope_scaling==dict(factor=2.0,type='dynamic') and cfg.llm_config.num_hidden_layers==24
  with torch.device('meta'):self.model=InternVLChatModel(cfg,use_flash_attn=False)
  # Original native analytic RoPE constructor, outside the parameter-meta scope.
  for layer in self.model.language_model.model.layers:layer.attention._init_rope()
  assert all(not t.is_meta for t in self.model.buffers())
  expected=dict(self.model.named_parameters(remove_duplicate=False));source={}
  for n in self.p['own_weight_files']:
   with safe_open(str(self.W/n),framework='pt',device='cpu') as sf:
    for key in sf.keys():
     t=sf.get_tensor(key);assert key in expected and tuple(t.shape)==tuple(expected[key].shape) and t.dtype==torch.bfloat16
     target=t.to(self.device);parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));assert digest(t)==digest(target)
     source[key]=dict(shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=digest(t),actual_NPU_SHA256=digest(target),device=str(target.device))
  assert set(source)==set(expected)
  buffers={n:record(t) for n,t in self.model.named_buffers()};self.model=self.model.eval().to(self.device)
  for n,t in self.model.named_buffers():assert t.device==self.device and record(t)==buffers[n]
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters())
  assert self.model.language_model.output.weight is not self.model.language_model.model.tok_embeddings.weight
  self.tokenizer=InternLM2Tokenizer.from_pretrained(str(self.W),local_files_only=True);self.processor=InternVLProcessor(InternVLImageProcessor(image_size=448,min_dynamic_patch=1,max_dynamic_patch=12,dynamic_image_size=True,use_thumbnail=True),self.tokenizer,image_seq_length=256)
  self.stop=js(self.W/'generation_config.json')['eos_token_id'];assert self.stop==[92542,92543]
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('InternVL2_2B_own_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  self.weights=self.home/'own_source_and_actual_NPU.json';put(self.weights,dict(model_id=MID,weights=source,analytic_original_buffers=buffers,rope=dict(type='dynamic',factor=2.0,base=1000000,max_position=32768),no_substituted_head_or_embedding=True))
  self.front=[json.loads(s) for s in Path(self.p['own_frontend200']).read_text().splitlines()];self.index=self.report_index=0;self.details=[];self.outer=[];self.failed=False;self.profile_done=False;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch();self.handles=[];self._install()
  libs={}
  for n in ['torch','torch_npu','transformers']:
   f=inspect.getfile(importlib.import_module(n));libs[f]=sha(f)
  self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own original InternVision24 + InternLM2_24 eager BF16; native FP32 softmax/logit cast and analytic RoPE buffers preserved',runtime_dtype='BF16',head_size=92553,LM_layers=24,vision_layers=24,max_new_tokens=2048,configured_outer_batch_size=10,actual_inner_batch_size=1,use_cache=True,EOS=self.stop,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),actual_batch_manifest=str(self.home/'completed_trace.json'),profiler_dir=str(self.home/'profiler'),runtime_library_source_SHA256=libs,historical_source_boundary=self.p['historical_source_boundary'],no_A100_tensor_capture_claimed=True)
 def _save_tensor(self,name,t,retain=True):
  assert t.device==self.device
  meta=record(t)
  if retain and name not in self.values:self.values[name]=arr(t)
  return meta
 def _install(self):
  def pre(m,args,kw):
   self.frame=dict(step=len(self.frames),positions=kw['position_ids'].detach().cpu().tolist(),layers=[])
  def post(m,args,kw,out):
   row=out.logits[0,-1];assert row.dtype==torch.float32 and row.numel()==92553 and bool(torch.isfinite(row).all());self.heads.append(arr(row));self.frame['argmax']=int(row.argmax());cache=out.past_key_values;assert len(cache)==24;N=self.N+len(self.frames);slices=[]
   for j,(k,v) in enumerate(cache):
    assert k.shape==v.shape==(1,8,N,128) and k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
    if self.previous:
     assert torch.equal(k[:,:,:-1],self.previous[j][0]) and torch.equal(v[:,:,:-1],self.previous[j][1]),'Original native cache prefix changed'
    slices.append(torch.stack([k[0] if not self.frames else k[0,:,-1:],v[0] if not self.frames else v[0,:,-1:]]))
   block=torch.stack(slices)
   if not self.frames:self.values['native_cache_prefill']=arr(block)
   else:self.deltas.append(arr(block))
   self.previous=cache;self.frame.update(cache_length=N,all24_native_cache_prefixes_equal=True,appended_or_initial_KV=record(block));self.frames.append(self.frame)
  lm=self.model.language_model;self.handles.extend([lm.register_forward_pre_hook(pre,with_kwargs=True),lm.register_forward_hook(post,with_kwargs=True)])
  for j,layer in enumerate(lm.model.layers):
   def attn(m,args,kw,j=j):
    mask=kw.get('attention_mask');name=None
    if mask is not None:
     cachekey=(mask.data_ptr(),tuple(mask.shape),mask.dtype)
     if cachekey not in self.maskcache:
      name='mask_'+str(len(self.maskcache));self.maskcache[cachekey]=name;self._save_tensor(name,mask)
     else:name=self.maskcache[cachekey]
    self.frame['layers'].append(dict(layer=j,mask=name,positions=kw['position_ids'].detach().cpu().tolist(),past_length=0 if kw.get('past_key_value') is None else kw['past_key_value'][0].shape[2]))
   self.handles.append(layer.attention.register_forward_pre_hook(attn,with_kwargs=True))
   def stage(m,args,out,j=j):
    t=out[0];name='LM_layer_'+str(j);self.counts[name]=self.counts.get(name,0)+1
    if not self.frames:self.activations[name]=self._save_tensor(name,t,self.index<3)
   self.handles.append(layer.register_forward_hook(stage))
 def predict(self,prompt,media_path=None):
  from PIL import Image
  if self.failed:raise RuntimeError('Prior batch failed; no additional NN executed')
  i=self.index;ref=self.front[i];assert prompt==ref['original_model_prompt'] and str(Path(media_path).resolve())==ref['media_path'] and sha(media_path)==ref['image_SHA256']
  got=self.processor(text=ref['chat'],images=Image.open(media_path).convert('RGB'),return_tensors='pt');ids=got['input_ids'];pixels=got['pixel_values_flat'];assert ids[0].tolist()==ref['input_ids'] and record(pixels)==ref['pixels_F32'] and record(pixels.bfloat16())==ref['pixels_BF16']
  reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
  self.values={};self.frames=[];self.heads=[];self.deltas=[];self.previous=None;self.activations={};self.counts={};self.maskcache={};self.N=ids.shape[1];tokens=ids[0].tolist();profile=None
  try:
   with torch.inference_mode():
    ids=ids.to(self.device);pixel=pixels.to(self.device,dtype=torch.bfloat16);self.values['input_ids']=arr(ids);self.values['pixel_values']=arr(pixel);mask=got['attention_mask'].to(self.device);self.values['attention_mask']=arr(mask)
    if not self.profile_done:
     profile=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));profile.start()
    features=self.model.extract_feature(pixel);text=self.model.language_model.get_input_embeddings()(ids);merged=text.clone();selected=ids==92546;assert int(selected.sum())==features.numel()//2048;merged[selected]=merged[selected]*0.0+features.reshape(-1,2048)
    for name,t in [('vision_features',features),('text_embeddings',text),('merged_embeddings',merged)]:self.activations[name]=self._save_tensor(name,t)
    for k in range(2048):
     positions=torch.arange(self.N,device=self.device).unsqueeze(0) if k==0 else torch.tensor([[self.N+k-1]],device=self.device)
     kwargs=dict(attention_mask=torch.ones((1,self.N+k),device=self.device,dtype=torch.long),position_ids=positions,past_key_values=self.previous,use_cache=True,return_dict=True)
     kwargs.update(inputs_embeds=merged) if k==0 else kwargs.update(input_ids=torch.tensor([[tokens[-1]]],device=self.device))
     out=self.model.language_model(**kwargs);token=self.frames[-1]['argmax'];tokens.append(token)
     if profile is not None:profile.stop();profile=None;self.profile_done=True
     if token in self.stop:break
   final={str(j):dict(K=record(k),V=record(v)) for j,(k,v) in enumerate(self.previous)};self.values['all_full_head_logits']=np.stack(self.heads);self.values['generated_tokens']=np.array(tokens,dtype=np.int64)
   self.values['native_cache_appends']=np.stack(self.deltas) if self.deltas else np.empty((0,24,2,8,1,128),dtype=np.uint16)
   size=sum(v.nbytes for v in self.values.values());assert shutil.disk_usage(self.home).free>reserve+size
   archive=self.home/f'sample_{i:05d}.npz';save(archive,self.values);raw=self.tokenizer.decode(tokens[self.N:],skip_special_tokens=True).strip();e=dict(index=i,actual_model_prompt=prompt,media_path=str(media_path),reference=ref['reference'],input_ids=tokens[:self.N],generated_tokens=tokens,raw_output=raw,error=None,actual_native_forwards=len(self.frames),frames=self.frames,activations=self.activations,activation_counts=self.counts,final_native_cache_SHA256=final,archive=str(archive),archive_bytes=archive.stat().st_size,archive_SHA256=sha(archive),original_budget_hit=len(self.frames)==2048 and tokens[-1] not in self.stop)
   f=self.home/f'sample_{i:05d}.json';put(f,e);detail=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),actual_native_forwards=len(self.frames),raw_output=raw,error=None,generated_tokens=tokens)
   self.details.append(detail)
   with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.index+=1;self.values={};self.previous=None;LOG.info('InternVL2 own NPU %d/200 forwards=%d archive=%d',self.index,len(self.frames),archive.stat().st_size);return raw
  except BaseException:
   self.failed=True
   if profile is not None:profile.stop()
   failure=dict(index=i,actual_model_prompt=prompt,reference=ref['reference'],generated_tokens=tokens,committed_native_forwards=len(self.frames),error=traceback.format_exc(),frames=self.frames)
   if self.values:
    f=self.home/f'sample_{i:05d}_failed_partial.npz';save(f,self.values);failure.update(partial_archive=str(f),partial_archive_SHA256=sha(f))
   failed_path=self.home/f'sample_{i:05d}_failed.json';put(failed_path,failure)
   detail=dict(index=i,actual_execution_file=str(failed_path),actual_execution_SHA256=sha(failed_path),actual_native_forwards=len(self.frames),raw_output=None,error=failure['error'],generated_tokens=tokens)
   self.details.append(detail)
   with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.index+=1;raise
 def batch_predict(self,batch):
  dispatch=dict(indices=list(range(self.index,self.index+len(batch))),status='started');self.outer.append(dispatch)
  try:values=[self.predict(p,m) for p,m in batch];dispatch['status']='completed';return values
  except BaseException:
   dispatch['status']='failed';self.failed=True
   while self.index<dispatch['indices'][-1]+1:
    i=self.index;ref=self.front[i];f=self.home/f'sample_{i:05d}_not_executed.json';error='Not executed after prior original batch failure'
    put(f,dict(index=i,reference=ref['reference'],actual_model_prompt=ref['original_model_prompt'],error=error,actual_native_forwards=0,not_executed=True))
    detail=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),actual_native_forwards=0,raw_output=None,error=error,generated_tokens=[])
    self.details.append(detail)
    with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(detail,ensure_ascii=False)+'\n')
    self.index+=1
   raise
 def get_execution_lineage(self):return self.lineage
 def get_sample_execution(self,prompt,media_path):
  if self.report_index==0:
   files={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()};put(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,actual_outer_batches=self.outer,first_real_vision_prefill_profiler_SHA256=files,sample_trace_SHA256=sha(self.trace)))
  d=self.details[self.report_index];self.report_index+=1;return d
 def unload(self):self.model=None;torch.npu.empty_cache()

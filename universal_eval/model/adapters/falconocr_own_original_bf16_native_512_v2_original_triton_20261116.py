"""Own Falcon-OCR, saved plain/category512 BF16 generation on real NPU.

The own model's generate, sampling, budget-rounding, cache and decoding source
are unchanged. Hooks preserve complete operands and outputs outside reports.
"""
import os,json,hashlib,zipfile,logging,shutil,traceback,inspect,importlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from transformers import AutoTokenizer
from universal_eval.model.interface import IModel
MID='tiiuae/Falcon-OCR';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def arr(t):
 t=t.detach().cpu().contiguous();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def rec(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=digest(t))
def put(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def save(p,values):
 with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n,v in values.items():
   with z.open(n+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(v),allow_pickle=False)

class FalconOCROwnOriginal512V2OriginalTriton(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):
  assert protocol['model_id']==MID and npu_devices==[4];self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.handles=[];self._owns_process_group=False
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**k):raise RuntimeError('Only own original local Falcon checkpoint is authorized')
 def load(self,model_path,**kw):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=js(self.protocol_file);self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,v in self.p['own_weight_files'].items():assert sha(self.W/n)==v['SHA256'] and (self.W/n).stat().st_size==v['bytes']
  self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4)
  for n in ['matmul','conv']:getattr(torch.npu,n).allow_hf32=False
  from universal_eval.model.falconocr_own_npu_v2_original_triton_20261116.configuration_falcon_ocr import FalconOCRConfig
  from universal_eval.model.falconocr_own_npu_v2_original_triton_20261116.modeling_falcon_ocr import FalconOCRForCausalLM
  cfg=FalconOCRConfig(**js(self.W/'config.json'));cfg._name_or_path=str(self.W)
  assert (cfg.dim,cfg.n_layers,cfg.n_heads,cfg.n_kv_heads,cfg.head_dim,cfg.vocab_size,cfg.max_seq_len)==(768,22,16,8,64,65536,8192)
  self.model,info=FalconOCRForCausalLM.from_pretrained(str(self.W),config=cfg,local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=False,output_loading_info=True)
  assert not any(info.get(n) for n in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
  probe=js(self.p['own_source_dtype_probe']);state=self.model.state_dict();assert set(state)==set(probe['source']) and len(state)==115
  source={}
  with safe_open(str(self.W/'model.safetensors'),framework='pt',device='cpu') as sf:
   for n in sf.keys():
    t=sf.get_tensor(n);actual=state[n];assert str(actual.dtype)==probe['source'][n]['original_saved_BF16_HF_loaded_CPU_dtype_diagnostic']['dtype'] and torch.equal(t.to(actual.dtype),actual)
    source[n]=dict(source=rec(t),original_HF_loaded_cast=rec(actual),is_parameter=probe['source'][n]['is_parameter'])
  del state
  self.model=self.model.eval().to(self.device);self.model._ensure_device_buffers();self.model._tokenizer=AutoTokenizer.from_pretrained(str(self.W),trust_remote_code=True,local_files_only=True);self.tokenizer=self.model._get_tokenizer();assert len(self.tokenizer)==65536
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters()) and self.model.freqs_cis_golden.dtype==torch.float32 and self.model.freqs_cis.dtype==torch.float32
  for n,t in self.model.state_dict().items():
   assert t.device==self.device and rec(t)==source[n]['original_HF_loaded_cast'];source[n]['actual_NPU']=rec(t);source[n]['device']=str(t.device)
  self.analytic=rec(self.model.freqs_cis)
  import torch.distributed as dist
  self._owns_process_group=not dist.is_initialized()
  if self._owns_process_group:dist.init_process_group('gloo',init_method=f'file:///tmp/910b_FalconOCR_{os.getpid()}',rank=0,world_size=1)
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('FalconOCR_own_BF16_native512_original_Triton_v2_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  self.weights=self.home/'own_source_cast_actual_NPU.json';put(self.weights,dict(model_id=MID,source=source,source_tensor_count=115,actual_NPU_source_tensor_count=115,own_F32_golden_buffer=True,original_intrinsic_F32_RoPE_and_original_Triton_BF16_intermediate_gate_preserved=True,analytic_freqs=self.analytic,no_source_or_head_substitution=True))
  self.front=[json.loads(x) for x in Path(self.p['own_frontend200']).read_text().splitlines()];assert len(self.front)==200
  self.index=self.report_index=0;self.details=[];self.failed=False;self.profile_done=False;self.prof=None;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch();self._install()
  libs={inspect.getfile(importlib.import_module(n)):sha(inspect.getfile(importlib.import_module(n))) for n in ['torch','torch_npu','transformers','tokenizers','einops']}
  self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own Falcon22-layer mixed image/text BF16; original native expanded16-head cache, actual CANN attention and original separate sinks; F32 learned golden buffer preserved',runtime_dtype='BF16 with original F32 persistent frequency buffer and intrinsic F32 arithmetic',head_size=65536,LM_layers=22,vision='Own image patch projector inside the same22 layers; no separate vision/audio tower claimed',configured_max_new_tokens=512,original_effective_budget='S=ceil((input_length+512)/128)*128; run original while cache.pos<S',configured_outer_batch_size=8,actual_inner_batch_size=1,use_cache=True,original_algorithm='greedy_inference',original_category='plain',original_seed=42,original_temperature=0.0,original_compile=False,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),actual_batch_manifest=str(self.home/'completed_trace.json'),profiler_dir=str(self.home/'profiler'),runtime_library_source_SHA256=libs,historical_source_boundary=self.p['historical_source_boundary'],no_A100_tensor_capture_claimed=True,activation_retention='Every sample full original inputs, patches, projector, merged input, every65536-logit head and head input, complete22-layer prefill cache plus lossless appends and final full cache SHA; first3 full prefill layer/QKV values, every layer/step full value SHA and count; original extra forward after terminal token retained')
 def _value(self,n,t,retain=True):
  assert t.device==self.device;meta=rec(t)
  if retain:self.values[n]=arr(t)
  self.activations[n]=meta;return meta
 def _install(self):
  def before(m,args,kw):
   step=len(self.frames);self.step=step;tokens=kw['tokens'];cache=kw['kv_cache'];self.cache=cache;self.current_tokens=tokens.detach().clone();self.cache_before=cache.get_pos()
   self._value(f'tokens_{step}',tokens)
   if step==0:
    assert tokens.tolist()==[self.front[self.index]['input_ids']]
    for key in ['rope_pos_t','rope_pos_hw','pixel_values','pixel_mask']:self._value('input_'+key,kw[key])
   self.counts.append(dict(LM_layers=0,QKV_layers=0,FFN_gates=0))
  def after(m,args,kw,out):
   step=self.step;assert out.shape==(1,self.current_tokens.shape[1],65536) and out.dtype==torch.bfloat16 and bool(torch.isfinite(out).all());self._value('full_head_'+str(step),out[0,-1],False);self.heads.append(arr(out[0,-1]));cache=self.cache;pos=cache.get_pos();new=cache.kv_cache[...,:pos,:]
   assert new.shape[:4]==(22,2,1,16) and new.shape[-1]==64
   if self.cache_before:
    assert torch.equal(new[...,:self.cache_before,:],self.prefix),'Native KV prefix changed'
    self.kv_appends.append(arr(new[...,self.cache_before:pos,:]))
   else:self._value('kv_prefill',new)
   self.prefix=new.clone();self.frames.append(dict(step=step,input_tokens=self.current_tokens.detach().cpu().tolist(),cache_before=self.cache_before,cache_after=pos,native_cache_prefix_unchanged=True,native_cache_dtype=str(new.dtype),full_cache_SHA256=digest(new) if self.index<3 or step==0 else None,pos_t=cache.pos_t.detach().cpu().tolist(),LM_layers=self.counts[-1]['LM_layers'],QKV_layers=self.counts[-1]['QKV_layers'],original_Triton_FFN_gate_layers=self.counts[-1]['FFN_gates']))
   assert self.counts[-1]==dict(LM_layers=22,QKV_layers=22,FFN_gates=22)
   if self.prof is not None:torch.npu.synchronize();self.prof.stop();self.prof=None;self.profile_done=True
  # Original generate calls self.forward directly, so nn.Module top-level hooks
  # do not fire. This instance wrapper calls the original bound method exactly.
  self.original_forward=self.model.forward
  def observed_forward(*args,**kw):
   before(self.model,args,kw);out=self.original_forward(*args,**kw);after(self.model,args,kw,out);return out
  self.model.forward=observed_forward
  def embed(m,args,out):
   if self.step==0:self._value('text_embeddings',out)
  self.handles.append(self.model.tok_embeddings.register_forward_hook(embed))
  def image(m,args,out):self._value('valid_image_patch_projector_input',args[0]);self._value('image_projector_output',out)
  self.handles.append(self.model.img_projector.register_forward_hook(image))
  def head(m,args):self.head_inputs.append(arr(args[0][0,-1]))
  self.handles.append(self.model.output.register_forward_pre_hook(head))
  for j,layer in enumerate(self.model.layers.values()):
   def layer_pre(m,args,kw,j=j):
    if j==0:
     if self.step==0:self._value('merged_embeddings',args[0])
     self._value('freqs_cis_'+str(self.step),kw['freqs_cis'])
     if kw['freqs_cis_2d'] is not None:self._value('freqs_cis_golden_prefill',kw['freqs_cis_2d'])
     mask=kw['attention_masks'];self._value('allowed_attention_mask_'+str(self.step),mask.allowed(args[0].shape[1],self.cache_before+args[0].shape[1]))
   def layer_post(m,args,out,j=j):
    self._value(f'layer_{j}_step_{self.step}',out,self.index<3 and self.step==0);self.counts[-1]['LM_layers']+=1
   def qkv(m,args,out,j=j):
    keep=self.index<3 and self.step==0;self._value(f'QKV_input_{j}_step_{self.step}',args[0],keep);self._value(f'QKV_{j}_step_{self.step}',out,keep);self.counts[-1]['QKV_layers']+=1
   def FFN_packed(m,args,out,j=j):self._value(f'FFN_packed_{j}_step_{self.step}',out,self.index<3 and self.step==0)
   def FFN_gate(m,args,j=j):
    self._value(f'FFN_original_Triton_gate_{j}_step_{self.step}',args[0],self.index<3 and self.step==0);self.counts[-1]['FFN_gates']+=1
   self.handles.extend([layer.register_forward_pre_hook(layer_pre,with_kwargs=True),layer.register_forward_hook(layer_post),layer.attention.wqkv.register_forward_hook(qkv)])
   self.handles.extend([layer.feed_forward.w13.register_forward_hook(FFN_packed),layer.feed_forward.w2.register_forward_pre_hook(FFN_gate)])
 def predict(self,prompt,media_path=None):
  from PIL import Image
  i=self.index;ref=self.front[i];self.values={};self.activations={};self.frames=[];self.counts=[];self.heads=[];self.head_inputs=[];self.kv_appends=[];self.prefix=None;self.cache=None
  try:
   assert not self.failed,'Prior real model failure; further NN skipped'
   from universal_eval.algorithm.inference import GreedyInference
   assert prompt==GreedyInference._add_answer_instruction(ref['prompt']) and str(Path(media_path).resolve())==ref['media_path'] and sha(media_path)==ref['image_SHA256']
   reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
   if not self.profile_done:
    self.prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.prof.start()
   with torch.inference_mode():result=self.model.generate(Image.open(media_path).convert('RGB'),category='plain',max_new_tokens=512,temperature=0.0,compile=False)
   assert isinstance(result,list) and len(result)==1 and self.frames
   generated=[f['input_tokens'][0][0] for f in self.frames[1:]];assert all(len(f['input_tokens'][0])==1 for f in self.frames[1:])
   raw=self.tokenizer.decode(generated,skip_special_tokens=False).replace('<|end_of_query|>','').replace('<|end_of_text|>','').strip();assert raw==result[0]
   self.values['head_inputs']=np.stack(self.head_inputs);self.values['all_full_heads']=np.stack(self.heads);self.values['kv_decode_appends']=np.stack(self.kv_appends) if self.kv_appends else np.empty((0,22,2,1,16,1,64),dtype=np.uint16)
   S=ref['original_allocation_S'];stop_ids=ref['original_stop_ids'];budget=self.cache.get_pos()==S and (not generated or generated[-1] not in stop_ids);assert self.cache.kv_shape[-2]==S and (budget or generated[-1] in stop_ids)
   final_cache=rec(self.cache.kv_cache[...,:self.cache.get_pos(),:]);assert len(self.frames)==len(generated)+1
   size=sum(a.nbytes for a in self.values.values());assert shutil.disk_usage(self.home).free>reserve+size;archive=self.home/f'sample_{i:05d}.npz';save(archive,self.values)
   e=dict(index=i,actual_model_prompt_ignored_by_original_Falcon=prompt,original_image_prompt=ref['actual_image_prompt'],media_path=str(media_path),reference=ref['reference'],input_ids=ref['input_ids'],generated_tokens=generated,raw_output=raw,error=None,actual_native_forwards=len(self.frames),full_head_count=len(self.heads),configured_max_new_tokens=512,original_effective_decode_slot_limit=ref['original_effective_decode_slot_limit'],original_allocation_S=S,original_seed=42,original_stop_ids=stop_ids,original_budget_hit=budget,stop_reason='original_rounded_allocation_budget' if budget else 'original_stop_token',native_final_valid_cache=final_cache,native_KV_storage_shape=list(self.cache.kv_shape),all_native_cache_prefixes_verified_unchanged=True,frames=self.frames,activations=self.activations,first3_full_prefill_layer_values_retained=i<3,archive=str(archive),archive_bytes=archive.stat().st_size,archive_SHA256=sha(archive))
   self._finish(e);self.values={};self.prefix=None;self.cache=None;LOG.info('Falcon-OCR own original BF16 NPU %d/200 forwards=%d tokens=%d archive=%d',self.index,len(self.frames),len(generated),archive.stat().st_size);return raw
  except BaseException:
   self.failed=True
   if self.prof is not None:self.prof.stop();self.prof=None
   e=dict(index=i,media_path=str(media_path),reference=ref['reference'],input_ids=ref['input_ids'],generated_tokens=[f['input_tokens'][0][0] for f in self.frames[1:]],raw_output=None,error=traceback.format_exc(),actual_native_forwards=len(self.frames),frames=self.frames,activations=self.activations)
   if self.values:
    f=self.home/f'sample_{i:05d}_failed_partial.npz';save(f,self.values);e.update(partial_archive=str(f),partial_archive_SHA256=sha(f))
   self._finish(e);raise
 def _finish(self,e):
  f=self.home/f'sample_{self.index:05d}.json';put(f,e);d=dict(index=self.index,actual_execution_file=str(f),actual_execution_SHA256=sha(f),actual_native_forwards=e['actual_native_forwards'],input_ids=e['input_ids'],generated_tokens=e['generated_tokens'],raw_output=e['raw_output'],error=e['error'],original_budget_hit=e.get('original_budget_hit',False),actual_generation_invocations=1)
  self.details.append(d)
  with self.trace.open('a',encoding='utf-8') as out:out.write(json.dumps(d,ensure_ascii=False)+'\n')
  self.index+=1;files={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()};put(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,actual_outer_batches=[dict(indices=list(range(n,min(n+8,self.index))),status='completed' if not self.failed else 'contains_error') for n in range(0,self.index,8)],first_real_mixed_image_prefill_profiler_SHA256=files,sample_trace_SHA256=sha(self.trace),actual_inner_serial1=True))
 def batch_predict(self,batch):return [self.predict(prompt,media_path) for prompt,media_path in batch]
 def get_execution_lineage(self):return self.lineage
 def get_sample_execution(self,prompt,media_path):
  d=self.details[self.report_index];self.report_index+=1;return d
 def unload(self):
  for h in self.handles:h.remove()
  self.model=None
  import torch.distributed as dist
  if self._owns_process_group and dist.is_initialized():dist.destroy_process_group()
  torch.npu.empty_cache()

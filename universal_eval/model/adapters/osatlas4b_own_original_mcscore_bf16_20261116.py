"""OS-Atlas own BF16, original one-forward MCScore; observation never generates.
The original score_choice method is invoked unchanged, with own real NPU model.
"""
import sys,os,json,hashlib,zipfile,logging,shutil,traceback,inspect,importlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.high_nine_osatlas_reference import OSAtlasMCScore
from universal_eval.model.adapters.high_nine_internvl_reference import InternVLAdapter
MID='OS-Copilot/OS-Atlas-Base-4B';LOG=logging.getLogger(__name__)
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
class OSAtlas4BOwnOriginalMCScore(IModel):
 _SCORE_TAIL=OSAtlasMCScore._SCORE_TAIL
 _preprocess_internvl=InternVLAdapter._preprocess_internvl
 _build_internvl_query=InternVLAdapter._build_internvl_query
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):
  self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices;assert protocol['model_id']==MID and npu_devices==[4];self.handles=[];self._owns_process_group=False
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**k):raise RuntimeError('Own original local checkpoint required')
 def predict(self,*a,**k):raise RuntimeError('Original OS-Atlas protocol is mc_score, never generation')
 def load(self,model_path,**kw):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=js(self.protocol_file);self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,r in self.p['own_weight_files'].items():assert sha(self.W/n)==r['SHA256'] and (self.W/n).stat().st_size==r['bytes']
  self.device=self._device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4)
  for n in ['matmul','conv']:getattr(torch.npu,n).allow_hf32=False
  from universal_eval.model.osatlas4b_own_original_mcscore_20261116.configuration_internvl_chat import InternVLChatConfig
  from universal_eval.model.osatlas4b_own_original_mcscore_20261116.modeling_internvl_chat import InternVLChatModel
  from transformers import AutoTokenizer
  cfg=InternVLChatConfig(**js(self.W/'config.json'))
  assert cfg.llm_config.num_hidden_layers==32 and cfg.llm_config.vocab_size==32022 and cfg.llm_config.hidden_size==3072 and cfg.llm_config.use_cache is False
  assert cfg.vision_config.drop_path_rate==0.1 and cfg.max_dynamic_patch==6 and cfg.template=='phi3-chat'
  cfg.vision_config.use_flash_attn=False;cfg.llm_config._attn_implementation='eager'
  with torch.device('meta'):self.model=InternVLChatModel(cfg)
  for layer in self.model.language_model.model.layers:layer.self_attn._init_rope()
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
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters()) and all(not m.training for m in self.model.modules())
  assert self.model.language_model.lm_head.weight is not self.model.language_model.model.embed_tokens.weight
  self.tokenizer=self._tokenizer=AutoTokenizer.from_pretrained(str(self.W),local_files_only=True,trust_remote_code=True)
  self._model=self.model;assert len(self.tokenizer)==32022
  import torch.distributed as dist
  self._owns_process_group=not dist.is_initialized()
  if self._owns_process_group:dist.init_process_group('gloo',init_method=f'file:///tmp/910b_OSAtlas_own_mcscore_{os.getpid()}',rank=0,world_size=1)
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('OSAtlas4B_own_original_MCScore_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  self.weights=self.home/'own_source_and_actual_NPU.json';put(self.weights,dict(model_id=MID,weights=source,analytic_original_buffers=buffers,rope=self.p['native_rope'],no_substituted_head_or_embedding=True,own_original_drop_path_rate=0.1,all_modules_eval=True,own_original_cache=False,own_original_score_choice_source=self.p['original_score_choice_source']))
  self.front=[json.loads(s) for s in Path(self.p['own_frontend200']).read_text().splitlines()];self.index=self.report_index=0;self.details=[];self.failed=False;self.profile_done=False;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch();self._install()
  libs={}
  for n in ['torch','torch_npu','transformers','torchvision','tokenizers']:
   f=inspect.getfile(importlib.import_module(n));libs[f]=sha(f)
  self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own original InternVision24/Phi3_32 eager BF16 one-forward MCScore, original F32 logits/log_softmax and SU RoPE preserved',runtime_dtype='BF16',head_size=32022,LM_layers=32,vision_layers=24,configured_max_new_tokens=2048,actual_generation_invocations=0,configured_outer_batch_size=1,actual_inner_batch_size=1,use_cache=False,original_algorithm='mc_score',original_raw_output=None,original_DropPath_rate=0.1,all_modules_eval=True,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),actual_batch_manifest=str(self.home/'completed_trace.json'),profiler_dir=str(self.home/'profiler'),runtime_library_source_SHA256=libs,historical_source_boundary=self.p['historical_source_boundary'],no_A100_tensor_capture_claimed=True,activation_retention='All samples full input/pixels/vision/text/injection/positions/causal masks/rotary values/last full-vocabulary head/logprobs/candidate scores; first3 all24vision+32LM+32QKV activation values; remaining layers complete shape/dtype/fullSHA/count metadata; original cacheFalse produces no returned KV')
 def _value(self,name,t,retain=True):
  assert t.device==self.device
  meta=record(t)
  if retain and name not in self.values:self.values[name]=arr(t)
  self.activations[name]=meta;return meta
 def _shared(self,kind,t):
  meta=record(t);key=(kind,meta['SHA256'],tuple(meta['shape']),meta['dtype'])
  if key not in self.shared:
   name=kind+'_'+str(len(self.shared));self.shared[key]=name;self._value(name,t)
  return self.shared[key]
 def _install(self):
  lm=self.model.language_model
  def top_pre(m,args,kw):
   assert set(kw)=={'input_ids','pixel_values','image_flags'} and kw['input_ids'].shape==(1,self.N)
   assert torch.equal(kw['image_flags'],torch.ones((len(kw['pixel_values']),1),dtype=torch.long,device=self.device))
   self._value('input_ids',kw['input_ids']);self._value('pixel_values',kw['pixel_values']);self._value('image_flags',kw['image_flags'])
  self.handles.append(self.model.register_forward_pre_hook(top_pre,with_kwargs=True))
  def embeddings(m,args,out):self._value('text_embeddings',out)
  self.handles.append(lm.get_input_embeddings().register_forward_hook(embeddings))
  def projected(m,args,out):self._value('vision_features',out)
  self.handles.append(self.model.mlp1.register_forward_hook(projected))
  def lm_pre(m,args,kw):
   assert kw.get('use_cache') is None and kw.get('past_key_values') is None and kw.get('attention_mask') is None and kw.get('position_ids') is None
   self._value('merged_embeddings',kw['inputs_embeds'])
  def lm_post(m,args,kw,out):
   assert out.past_key_values is None and out.logits.shape==(1,self.N,32022)
   row=out.logits[0,-1];assert row.dtype==torch.float32 and bool(torch.isfinite(row).all());self._value('full_head_logits',row);self.forward_count+=1
  self.handles.extend([lm.register_forward_pre_hook(lm_pre,with_kwargs=True),lm.register_forward_hook(lm_post,with_kwargs=True)])
  def head_input(m,args):self._value('head_input',args[0][0,-1])
  self.handles.append(lm.lm_head.register_forward_pre_hook(head_input))
  for j,layer in enumerate(lm.model.layers):
   def attention(m,args,kw,j=j):
    assert kw.get('past_key_value') is None and kw.get('use_cache') is False
    pos=kw['position_ids'];self._value('position_ids',pos)
    mask=kw.get('attention_mask');name=self._shared('causal_mask',mask) if mask is not None else None
    self.layer_frames.append(dict(layer=j,positions=pos.detach().cpu().tolist(),mask=name,past_key_value=None,use_cache=False))
   self.handles.append(layer.self_attn.register_forward_pre_hook(attention,with_kwargs=True))
   def rotary(m,args,out,j=j):
    assert len(out)==2;self.rotary_frames.append(dict(layer=j,cos=self._shared('rope_cos',out[0]),sin=self._shared('rope_sin',out[1])))
   self.handles.append(layer.self_attn.rotary_emb.register_forward_hook(rotary))
   def qkv(m,args,out,j=j):
    name='QKV_layer_'+str(j);self.counts[name]=self.counts.get(name,0)+1;self._value(name,out,self.index<3)
   self.handles.append(layer.self_attn.qkv_proj.register_forward_hook(qkv))
   def stage(m,args,out,j=j):
    name='LM_layer_'+str(j);self.counts[name]=self.counts.get(name,0)+1;self._value(name,out[0],self.index<3)
   self.handles.append(layer.register_forward_hook(stage))
  for j,layer in enumerate(self.model.vision_model.encoder.layers):
   def vision(m,args,out,j=j):
    t=out[0] if isinstance(out,(tuple,list)) else out;name='vision_layer_'+str(j);self.counts[name]=self.counts.get(name,0)+1;self._value(name,t,self.index<3)
   self.handles.append(layer.register_forward_hook(vision))
 def score_choice(self,prompt,options,media_path=None,image=None):
  i=self.index;self.values={};self.activations={};self.counts={};self.shared={};self.layer_frames=[];self.rotary_frames=[];self.forward_count=0;profile=None;ref=self.front[i]
  try:
   assert not self.failed,'Prior scoring failure; no further NN executed'
   assert image is None and prompt==ref['original_model_prompt'] and options==ref['options'] and str(Path(media_path).resolve())==ref['media_path'] and sha(media_path)==ref['image_SHA256']
   reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes'];self.N=len(ref['input_ids'])
   if not self.profile_done:
    profile=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));profile.start()
   # Exact original method: original preprocessing/query, one real model forward,
   # full-head F32 log_softmax, original three first-token candidates, no generation.
   with torch.inference_mode():scores=OSAtlasMCScore.score_choice(self,prompt,options,media_path)
   if profile is not None:profile.stop();profile=None;self.profile_done=True
   assert self.forward_count==1 and self.values['input_ids'].tolist()==[ref['input_ids']]
   assert hashlib.sha256(self.values['pixel_values'].tobytes()).hexdigest()==ref['pixels_BF16']['SHA256']
   # Additional observation only: reproduce the original unmodified F32 log_softmax
   # on the observed full NPU head to retain every vocabulary value.
   with torch.inference_mode():probs=torch.log_softmax(torch.from_numpy(self.values['full_head_logits'].copy()).to(self.device),dim=-1)
   self._value('full_log_probs',probs)
   reproduced={letter:max(float(probs[v['scored_id']].item()) for v in vs) for letter,vs in ref['option_variants'].items()};assert scores==reproduced
   size=sum(v.nbytes for v in self.values.values());assert shutil.disk_usage(self.home).free>reserve+size
   archive=self.home/f'sample_{i:05d}.npz';save(archive,self.values)
   e=dict(index=i,actual_model_prompt=prompt,media_path=str(media_path),reference=ref['reference'],input_ids=ref['input_ids'],generated_tokens=[],raw_output=None,scores=scores,options=options,option_variants=ref['option_variants'],error=None,actual_native_forwards=1,actual_generation_invocations=0,use_cache=False,returned_past_key_values=None,layer_frames=self.layer_frames,rotary_frames=self.rotary_frames,activations=self.activations,activation_counts=self.counts,first3_full_layer_value_retention=i<3,archive=str(archive),archive_bytes=archive.stat().st_size,archive_SHA256=sha(archive),original_budget_hit=False)
   f=self.home/f'sample_{i:05d}.json';put(f,e);detail=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),actual_native_forwards=1,actual_generation_invocations=0,raw_output=None,error=None,generated_tokens=[],input_ids=ref['input_ids'],scores=scores)
   self.details.append(detail)
   with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.index+=1;self.values={};LOG.info('OS-Atlas own original MCScore NPU %d/200 forward=1 archive=%d',self.index,archive.stat().st_size);return scores
  except BaseException:
   self.failed=True
   if profile is not None:profile.stop()
   failure=dict(index=i,actual_model_prompt=prompt,reference=ref['reference'],raw_output=None,generated_tokens=[],actual_native_forwards=self.forward_count,error=traceback.format_exc(),layer_frames=self.layer_frames,rotary_frames=self.rotary_frames)
   if self.values:
    f=self.home/f'sample_{i:05d}_failed_partial.npz';save(f,self.values);failure.update(partial_archive=str(f),partial_archive_SHA256=sha(f))
   f=self.home/f'sample_{i:05d}_failed.json';put(f,failure);detail=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),actual_native_forwards=self.forward_count,actual_generation_invocations=0,raw_output=None,error=failure['error'],generated_tokens=[])
   self.details.append(detail)
   with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.index+=1;raise
 def get_execution_lineage(self):return self.lineage
 def get_sample_execution(self,prompt,media_path):
  if self.report_index==0:
   files={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()};put(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,actual_outer_batches=[dict(indices=[i],status='completed' if not x['error'] else 'failed') for i,x in enumerate(self.details)],first_real_vision_prefill_profiler_SHA256=files,sample_trace_SHA256=sha(self.trace),algorithm='mc_score',actual_generation_invocations=0))
  d=self.details[self.report_index];self.report_index+=1;return d
 def unload(self):
  for h in self.handles:h.remove()
  self.model=self._model=None
  import torch.distributed as dist
  if self._owns_process_group and dist.is_initialized():dist.destroy_process_group()
  torch.npu.empty_cache()

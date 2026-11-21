from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
"""Own original Ovis-U1 understanding on NPU, complete native head/KV/vision execution values."""
import sys
sys.dont_write_bytecode=True
import os,json,hashlib,zipfile,logging,shutil,inspect,importlib,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import torch,torch_npu,numpy as np
from safetensors import safe_open
from transformers import GenerationConfig,LogitsProcessor
from universal_eval.model.interface import IModel
MID='AIDC-AI/Ovis-U1-3B';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def arr(t):
 t=t.detach().contiguous().cpu();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def rec(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
def put(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def save(p,values):
 write_full_standard_npz(p,values)
class _ObserveOriginalScores(LogitsProcessor):
 def __init__(self,owner):self.owner=owner
 def __call__(self,input_ids,scores):
  o=self.owner;assert scores.device==o.device and scores.shape==(1,151936) and scores.dtype==torch.float32
  # Lossless alias: all actual F32 score bits must equal the archived BF16 head cast, every row.
  original_head=torch.from_numpy(o.raw_heads[-1].copy()).view(torch.bfloat16).float();assert torch.equal(scores[0].cpu(),original_head)
  digest=rec(scores[0]);o.post_scores.append(digest);o.frames[-1].update(post_processor_dtype=str(scores.dtype),post_processor_F32_SHA256=digest['SHA256'],scores_losslessly_reconstructible_by_BF16_head_F32_cast=True,selected_argmax=int(scores[0].argmax()),HF_generated_prefix=input_ids[0].cpu().tolist());return scores
class OvisU1OwnUnderstandingNative(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):
  assert protocol['model_id']==MID and npu_devices==[4];self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Own original Ovis-U1 CANN vision + native Qwen3 SDPA BF16'
 def download(self,*a,**k):raise RuntimeError('Use own original local checkpoint')
 def load(self,model_path,**kw):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=js(self.protocol_file);self.W=Path(model_path);assert str(self.W)==self.p['weight_path'] and self.p['model_id']==MID
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,v in self.p['own_weight_files'].items():assert (self.W/n).stat().st_size==v['bytes'] and sha(self.W/n)==v['SHA256']
  from universal_eval.model.ovisu1_own_understanding_20261116.configuration_ovis_u1 import OvisU1Config
  from universal_eval.model.ovisu1_own_understanding_20261116.understanding_v2_tokenizer_class_alias import OwnOvisUnderstandingV2
  from universal_eval.model.ovisu1_own_understanding_20261116.modeling_aimv2 import VisionRotaryEmbedding
  from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
  torch.set_num_threads(4);self.device=torch.device('npu:4');torch.npu.set_device(self.device)
  for n in ['matmul','conv']:getattr(torch.npu,n).allow_hf32=False
  cfg=js(self.W/'config.json');cfg.update(_name_or_path=str(self.W),multimodal_max_length=4096)
  with torch.device('meta'):self.model=OwnOvisUnderstandingV2(OvisU1Config(**cfg),self.W)
  assert self.model.llm.config._attn_implementation=='sdpa' and not bool(self.model.llm._keep_in_fp32_modules) and not bool(self.model.visual_tokenizer._keep_in_fp32_modules)
  self.model.llm.model.rotary_emb=Qwen3RotaryEmbedding(config=self.model.llm.config,device='cpu')
  self.model.visual_tokenizer.backbone.trunk.rotary_pos_emb=VisionRotaryEmbedding(64)
  expected=dict(self.model.named_parameters(remove_duplicate=False));records={};self.unused=[]
  for n in self.p['own_weight_files']:
   with safe_open(str(self.W/n),framework='pt',device='cpu') as sf:
    for key in sf.keys():
     source=sf.get_tensor(key);target=source.to(torch.bfloat16).to(self.device);r=dict(shape=list(source.shape),source_dtype=str(source.dtype),source_SHA256=rec(source)['SHA256'],original_requested_dtype='torch.bfloat16',original_cast_SHA256=rec(source.bfloat16())['SHA256'],actual_NPU_SHA256=rec(target)['SHA256'],device=str(target.device),executed_in_image_understanding=not key.startswith('visual_generator.'))
     assert r['original_cast_SHA256']==r['actual_NPU_SHA256']
     if key.startswith('visual_generator.'):self.unused.append(target)
     else:
      assert key in expected and tuple(source.shape)==tuple(expected[key].shape);parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False))
     records[key]=r
  assert len(records)==1060 and sum(v['executed_in_image_understanding'] for v in records.values())==487
  self.model.llm.lm_head.weight=self.model.llm.model.embed_tokens.weight;self.model=self.model.eval().to(self.device);assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters()) and all(not t.is_meta and t.device==self.device for t in self.model.buffers())
  self.tokenizer=self.model.text_tokenizer;self.front=[json.loads(s) for s in Path(self.p['own_frontend200']).read_text().splitlines()];self.index=self.report_index=0;self.details=[];self.outer=[];self.failed=False;self.profile_done=False;self.handles=[]
  self.root_generation=GenerationConfig.from_model_config(self.model.config);self.kw=dict(max_new_tokens=512,do_sample=False,top_p=None,top_k=None,temperature=None,repetition_penalty=None,eos_token_id=self.root_generation.eos_token_id,pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
  self.gen,*_=self.model.llm._prepare_generation_config(None,**self.kw);assert self.root_generation.eos_token_id is None and self.gen.eos_token_id is None and self.gen.repetition_penalty is None and self.gen.num_beams==1
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('OvisU1_own_original_BF16_512_v2_guard_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False);self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch();self.weights=self.home/'own_source_actual_NPU_and_original_inactive_generator.json';put(self.weights,dict(model_id=MID,weights=records,original_active_physical_parameters=487,original_inactive_Yak_parameters=573,own_lm_head_tied_to_own_embedding=True,analytic_buffers={n:rec(t) for n,t in self.model.named_buffers()}));self._install()
  libraries={}
  for n in ['torch','torch._C','torch_npu','torch_npu._C','transformers','transformers.models.qwen3.modeling_qwen3','transformers.cache_utils','transformers.generation.utils','transformers.generation.logits_process','transformers.masking_utils','transformers.image_processing_backends','transformers.models.clip.image_processing_clip','transformers.models.qwen2.tokenization_qwen2']:
   p=inspect.getfile(importlib.import_module(n));libraries[p]=sha(p)
  self.lineage=dict(hardware='Ascend910B',devices=[4],runtime=self.inference_framework,runtime_dtype='BF16',head_size=151936,LM_layers=28,vision_layers=24,native_KV_heads=8,native_KV_head_dim=128,configured_outer_batch_size=4,actual_inner_batch_size=1,max_new_tokens=512,original_eos_override=None,original_generation_kwargs=self.kw,actual_prepared_generation_config=self.gen.to_dict(),lossless_post_score_storage='Every actual full F32 score row proven exactly equal to the archived BF16 head cast; independent full-row F32 SHA recorded for bit-exact reconstruction',original_multimodal_max_length=4096,original_enable_thinking_default=False,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),actual_batch_manifest=str(self.home/'completed_trace.json'),profiler_dir=str(self.home/'profiler'),runtime_library_source_SHA256=libraries,historical_source_boundary=self.p['historical_source_boundary'],A100_historical_tensor_capture_claimed=False)
 def _save(self,n,t):self.values[n]=arr(t);return rec(t)
 def _install(self):
  def pre(m,args,kw):
   self.frame=dict(step=len(self.frames),positions=kw['position_ids'].cpu().tolist(),layers=[])
  def post(m,args,kw,out):
   logits=out.logits[0,-1];assert logits.numel()==151936 and logits.dtype==torch.bfloat16 and bool(torch.isfinite(logits).all());self.raw_heads.append(arr(logits));N=self.N+len(self.frames);cache=out.past_key_values;groups=[];current=[]
   for j in range(28):
    layer=cache.layers[j];k,v=layer.keys,layer.values;assert k.shape==v.shape==(1,8,N,128) and k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
    if self.previous:assert torch.equal(k[:,:,:-1],self.previous[j][0]) and torch.equal(v[:,:,:-1],self.previous[j][1]),'Native KV prefix changed'
    current.append((k,v));groups.append(torch.stack([k[0] if not self.frames else k[0,:,-1:],v[0] if not self.frames else v[0,:,-1:]]))
   block=torch.stack(groups)
   if not self.frames:self.values['native_cache_prefill']=arr(block)
   else:self.deltas.append(arr(block))
   self.previous=current;self.frame.update(cache_length=N,all28_KV_prefixes_equal=True,initial_or_appended_KV=rec(block));self.frames.append(self.frame)
   if self.profile is not None:self.profile.stop();self.profile=None;self.profile_done=True
  self.handles.extend([self.model.llm.register_forward_pre_hook(pre,with_kwargs=True),self.model.llm.register_forward_hook(post,with_kwargs=True)])
  self.handles.append(self.model.llm.lm_head.register_forward_pre_hook(lambda m,args:self.head_inputs.append(arr(args[0][0,-1]))))
  for j,layer in enumerate(self.model.llm.model.layers):
   def att(m,args,kw,j=j):
    mask=kw.get('attention_mask');name=None
    if mask is not None:
     key=(mask.data_ptr(),tuple(mask.shape),mask.dtype)
     if key not in self.masks:self.masks[key]='attention_mask_'+str(len(self.masks));self._save(self.masks[key],mask)
     name=self.masks[key]
    cos,sin=kw['position_embeddings']
    if j==0:
     self._save('position_cos_'+str(len(self.frames)),cos);self._save('position_sin_'+str(len(self.frames)),sin);self.frame_cos_sin=(cos,sin)
    else:assert torch.equal(cos,self.frame_cos_sin[0]) and torch.equal(sin,self.frame_cos_sin[1])
    self.frame['layers'].append(dict(layer=j,mask=name,implicit_native_SDPA_causal=mask is None,position_embedding_shapes=[list(x.shape) for x in kw['position_embeddings']],all28_position_values_equal=True))
   self.handles.append(layer.self_attn.register_forward_pre_hook(att,with_kwargs=True))
   def hidden(m,args,out,j=j):
    name='LM_layer_'+str(j);self.counts[name]=self.counts.get(name,0)+1
    if not self.frames and self.index<3:self._save(name,out[0] if isinstance(out,(tuple,list)) else out)
   self.handles.append(layer.register_forward_hook(hidden))
  for j,layer in enumerate(self.model.visual_tokenizer.backbone.trunk.blocks):
   def visual(m,args,out,j=j):
    name='vision_layer_'+str(j);self.counts[name]=self.counts.get(name,0)+1
    if self.index<3:self._save(name,out)
   self.handles.append(layer.register_forward_hook(visual))
  def vision_head(m,a,o):self._save('vision_full_head_logits',o)
  def vision_input(m,a):self._save('vision_head_inputs',a[0])
  def vision_soft(m,a,o):self._save('vision_soft_tokens',o)
  self.handles.append(self.model.visual_tokenizer.head.register_forward_hook(vision_head))
  self.handles.append(self.model.visual_tokenizer.head.register_forward_pre_hook(vision_input))
  self.handles.append(self.model.visual_tokenizer.register_forward_hook(vision_soft))
  def vte(m,args,out):
   if args[0].is_floating_point():self._save('visual_embeddings',out)
   else:self._save('visual_indicator_embeddings',out)
  self.handles.append(self.model.vte.register_forward_hook(vte))
 def predict(self,prompt,media_path=None):
  from PIL import Image
  if self.failed:raise RuntimeError('Prior own batch failed; no additional NPU inference')
  i=self.index;r=self.front[i];assert prompt==r['original_model_prompt'] and str(Path(media_path).resolve())==r['media_path'] and sha(media_path)==r['image_SHA256']
  formatted,ids,pixels,grid=self.model.preprocess_inputs('<image>\n'+prompt,[Image.open(media_path).convert('RGB')]);mask=ids.ne(self.tokenizer.pad_token_id);assert ids.tolist()==r['input_ids'] and mask.tolist()==r['attention_mask'] and grid.tolist()==r['grid_thws'] and rec(pixels)==r['pixels_F32'] and rec(pixels.bfloat16())==r['pixels_BF16']
  reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes'];self.values={};self.frames=[];self.raw_heads=[];self.post_scores=[];self.head_inputs=[];self.deltas=[];self.previous=[];self.counts={};self.masks={};self.profile=None
  try:
   with torch.inference_mode():
    ids=ids[None].to(self.device);mask=mask[None].to(self.device);pixels=pixels.to(self.device,dtype=torch.bfloat16);grid=grid.to(self.device)
    for n,t in [('input_ids',ids),('input_attention_mask',mask),('pixels',pixels),('grid_thws',grid)]:self._save(n,t)
    if not self.profile_done:self.profile=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.profile.start()
    _,merged,labels,merged_mask,image_labels=self.model.merge_multimodal(ids,mask,None,pixels,grid,left_padding=True);self.N=merged.shape[1]
    for n,t in [('merged_embeddings',merged),('merged_attention_mask',merged_mask),('merged_image_labels',image_labels)]:self._save(n,t)
    generated=self.model.llm.generate(inputs=None,inputs_embeds=merged.detach(),attention_mask=merged_mask,**self.kw,logits_processor=[_ObserveOriginalScores(self)])
    if self.profile is not None:self.profile.stop();self.profile=None;self.profile_done=True
   tokens=generated[0].cpu().tolist();assert len(tokens)==len(self.frames)==len(self.post_scores)==512 and [x['selected_argmax'] for x in self.frames]==tokens
   self.values.update(all_raw_head_logits=np.stack(self.raw_heads),all_head_inputs=np.stack(self.head_inputs),generated_tokens=np.asarray(tokens,dtype=np.int64),native_cache_appends=np.stack(self.deltas))
   final={str(j):dict(K=rec(k),V=rec(v)) for j,(k,v) in enumerate(self.previous)};size=sum(v.nbytes for v in self.values.values());assert shutil.disk_usage(self.home).free>reserve+size;archive=self.home/f'sample_{i:05d}.npz';save(archive,self.values);raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
   e=dict(index=i,actual_model_prompt=prompt,reference=r['reference'],media_path=str(media_path),raw_output=raw,error=None,input_ids=r['input_ids'],generated_tokens=tokens,merged_length=self.N,frames=self.frames,actual_native_forwards=len(self.frames),activation_counts=self.counts,final_native_cache_SHA256=final,archive=str(archive),archive_bytes=archive.stat().st_size,archive_SHA256=sha(archive),original_budget_hit=True,original_stop_reason='length_512_original_eos_override_None')
   f=self.home/f'sample_{i:05d}.json';put(f,e);d=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),raw_output=raw,error=None,generated_tokens=tokens,actual_native_forwards=len(self.frames));self.details.append(d)
   with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(d,ensure_ascii=False)+'\n')
   self.index+=1;self.values={};self.previous=[];LOG.info('Own Ovis-U1 NPU %d/200 forwards=%d original512 kept',self.index,len(self.frames));return raw
  except BaseException:
   self.failed=True
   if self.profile is not None:self.profile.stop();self.profile=None
   failure=dict(index=i,actual_model_prompt=prompt,reference=r['reference'],error=traceback.format_exc(),committed_forward_count=len(self.frames),frames=self.frames)
   if self.values:
    a=self.home/f'sample_{i:05d}_failed_partial.npz';save(a,self.values);failure.update(partial_archive=str(a),partial_archive_SHA256=sha(a))
   f=self.home/f'sample_{i:05d}_failed.json';put(f,failure);d=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),raw_output=None,error=failure['error'],actual_native_forwards=len(self.frames),generated_tokens=[]);self.details.append(d)
   with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(d,ensure_ascii=False)+'\n')
   self.index+=1;raise
 def batch_predict(self,batch):
  group=dict(indices=list(range(self.index,self.index+len(batch))),status='started');self.outer.append(group)
  try:out=[self.predict(p,m) for p,m in batch];group['status']='completed';put(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,actual_outer_batches=self.outer,sample_trace_SHA256=sha(self.trace),first_real_vision_prefill_profiler_SHA256={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()}));return out
  except BaseException:
   group['status']='failed'
   while self.index<=group['indices'][-1]:
    i=self.index;f=self.home/f'sample_{i:05d}_not_executed.json';error='Not executed after prior own original batch failure';put(f,dict(index=i,actual_model_prompt=self.front[i]['original_model_prompt'],reference=self.front[i]['reference'],error=error,actual_native_forwards=0,not_executed=True));d=dict(index=i,actual_execution_file=str(f),actual_execution_SHA256=sha(f),raw_output=None,error=error,actual_native_forwards=0,generated_tokens=[]);self.details.append(d)
    with self.trace.open('a',encoding='utf-8') as h:h.write(json.dumps(d,ensure_ascii=False)+'\n')
    self.index+=1
   raise
 def get_sample_execution(self,prompt,media_path):d=self.details[self.report_index];self.report_index+=1;return d
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;self.unused=[];torch.npu.empty_cache()

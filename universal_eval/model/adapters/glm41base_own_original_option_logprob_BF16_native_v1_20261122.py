"""Own GLM4.1 Base original option-text mean-logprob protocol on Ascend910B."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import collections,hashlib,inspect,json,logging,os,re,shutil,traceback
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from transformers import Glm4vConfig,Glm4vForConditionalGeneration,GenerationConfig
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.glm41base_own_original_option_inputs_v1_20261122 import OwnGLM41BaseInputs,tensor_sha
from universal_eval.utils.complete_module_values_tree_20261122 import CompleteValueTree,raw_array
from universal_eval.utils.complete_standard_LZMA_NPZ_parallel_values_20261122 import write_full_standard_npz
MID='zai-org/GLM-4.1V-9B-Base';R=Path('/root/projects/comperror');LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def stamp():return datetime.now(timezone.utc).isoformat()
def source_inventory():
 own={};external={}
 for m in list(sys.modules.values()):
  n=getattr(m,'__file__',None)
  if not n:continue
  p=Path(n).resolve()
  if not p.is_file():continue
  (own if p.is_relative_to(R) else external)[str(p)]=sha(p)
 for line in Path('/proc/self/maps').read_text().splitlines():
  fields=line.split(None,5)
  if len(fields)<6 or 'x' not in fields[1] or not fields[5].startswith('/'):continue
  p=Path(fields[5])
  if p.is_file() and str(p) not in own and str(p) not in external:(own if p.is_relative_to(R) else external)[str(p)]=sha(p)
 return own,external

class OwnGLM41BaseOriginalProbabilityBF16(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
  assert protocol=={'model_id':MID} and npu_devices==[4];self.protocol_file=Path(protocol_file);assert sha(self.protocol_file)==protocol_sha256;self.protocol_sha=protocol_sha256;self.p=json.loads(self.protocol_file.read_bytes());self.model=None;self.active=False
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**k):raise RuntimeError('Only own existing weights and assets authorized')
 def load(self,model_path,**kwargs):
  p=self.p;self.W=Path(model_path);assert p['model_id']==MID and str(self.W)==p['weight_path'] and p['original_option_text_mean_logprob_protocol_preserved']
  for n,h in p['implementation_source_SHA256'].items():assert sha(R/n)==h,n
  for n,h in p['static_evidence_SHA256'].items():assert sha(n)==h,n
  for n,h in p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  self.inputs=OwnGLM41BaseInputs(self.W);self.tokenizer=self.inputs.tokenizer;self.stops=[151329,151336,151338,151348];self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  cfg=Glm4vConfig(**json.loads((self.W/'config.json').read_bytes()));cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';assert cfg.text_config.num_hidden_layers==40 and cfg.text_config.num_key_value_heads==2 and cfg.text_config.vocab_size==151552
  previous_dtype=torch.get_default_dtype()
  try:torch.set_default_dtype(torch.bfloat16);self.model=Glm4vForConditionalGeneration(cfg)
  finally:torch.set_default_dtype(previous_dtype)
  expected=json.loads(Path(p['own_source_manifest']).read_bytes());params=dict(self.model.named_parameters(remove_duplicate=False));assert len(expected)==len(params)==704 and set(expected)==set(params);weights={}
  for name,rec in p['own_weight_files'].items():
   f=self.W/name;assert f.stat().st_size==rec['bytes'] and sha(f)==rec['SHA256']
   with safe_open(str(f),framework='pt',device='cpu') as sf:
    assert set(sf.keys())=={n for n,r in expected.items() if r['source_file']==name}
    for key in sf.keys():
     t=sf.get_tensor(key);r=expected[key];assert t.dtype==torch.bfloat16 and torch.isfinite(t).all() and list(t.shape)==r['shape'] and tensor_sha(t)==r['source_SHA256']==r['cast_SHA256'];weights[key]=t
  self.model.load_state_dict(weights,strict=True,assign=True);del weights
  before={n:dict(shape=list(t.shape),dtype=str(t.dtype),CPU_analytic_SHA256=tensor_sha(t)) for n,t in self.model.named_buffers(remove_duplicate=False)}
  self.model.to(self.device);self.model.eval();assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters())
  generation=GenerationConfig.from_pretrained(self.W,local_files_only=True);assert generation.eos_token_id==self.stops
  buffers={n:dict(v,actual_NPU_SHA256=tensor_sha(dict(self.model.named_buffers(remove_duplicate=False))[n])) for n,v in before.items()};assert all(v['CPU_analytic_SHA256']==v['actual_NPU_SHA256'] for v in buffers.values())
  now=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=R/'workspace/diagnostics/model_runs'/('GLM41Base_own_original_option_logprob_BF16_'+now+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  actual={n:dict(r,actual_NPU_SHA256=tensor_sha(dict(self.model.named_parameters(remove_duplicate=False))[n]),actual_NPU_device=str(self.device)) for n,r in expected.items()};assert all(r['actual_NPU_SHA256']==r['cast_SHA256'] for r in actual.values());self.weight_proof=self.home/'all_own_source_actual_NPU.json';write(self.weight_proof,dict(weights=actual,buffers=buffers,own_tied_head=False,all704_own_sources_loaded=True))
  self.front=json.loads(Path(p['own_frontend200']).read_bytes());assert len(self.front)==200;self.index=0;self.report_index=0;self.failed=False;self.details=[];self.batches=[];self.handles=[];self.calls=[];self.stack=collections.defaultdict(list);self.profile_done=False;self.profiler=None;self.profile=self.home/'first_vision_prefill_npu_trace.json';self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.progress=self.home/'committed_option_forwards.jsonl';self.progress.touch(exist_ok=False);self._install()
  own,external=source_inventory();own.update({str(R/n):h for n,h in self.p['implementation_source_SHA256'].items()});snap=self.home/'source_snapshot';snap.mkdir()
  for n,h in own.items():
   path=Path(n);dest=snap/path.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());assert sha(dest)==h
  cli=Path(sys.argv[sys.argv.index('--config')+1]).resolve();assert cli.is_relative_to(R/'configs');write(self.home/'actual_runtime_source_SHA256.json',dict(own=own,external=external,actual_YAML=str(cli),actual_YAML_SHA256=sha(cli)))
  for path in [cli,self.protocol_file]:dest=snap/path.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
  self.lineage=dict(hardware='Ascend910B',devices=[4],runtime_dtype='BF16',model_id=MID,original_probability_not_generation=True,original_option_text_mean_logprob=True,original_saved_fallback_max_tokens=2048,actual_option_max_generated_tokens=1,original_prompt_logprobs=0,actual_outer_batch_size=10,actual_inner_batch_size=1,original_original_request_order_preserved=True,original_use_cache=True,native_fresh_request_40_physical_KV=True,head_size=151552,original_generation_config=vars(generation),original_no_new_thinking_override=True,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),committed_option_progress=str(self.progress),final_trace_manifest=str(self.home/'completed_trace.json'),evidence_root=str(self.home),actual_runtime_source_inventory=str(self.home/'actual_runtime_source_SHA256.json'),actual_runtime_source_inventory_SHA256=sha(self.home/'actual_runtime_source_SHA256.json'),first_real_vision_and_prefill_profiler=str(self.profile),profiler_scope='First genuine full vision and text prefill only; not all option forwards',HF32_enabled={'matmul':False,'conv':False},complete_module_capture_names=self.capture_names,historical_A100_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'])
 def _install(self):
  self.capture_names=[]
  for name,module in self.model.named_modules():
   if not name:continue
   selected=(not list(module.children()) or re.fullmatch(r'model\.visual(?:\.blocks\.\d+(?:\.attn|\.mlp)?)?',name) or re.fullmatch(r'model\.language_model\.layers\.\d+(?:\.self_attn|\.mlp)?',name) or name in ['model.visual.patch_embed','model.visual.merger'])
   if not selected:continue
   self.capture_names.append(name)
   def pre(m,args,kw,name=name):
    if not self.active:return
    layer=getattr(m,'layer_idx',None)
    if layer is None and hasattr(m,'self_attn'):layer=getattr(m.self_attn,'layer_idx',None)
    call=dict(sequence=len(self.calls),module=name,module_class=type(m).__module__+'.'+type(m).__name__,args=self.values.encode(args,layer),kwargs=self.values.encode(kw,layer));self.calls.append(call);self.stack[name].append(call)
   def post(m,args,kw,out,name=name):
    if not self.active:return
    call=self.stack[name].pop();call['output']=self.values.encode(out,cache_output=True)
   self.handles.extend([module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(post,with_kwargs=True)])
  def fused(m,args,kw):
   if self.active:self.current_fused=self.values.tensor(kw['inputs_embeds'])
  self.handles.append(self.model.model.language_model.register_forward_pre_hook(fused,with_kwargs=True))
 def predict(self,prompt,media_path=None):
  assert not self.failed,'Original failure retained; no silent replay';i=self.index;ref=self.front[i];assert prompt==ref['actual_model_prompt'] and sha(media_path)==ref['image_SHA256'];self.values=CompleteValueTree(self.device);self.options=[]
  try:
   reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
   for oi,opt in enumerate(ref['options']):
    v=self.inputs.prepare(ref,opt,media_path);data={n:t.to(self.device) for n,t in v.items() if torch.is_tensor(t)};self.calls=[];self.stack=collections.defaultdict(list);self.active=True
    input_refs=self.values.encode(data);start=datetime.now(timezone.utc).isoformat()
    if not self.profile_done:self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.profiler.start()
    with torch.inference_mode():out=self.model(**data,use_cache=True,logits_to_keep=0,return_dict=True)
    self.active=False
    if self.profiler is not None:self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None;self.profile_done=True
    assert all(not v for v in self.stack.values()) and out.logits.shape==(1,len(opt['own_complete_input_ids']),151552) and out.logits.dtype==torch.bfloat16 and torch.isfinite(out.logits).all()
    full=self.values.tensor(out.logits);head_input=next(c['args']['tuple'][0] for c in self.calls if c['module']=='lm_head');aux=int(out.logits[0,-1].argmax().item());ids=data['input_ids'][0];logprobs=out.logits[0,:-1].float().log_softmax(-1).gather(1,ids[1:,None]).squeeze(1);assert torch.isfinite(logprobs).all();plp_ref=self.values.tensor(logprobs);plp=[None]+logprobs.detach().cpu().tolist();k=opt['option_token_count'];vals=plp[-(k+2):-2];assert len(vals)==k and all(x is not None for x in vals);score=sum(vals)/len(vals)
    cache=out.past_key_values;assert len(cache.layers)==40
    for layer in cache.layers:assert layer.keys.shape==layer.values.shape==(1,2,len(ids),128) and layer.keys.dtype==layer.values.dtype==torch.bfloat16
    cache_refs=self.values.encode(cache,cache_output=True);rec=dict(index=oi,letter=opt['letter'],original_manual_prompt=opt['own_manual_prompt'],original_option_text=opt['original_option_text'],original_continuation=opt['original_continuation'],own_complete_input_ids=opt['own_complete_input_ids'],option_token_count=k,original_score_positions=opt['original_prompt_logprob_positions'],original_score_token_ids=opt['original_score_token_ids'],input_tensors=input_refs,native_fused_inputs=self.current_fused,complete_module_calls=self.calls,complete_full_head_logits=full,complete_head_inputs=head_input,complete_prompt_token_logprobs=plp_ref,original_prompt_logprob_first_is_None=True,native_all_prompt_token_logprobs=plp,original_selected_logprobs=vals,original_mean_logprob=score,original_auxiliary_generated_token_ids=[aux],complete_native_40KV=cache_refs,actual_native_forwards=1,forward_start_UTC=start,committed_UTC=stamp())
    self.options.append(rec)
    with self.progress.open('a',encoding='utf8') as f:f.write(json.dumps(dict(index=i,option_index=oi,letter=opt['letter'],actual_native_forward=1,UTC=rec['committed_UTC']),ensure_ascii=False)+'\n')
    LOG.info('Own GLM41 Base sample %d option %s genuine prefill committed, original mean logprob %.9f',i,opt['letter'],score);del out,data,logprobs
   best=max(self.options,key=lambda x:x['original_mean_logprob']);raw=best['letter'];size=sum(a.nbytes for a in self.values.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
   archive=self.home/('sample_%05d_complete_values.npz'%i);write_full_standard_npz(archive,self.values.arrays);record=self.home/('sample_%05d.json'%i);e=dict(index=i,actual_model_prompt=prompt,reference=ref['reference'],media_path=str(media_path),image_SHA256=ref['image_SHA256'],raw_output=raw,error=None,original_probability_not_generation=True,original_candidate_tie_order=[o['letter'] for o in self.options],actual_native_forwards=len(self.options),options=self.options,archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,unique_complete_arrays=len(self.values.arrays),complete_tensor_occurrences=self.values.occurrences,lossless_uncompressed_bytes=size);write(record,e)
   detail=dict(index=i,actual_model_prompt=prompt,actual_execution_file=str(record),actual_execution_SHA256=sha(record),actual_native_forwards=len(self.options),generated_tokens=[],original_auxiliary_generated_token_ids_per_option=[dict(letter=o['letter'],ids=o['original_auxiliary_generated_token_ids']) for o in self.options],own_actual_input_ids_per_option=[dict(letter=o['letter'],input_ids=o['own_complete_input_ids'],score_positions=o['original_score_positions']) for o in self.options],original_probability_scores=[dict(letter=o['letter'],mean_logprob=o['original_mean_logprob']) for o in self.options],raw_output=raw,error=None)
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.details.append(detail);self.index+=1;self.values=None;self.options=[];return raw
  except BaseException:
   self.active=False;self.failed=True
   if self.profiler is not None:self.profiler.stop();self.profiler=None
   failure=dict(index=i,committed_original_option_forwards=len(self.options),error=traceback.format_exc())
   if self.values.arrays and shutil.disk_usage(self.home).free>self.p['hard_reserve_bytes']+sum(a.nbytes for a in self.values.arrays.values()):
    file=self.home/('sample_%05d_failed_complete_values.npz'%i);write_full_standard_npz(file,self.values.arrays);failure.update(archive=str(file),archive_SHA256=sha(file))
   write(self.home/('sample_%05d_failed.json'%i),failure);raise
 def batch_predict(self,batch):
  assert 1<=len(batch)<=10;row=dict(indices=list(range(self.index,self.index+len(batch))),status='running');self.batches.append(row)
  try:out=[self.predict(p,m) for p,m in batch];row['status']='completed';return out
  except BaseException:row['status']='failed';raise
 def get_sample_execution(self,prompt,media_path):
  i=self.report_index;self.report_index+=1
  if i==0:
   own,external=source_inventory();initial=json.loads((self.home/'actual_runtime_source_SHA256.json').read_bytes());own.update({str(R/n):h for n,h in self.p['implementation_source_SHA256'].items()})
   for n,h in {**initial['own'],**initial['external']}.items():assert sha(n)==h
   for n,h in own.items():
    path=Path(n);dest=self.home/'source_snapshot'/path.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==h
    else:dest.write_bytes(path.read_bytes())
   final_sources=self.home/'all_actual_final_runtime_source_SHA256.json';write(final_sources,dict(own=own,external=external))
   write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),actual_outer_batches=self.batches,all_actual_final_runtime_source_inventory=str(final_sources),all_actual_final_runtime_source_inventory_SHA256=sha(final_sources),first_vision_prefill_profiler_SHA256={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()},first_vision_prefill_trace_SHA256=sha(self.profile) if self.profile.is_file() else None))
  if i<len(self.details):return self.details[i]
  return dict(index=i,actual_model_prompt=prompt,actual_native_forwards=0,generated_tokens=[],raw_output=None,error='Original actual failure evidence preserved before this sample completed')
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  for h in getattr(self,'handles',[]):h.remove()
  self.model=None;torch.npu.empty_cache()

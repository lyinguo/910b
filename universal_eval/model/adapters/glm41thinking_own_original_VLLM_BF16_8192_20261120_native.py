"""Own GLM-4.1V-9B-Thinking complete BF16 NPU execution, head/cache evidence and full reports."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,inspect,subprocess,shutil,os,traceback,re,zipfile,logging
import torch,torch_npu,numpy as np
from safetensors import safe_open
from transformers import Glm4vConfig,Glm4vForConditionalGeneration,GenerationConfig
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend import OwnGLM41ThinkingInputs,tensor_sha,flatten_tensors
MID='zai-org/GLM-4.1V-9B-Thinking';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array(t):
 t=t.detach().cpu().contiguous();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def save_npz(p,values):
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
 write_full_standard_npz(p,values)
class GLM41ThinkingOwnNativeBF16(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
  assert protocol=={'model_id':MID} and npu_devices==[4];self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256;self.devices=npu_devices
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**kw):raise RuntimeError('Only own existing checkpoint authorized')
 def load(self,model_path,**kwargs):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=json.loads(self.protocol_file.read_bytes());self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h
  for n,v in self.p['own_weight_files'].items():assert sha(self.W/n)==v['SHA256']
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert info.returncode==0 and 'No running processes found in NPU 4' in info.stdout
  self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  self.inputs=OwnGLM41ThinkingInputs(self.W);self.tokenizer=self.inputs.tokenizer;self.stops=[151329,151336,151338,151348]
  cfg=Glm4vConfig(**json.loads((self.W/'config.json').read_bytes()));cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';assert cfg.text_config.num_hidden_layers==40 and cfg.text_config.num_key_value_heads==2 and cfg.text_config.vocab_size==151552
  dtype=torch.get_default_dtype()
  try:
   torch.set_default_dtype(torch.bfloat16);self.model=Glm4vForConditionalGeneration(cfg)
  finally:torch.set_default_dtype(dtype)
  expected=json.loads(Path(self.p['own_source_manifest']).read_bytes());unused=json.loads(Path(self.p['own_unused_MTP_source_manifest']).read_bytes());params=dict(self.model.named_parameters(remove_duplicate=False));assert set(params)==set(expected) and len(params)==704 and len(unused)==0
  weights={}
  for filename in sorted(self.p['own_weight_files']):
   with safe_open(str(self.W/filename),framework='pt',device='cpu') as f:
    assert set(f.keys())=={n for n,v in expected.items() if v['source_file']==filename}
    for key in f.keys():
     t=f.get_tensor(key)
     if key in unused:assert key.startswith('model.language_model.layers.16.') and tensor_sha(t)==unused[key]['SHA256'];continue
     r=expected[key];assert str(t.dtype)==r['source_dtype']=='torch.bfloat16' and tensor_sha(t)==r['source_SHA256']==r['cast_SHA256'] and list(t.shape)==r['shape'];weights[key]=t
  self.model.load_state_dict(weights,strict=True,assign=True);del weights
  before={n:dict(shape=list(t.shape),dtype=str(t.dtype),CPU_analytic_SHA256=tensor_sha(t)) for n,t in self.model.named_buffers(remove_duplicate=False)}
  self.model.to(self.device);self.model.eval();self.model.generation_config=GenerationConfig.from_pretrained(self.W,local_files_only=True);assert self.model.generation_config.eos_token_id==self.stops
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters())
  buffers={n:dict(before[n],actual_NPU_SHA256=tensor_sha(t)) for n,t in self.model.named_buffers(remove_duplicate=False)};assert all(v['CPU_analytic_SHA256']==v['actual_NPU_SHA256'] for v in buffers.values())
  now=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('GLM41Thinking_own_original_BF16_8192_'+now+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  actual={n:dict(v,actual_NPU_SHA256=tensor_sha(dict(self.model.named_parameters(remove_duplicate=False))[n]),actual_NPU_device=str(self.device)) for n,v in expected.items()};assert all(v['actual_NPU_SHA256']==v['cast_SHA256'] for v in actual.values())
  self.weight_proof=self.home/'all_own_source_actual_NPU.json';write(self.weight_proof,dict(weights=actual,buffers=buffers,own_tied_head=False,original_unused_MTP_source=unused,unused_MTP_executed=False))
  self.front=json.loads(Path(self.p['own_frontend200']).read_bytes());assert len(self.front)==200
  self.index=0;self.report_index=0;self.details=[];self.handles=[];self.active=False;self.failed=False;self.profile_done=False;self.profiler=None;self.batches=[];self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False);self.profile=self.home/'npu_trace.json';self._install()
  import transformers,transformers.masking_utils,transformers.cache_utils,transformers.models.glm4v.modeling_glm4v,transformers.models.glm46v.processing_glm46v,transformers.models.glm46v.image_processing_glm46v,transformers.integrations.sdpa_attention
  libs={inspect.getfile(m):sha(inspect.getfile(m)) for m in [torch,torch_npu,transformers,transformers.masking_utils,transformers.cache_utils,transformers.models.glm4v.modeling_glm4v,transformers.models.glm46v.processing_glm46v,transformers.models.glm46v.image_processing_glm46v,transformers.integrations.sdpa_attention]}
  libs.update(self.inputs.native_media_external_SHA256)
  self.lineage=dict(original_enable_thinking=None,original_generation_config=vars(self.model.generation_config),hardware='Ascend910B',devices=[4],runtime='Own GLM-4.1V-9B-Thinking own vision24/text40/GQA2; saved saved original VLLM greedy8192 native input and stopping contract, native HF SDPA execution, no original chat thinking override, all704 own sources active',runtime_dtype='BF16',max_new_tokens=8192,actual_outer_batch_size=10,actual_inner_batch_size=1,attention='Native HF SDPA and own three-dimensional RoPE',use_cache=True,head_size=151552,tokenizer_size=len(self.tokenizer),protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,evidence_root=str(self.home),weight_evidence=str(self.weight_proof),weight_evidence_SHA256=sha(self.weight_proof),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),first_real_vision_and_prefill_profiler=str(self.profile),runtime_library_source_SHA256=libs,historical_source_boundary=self.p['historical_source_boundary'])
 def predict(self,prompt,media_path=None):
  assert not self.failed,'Original failure retained; no silent replay';i=self.index;self.frames=[];self.active=False
  try:
   ref=self.front[i];assert prompt==ref['actual_model_prompt'] and sha(media_path)==ref['media_file_SHA256'];text,v=self.inputs.prepare(prompt,media_path);assert text==ref['query']
   for n,t in flatten_tensors(v).items():assert tensor_sha(t)==ref['tensor_metadata'][n]['SHA256']
   reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
   self.arrays={'input_ids':v['input_ids'].numpy(),'own_original_HF_mask':v['attention_mask'].numpy(),'own_pixels_F32':v['pixel_values'].numpy(),'own_pixels_BF16':array(v['pixel_values'].to(torch.bfloat16)),'original_input_image_grid_thw':v['image_grid_thw'].numpy()}
   self.logits=[];self.head_inputs=[];self.deltas=[];self.activations={};self.counts={};self.previous_pairs=[];self.active=True;data={k:t.to(self.device) if torch.is_tensor(t) else t for k,t in v.items()}
   if not self.profile_done:self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.profiler.start()
   with torch.inference_mode():generated=self.model.generate(**data,max_new_tokens=8192,do_sample=False)
   tokens=generated[0,v['input_ids'].shape[1]:].tolist();self.active=False;assert tokens==[f['argmax'] for f in self.frames]
   raw=self.inputs.processor.batch_decode([tokens],skip_special_tokens=True)[0].strip();self.arrays['all_original_full_head_logits']=np.stack(self.logits);self.arrays['all_head_inputs']=np.stack(self.head_inputs);self.arrays['native_cache_decode_delta']=np.concatenate(self.deltas,axis=3) if self.deltas else np.empty((40,2,2,0,128),np.uint16)
   final_cache=[[tensor_sha(t) for t in pair] for pair in self.previous_pairs];size=sum(t.nbytes for t in self.arrays.values());assert shutil.disk_usage(self.home).free>reserve+size
   ap=self.home/('sample_%05d.npz'%i);save_npz(ap,self.arrays);rp=self.home/('sample_%05d.json'%i);rec=dict(index=i,actual_model_prompt=prompt,query=text,media_path=str(media_path),media_SHA256=ref['media_file_SHA256'],input_ids=ref['input_ids'],generated_tokens=tokens,raw_output=raw,reference=ref['reference'],error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,original_budget_hit=len(tokens)==8192 and tokens[-1] not in self.stops,stop_ids=self.stops,compressed_length=self.compressed_length,final_cache_SHA256=final_cache,archive=str(ap),archive_SHA256=sha(ap),archive_bytes=ap.stat().st_size,lossless_uncompressed_bytes=size);write(rp,rec)
   detail=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(tokens),generated_tokens=tokens,raw_output=raw,error=None)
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.details.append(detail);self.index+=1;self.arrays={};self.logits=[];self.head_inputs=[];self.deltas=[];self.previous_pairs=[];LOG.info('Own GLM-4.1V-9B-Thinking sample %d, %d genuine forwards',i+1,len(tokens));return raw
  except BaseException:
   self.active=False;self.failed=True
   if self.profiler is not None:self.profiler.stop();self.profiler=None
   write(self.home/('sample_%05d_failed.json'%i),dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
 def batch_predict(self,batch):
  assert 1<=len(batch)<=10;rec=dict(indices=list(range(self.index,self.index+len(batch))),status='running');self.batches.append(rec)
  try:r=[self.predict(p,m) for p,m in batch];rec['status']='completed';return r
  except BaseException:rec['status']='failed';raise
 def get_sample_execution(self,prompt,media_path):
  i=self.report_index;self.report_index+=1
  if i==0:write(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),actual_outer_batches=self.batches,first_prefill_profiler_SHA256=sha(self.profile) if self.profile.is_file() else None,first_real_vision_prefill_profiler_SHA256={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()}))
  if i<len(self.details):return self.details[i]
  return dict(index=i,actual_model_prompt=prompt,actual_native_forwards=0,generated_tokens=[],raw_output=None,error='Actual run failed before this sample completed; original failure JSON/log retained')
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  for h in getattr(self,'handles',[]):h.remove()
  self.model=None;self.previous_pairs=[];torch.npu.empty_cache()
 def _record(self,name,t,retain=True):
  assert torch.is_tensor(t) and t.device==self.device;v=t.detach().cpu().contiguous();meta=dict(shape=list(v.shape),dtype=str(v.dtype),SHA256=tensor_sha(v))
  if retain:assert name not in self.arrays;self.arrays[name]=array(v)
  return meta
 def _install(self):
  lm=self.model.model.language_model
  def top_pre(m,args,kw):
   if self.active:self.frame=dict(step=len(self.frames),attention=[]);self.tensor_cache={}
  def lm_pre(m,args,kw):
   if self.active and not self.frames:self.activations['prefill_inputs_embeds']=self._record('prefill_inputs_embeds',kw['inputs_embeds']);self.compressed_length=kw['inputs_embeds'].shape[1]
  def rope_pre(m,args,kw):
   if self.active:
    ids=kw.get('position_ids',args[1] if len(args)>1 else None);assert torch.is_tensor(ids);self.frame['position_ids_values']=ids.detach().cpu().tolist();self.frame['position_ids']=self._record('f%d_native_positions'%(len(self.frames)+1),ids)
  def head_hook(m,args,out):
   if self.active:
    row=out[0,-1];assert row.shape==(151552,) and row.dtype==torch.bfloat16;self.logits.append(array(row));self.head_inputs.append(array(args[0][0,-1]));self.frame.update(head_SHA256=tensor_sha(row),head_shape=list(out.shape),head_dtype=str(row.dtype),argmax=int(row.argmax().item()))
  def top_post(m,args,kw,out):
   if not self.active:return
   n=len(self.frames);length=self.compressed_length+n;cache=out.past_key_values;assert len(cache.layers)==40;pieces=[];pairs=[]
   for i,l in enumerate(cache.layers):
    k,v=l.keys,l.values;assert k.shape==v.shape==(1,2,length,128) and k.dtype==v.dtype==torch.bfloat16 and k.device==v.device==self.device
    if n:
     pk,pv=self.previous_pairs[i];assert torch.equal(k[:,:,:-1],pk) and torch.equal(v[:,:,:-1],pv),(n,i);pieces.extend((k[0,:,-1:],v[0,:,-1:]))
    else:pieces.extend((k[0],v[0]))
    pairs.append((k,v))
   packed=torch.stack(pieces).reshape(40,2,2,-1,128)
   if n:self.deltas.append(array(packed))
   else:self.arrays['native_cache_prefill']=array(packed)
   self.previous_pairs=pairs;self.frame.update(cache_seq_len=length,unchanged_prefix_exact_checked_layers=40 if n else 0);self.frames.append(self.frame)
   if self.profiler is not None and n==0:self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None;self.profile_done=True
  self.handles.extend([self.model.register_forward_pre_hook(top_pre,with_kwargs=True),lm.register_forward_pre_hook(lm_pre,with_kwargs=True),lm.rotary_emb.register_forward_pre_hook(rope_pre,with_kwargs=True),self.model.lm_head.register_forward_hook(head_hook),self.model.register_forward_hook(top_post,with_kwargs=True)])
  for name,module in self.model.named_modules():
   if re.fullmatch(r'model\.(visual.blocks|language_model.layers)\.\d+',name):
    def stage(m,args,out,name=name):
     if not self.active:return
     self.counts[name]=self.counts.get(name,0)+1
     if not self.frames:
      t=out[0] if isinstance(out,(tuple,list)) else out;k='stage_'+re.sub('[^a-zA-Z0-9_]','_',name);self.activations[k]=self._record(k,t,retain=True)
    self.handles.append(module.register_forward_hook(stage))
  for i,l in enumerate(lm.layers):
   def attn_pre(m,args,kw,i=i):
    if not self.active:return
    obs=dict(layer=i,implicit_SDPA_causal_for_query_gt1=kw.get('attention_mask') is None)
    for key in ['attention_mask','position_ids','position_embeddings']:
     values=kw.get(key)
     if values is None:obs[key]=None;continue
     values=values if isinstance(values,(tuple,list)) else [values];refs=[]
     for j,t in enumerate(values):
      assert torch.is_tensor(t);ident=(id(t),t.data_ptr(),tuple(t.shape),str(t.dtype))
      if ident not in self.tensor_cache:
       k='f%d_attn_%s_%d_%d'%(len(self.frames)+1,key,i,j);self.tensor_cache[ident]=dict(array=k,**self._record(k,t))
      refs.append(self.tensor_cache[ident])
     obs[key]=refs
    self.frame['attention'].append(obs)
   self.handles.append(l.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
  original=self.model.model.get_image_features
  def observe(*args,**kw):
   out=original(*args,**kw)
   if self.active:self.activations['own_visual_features']=self._record('own_visual_features',torch.cat(out.pooler_output,dim=0))
   return out
  self.model.model.get_image_features=observe

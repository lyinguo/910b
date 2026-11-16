"""Own MiniCPM-V-4.6 BF16 native NPU graph, saved MCQ protocol and full lossless audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,hashlib,inspect,traceback,shutil,zipfile,types,logging,subprocess
import numpy as np
import torch,torch_npu
from safetensors import safe_open
from transformers import AutoConfig,GenerationConfig,MiniCPMV4_6ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.minicpmv46_own_frontend_20261116 import OwnMiniCPMV46Inputs,tsha,flatten
from universal_eval.model.reference_sources.minicpmv46_original_mcq_protocol import _extract_letter
from universal_eval.model.reference_sources.minicpmv46_original_window_merger_patch import _merger_forward_per_image
MID='openbmb/MiniCPM-V-4.6';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False)
def arr(t):
 t=t.detach().cpu().contiguous();return t.view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.numpy()
class MiniCPMV46OwnNative(IModel):
 def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kw):
  assert protocol=={'model_id':MID} and npu_devices==[4];self.protocol_file=Path(protocol_file);self.protocol_sha=protocol_sha256
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers'
 def download(self,*a,**kw):raise RuntimeError('Only original own existing local assets')
 def load(self,model_path,**kw):
  assert sha(self.protocol_file)==self.protocol_sha;self.p=js(self.protocol_file);self.W=Path(model_path);assert self.p['model_id']==MID and str(self.W)==self.p['weight_path']
  for n,h in self.p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in self.p['static_evidence_SHA256'].items():assert sha(n)==h,n
  for n,h in self.p['native_libraries_SHA256'].items():assert sha(n)==h,n
  for n,h in self.p['own_assets_SHA256'].items():assert sha(self.W/n)==h,n
  for n,v in self.p['own_weight_files'].items():assert sha(self.W/n)==v['SHA256'] and (self.W/n).stat().st_size==v['bytes']
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert info.returncode==0 and 'No running processes found in NPU 4' in info.stdout
  self.device=torch.device('npu:4');torch.npu.set_device(self.device);torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  cfg=AutoConfig.from_pretrained(str(self.W),local_files_only=True,trust_remote_code=False);cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='eager'
  with torch.device('meta'):self.model=MiniCPMV4_6ForConditionalGeneration(cfg)
  expected=dict(self.model.named_parameters(remove_duplicate=False));source=js(self.p['own_source_manifest']);assert set(expected)-set(source)=={'lm_head.weight'} and cfg.tie_word_embeddings
  actual={}
  for fn in self.p['own_weight_files']:
   with safe_open(str(self.W/fn),framework='pt',device='cpu') as f:
    for n in f.keys():
     t=f.get_tensor(n);v=source[n];assert t.dtype==torch.bfloat16 and list(t.shape)==v['shape']==list(expected[n].shape) and tsha(t)==v['source_SHA256']==v['cast_SHA256']
     target=t.to(self.device);parent,leaf=n.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));assert tsha(target)==v['source_SHA256'];actual[n]=dict(v,actual_NPU_SHA256=tsha(target),actual_NPU_device=str(target.device))
  assert set(actual)==set(source);self.model.lm_head.weight=self.model.model.language_model.embed_tokens.weight
  rotary=Qwen3_5TextRotaryEmbedding(cfg.text_config,device='cpu');buffers={}
  for n,t in list(self.model.named_buffers(remove_duplicate=False)):
   assert n in ['model.language_model.rotary_emb.inv_freq','model.language_model.rotary_emb.original_inv_freq'];v=getattr(rotary,n.rsplit('.',1)[-1]);assert tuple(t.shape)==tuple(v.shape)
   parent,leaf=n.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,v.to(self.device));buffers[n]=dict(shape=list(v.shape),dtype=str(v.dtype),CPU_analytic_SHA256=tsha(v),actual_NPU_SHA256=tsha(self.model.get_buffer(n)))
  self.model.eval();assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters())
  self.model.generation_config=GenerationConfig.from_pretrained(str(self.W),local_files_only=True)
  assert self.model.generation_config.eos_token_id==248044 and self.model.generation_config.repetition_penalty in [None,1.,1] and self.model.generation_config.no_repeat_ngram_size in [None,0]
  patched=[]
  for n,m in self.model.named_modules():
   if m.__class__.__name__=='MiniCPMV4_6ViTWindowAttentionMerger':m.forward=types.MethodType(_merger_forward_per_image,m);patched.append(n)
   if m.__class__.__name__=='Qwen3_5GatedDeltaNet':
    assert m.chunk_gated_delta_rule.__name__=='torch_chunk_gated_delta_rule' and m.recurrent_gated_delta_rule.__name__=='torch_recurrent_gated_delta_rule' and m.causal_conv1d_fn is None,'Only installed genuine native torch fallback; no CUDA-only borrowed kernel'
  assert patched
  self.inputs=OwnMiniCPMV46Inputs(self.W);self.tokenizer=self.inputs.tokenizer;self.front=js(self.p['own_frontend200']);assert len(self.front)==200;self.stops=[248044]
  stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');self.home=Path.cwd()/'workspace/diagnostics/model_runs'/('MiniCPMV46_own_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(exist_ok=False)
  self.weights=self.home/'all_own_source_actual_NPU.json';put(self.weights,dict(weights=actual,own_tied_head=True,tied_alias={'lm_head.weight':'model.language_model.embed_tokens.weight'},buffers=buffers,original_window_merger_patch_instances=patched))
  self.index=self.report_index=0;self.details=[];self.batches=[];self.handles=[];self.active=self.failed=self.profile_done=False;self.profiler=None;self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch();self.profile=self.home/'npu_trace.json';self._install()
  libs=dict(self.p['native_libraries_SHA256'])
  for m in [torch,torch_npu]:f=inspect.getfile(m);libs[f]=sha(f)
  self.lineage=dict(hardware='Ascend910B',devices=[4],runtime='Own native MiniCPMV4_6 vision27 + Qwen3_5 text24:18 original linear states/6 full KV; native BF16 and original F32 recurrent accumulation',runtime_dtype='BF16',max_new_tokens=2048,head_size=248094,use_cache=True,actual_outer_batch_size=4,actual_inner_batch_size=1,protocol_file=str(self.protocol_file),protocol_SHA256=self.protocol_sha,evidence_root=str(self.home),weight_evidence=str(self.weights),weight_evidence_SHA256=sha(self.weights),sample_trace=str(self.trace),final_trace_manifest=str(self.home/'completed_trace.json'),runtime_library_source_SHA256=libs,historical_source_boundary=self.p['historical_source_boundary'])
 def _record(self,name,t):
  assert torch.is_tensor(t) and t.device==self.device,(name,getattr(t,'device',None));a=arr(t);meta=dict(array=name,shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(a.tobytes()).hexdigest())
  reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+a.nbytes,'Original hard/log reserve; preserve failed attempt'
  assert name not in self.saved;self.saved.add(name)
  with self.archive.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,a,allow_pickle=False)
  return meta
 def _install(self):
  lm=self.model.model.language_model
  def lm_pre(m,args,kw):
   if not self.active:return
   step=len(self.frames);self.frame=dict(step=step,attention=[],linear_projected={},cache=[])
   embeds=kw.get('inputs_embeds');assert embeds is not None and embeds.device==self.device
   if not step:self.activations['prefill_inputs_embeds']=self._record('prefill_inputs_embeds',embeds);self.N=embeds.shape[1]
  def head(m,args,out):
   if self.active:
    step=len(self.frames);row=out[0,-1];assert row.shape==(248094,) and row.dtype==torch.bfloat16
    self.frame['head']=self._record(f'f{step}_head',row);self.frame['head_input']=self._record(f'f{step}_head_input',args[0][0,-1]);self.frame['argmax']=int(row.argmax())
  def post(m,args,kw,out):
   if not self.active:return
   step=len(self.frames);cache=out.past_key_values;assert len(cache.layers)==24;full=linear=0
   for j,layer in enumerate(cache.layers):
    fields={}
    for key in ['keys','values','conv_states','recurrent_states']:
     val=getattr(layer,key,None)
     if val is None:continue
     if torch.is_tensor(val):val=[val]
     for q,t in enumerate(val):
      if t is not None:
       assert torch.is_tensor(t);fields[f'{key}.{q}']=self._record(f'f{step}_l{j}_{key}_{q}',t)
    if j%4==3:
     full+=1;assert set(fields)=={'keys.0','values.0'}
     for k,v in fields.items():assert v['shape']==[1,2,self.N+step,256] and v['dtype']=='torch.bfloat16'
    else:
     linear+=1;assert set(fields)=={'conv_states.0','recurrent_states.0'}
     assert fields['conv_states.0']['shape']==[1,6144,4] and fields['recurrent_states.0']['shape']==[1,16,128,128] and fields['recurrent_states.0']['dtype']=='torch.float32'
    self.frame['cache'].append(dict(layer=j,fields=fields))
   assert full==6 and linear==18;self.frames.append(self.frame)
   if self.profiler is not None:self.profiler.stop();self.profiler.export_chrome_trace(str(self.profile));self.profiler=None;self.profile_done=True
  self.handles.extend([lm.register_forward_pre_hook(lm_pre,with_kwargs=True),self.model.lm_head.register_forward_hook(head),self.model.register_forward_hook(post,with_kwargs=True)])
  for j,layer in enumerate(lm.layers):
   att=getattr(layer,'self_attn',None)
   if att is not None:
    def pre(m,args,kw,j=j):
     if not self.active:return
     z={'layer':j}
     for k in ['attention_mask','position_embeddings']:
      v=kw.get(k);vv=v if isinstance(v,(tuple,list)) else [v];z[k]=[self._record(f'f{len(self.frames)}_attn{j}_{k}{q}',t) for q,t in enumerate(vv) if torch.is_tensor(t)]
     self.frame['attention'].append(z)
    self.handles.append(att.register_forward_pre_hook(pre,with_kwargs=True))
   delta=getattr(layer,'linear_attn',None)
   if delta is not None:
    def projection(m,args,out,j=j):
     if self.active:self.frame['linear_projected'][str(j)]=self._record(f'f{len(self.frames)}_linear{j}_projected',out)
    self.handles.append(delta.in_proj_qkv.register_forward_hook(projection))
  for prefix,layers in [('vision',self.model.model.vision_model.encoder.layers),('LM',lm.layers)]:
   for j,layer in enumerate(layers):
    def stage(m,args,out,prefix=prefix,j=j):
     if not self.active:return
     key=f'{prefix}_{j}';self.counts[key]=self.counts.get(key,0)+1
     if not self.frames:
      t=out[0] if isinstance(out,tuple) else out
      if torch.is_tensor(t):self.activations[key]=self._record('stage_'+key,t) if self.index<3 else dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tsha(t))
    self.handles.append(layer.register_forward_hook(stage))
  original=self.model.model.get_image_features
  def features(*a,**kw):
   result=original(*a,**kw)
   if self.active:self.activations['own_visual_features']=self._record('own_visual_features',torch.cat(result.pooler_output,dim=0))
   return result
  self.model.model.get_image_features=features
 def predict(self,prompt,media_path=None):
  assert not self.failed;i=self.index;self.frames=[];self.activations={};self.counts={};self.saved=set();self.active=False;self.archive=None
  try:
   ref=self.front[i];assert prompt==ref['actual_model_prompt'] and sha(media_path)==ref['media_file_SHA256'];text,v,opts,model_prompt=self.inputs.prepare(prompt,media_path);assert text==ref['text'] and model_prompt==ref['own_saved_MCq_model_prompt']
   for n,t in flatten(v).items():assert tsha(t)==ref['tensor_metadata'][n]['SHA256']
   reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes'];assert shutil.disk_usage(self.home).free>reserve+self.p['per_sample_lossless_space_estimate_bytes']
   ap=self.home/f'sample_{i:05d}.npz';self.archive=zipfile.ZipFile(ap,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1)
   data={k:t.to(self.device,dtype=torch.bfloat16 if t.is_floating_point() else t.dtype) if torch.is_tensor(t) else t for k,t in v.items()}
   for n,t in flatten(data).items():self._record('input_'+n.replace('.','_'),t)
   self.active=True
   if not self.profile_done:
    self.profiler=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.home/'profiler')));self.profiler.start()
   with torch.inference_mode():out=self.model.generate(**data,max_new_tokens=2048,do_sample=False,temperature=0.0,use_cache=True)
   self.active=False;tokens=out[0,len(ref['input_ids']):].detach().cpu().tolist();assert tokens==[f['argmax'] for f in self.frames]
   raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip();letter=_extract_letter(raw,opts);returned=raw.rstrip()+'\n\n<answer>'+letter+'</answer>' if letter else raw
   self.archive.close();self.archive=None
   rec=dict(index=i,actual_model_prompt=prompt,own_saved_MCq_model_prompt=model_prompt,media_path=str(media_path),input_ids=ref['input_ids'],generated_tokens=tokens,original_native_raw_output=raw,raw_output=returned,original_MCq_normalized_letter=letter,reference=ref['reference'],error=None,actual_native_forwards=len(self.frames),frames=self.frames,activation_prefill=self.activations,activation_counts=self.counts,stop_ids=self.stops,original_budget_hit=len(tokens)==2048 and tokens[-1] not in self.stops,archive=str(ap),archive_SHA256=sha(ap),archive_bytes=ap.stat().st_size,complete_all24_hybrid_cache_values_losslessly_stored=True)
   rp=self.home/f'sample_{i:05d}.json';put(rp,rec);small=dict(index=i,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),actual_model_prompt=prompt,actual_native_forwards=len(tokens),generated_tokens=tokens,raw_output=returned,error=None)
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(small,ensure_ascii=False)+'\n')
   self.details.append(small);self.index+=1;LOG.info('MiniCPM-V-4.6 own sample %d: %d actual forwards',i+1,len(tokens));return returned
  except BaseException:
   self.active=False;self.failed=True
   if self.profiler is not None:self.profiler.stop();self.profiler=None
   if self.archive is not None:self.archive.close();self.archive=None
   put(self.home/f'sample_{i:05d}_failed.json',dict(index=i,committed_native_forwards=len(self.frames),error=traceback.format_exc()));raise
 def batch_predict(self,batch):
  assert 1<=len(batch)<=4;rec=dict(indices=list(range(self.index,self.index+len(batch))),status='running');self.batches.append(rec)
  try:z=[self.predict(p,m) for p,m in batch];rec['status']='completed';return z
  except BaseException:rec['status']='failed';raise
 def get_sample_execution(self,prompt,media_path):
  i=self.report_index;self.report_index+=1
  if not i:put(self.home/'completed_trace.json',dict(samples=self.index,failed=self.failed,sample_trace_SHA256=sha(self.trace),actual_outer_batches=self.batches,first_real_vision_prefill_profiler_SHA256={str(p):sha(p) for p in (self.home/'profiler').rglob('*') if p.is_file()}))
  return self.details[i]
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  for h in getattr(self,'handles',[]):h.remove()
  self.model=None;torch.npu.empty_cache()

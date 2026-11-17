"""Own Donut checkpoint, original FP16/greedy/position cap; observation only."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import contextlib,datetime,functools,hashlib,inspect,json,logging,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,torch_npu
from PIL import Image
from universal_eval.model.interface import IModel

MID='katanaml-org/invoices-donut-model-v1'
log=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(t):return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=fp(t))
def cpu(t):return t.detach().cpu().contiguous().clone()
def runtime_files():
 paths=set()
 for m in list(sys.modules.values()):
  n=getattr(m,'__file__',None)
  if n and Path(n).is_file() and not str(n).startswith('/tmp/'):
   p=Path(n)
   if p.suffix in ['.py','.so'] or '.so.' in p.name:paths.add(p.resolve())
 for line in Path('/proc/self/maps').read_text().splitlines():
  n=line.split()[-1]
  if n.startswith('/') and ('.so' in Path(n).name) and Path(n).is_file():paths.add(Path(n).resolve())
 return {str(p):sha(p) for p in sorted(paths)}

class DonutInvoiceOwnFP16(IModel):
 def __init__(self,protocol_file=None,protocol_sha256=None,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.protocol_file=str(Path(protocol_file).resolve());self.p=json.loads(Path(protocol_file).read_bytes());assert self.p['model_id']==MID
  self.devices=list(npu_devices or [4]);assert self.devices==[4];self.device=torch.device('npu:4');torch.set_num_threads(4)
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
  self.home=Path(evidence_dir).resolve()/('DonutInvoice_own_FP16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(parents=True,exist_ok=False)
  self.trace=self.home/'actual_sample_trace.jsonl';self.calls=0;self.details={};self.phase='idle';self.profile_done=False;self.profile_active=False;self.lineage={};self.loaded=False
  self.expected_count=200
  if '--config' in sys.argv:
   import yaml
   c=yaml.safe_load(Path(sys.argv[sys.argv.index('--config')+1]).read_bytes());self.expected_count=int(c.get('sample_count',200))
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'VisionEncoderDecoderModel'
 @property
 def inference_framework(self):return 'Own genuine HF Donut Swin/MBart native NPU FP16'
 def download(self,*a,**k):raise RuntimeError('Downloads are not authorized')
 def load(self,model_path,**kwargs):
  from transformers import VisionEncoderDecoderModel,DonutProcessor
  from transformers.generation.logits_process import LogitsProcessor
  p=self.p;root=Path(model_path).resolve();assert str(root)==p['weight_path'];assert sha(root/'pytorch_model.bin')==p['weight_SHA256']
  for n,h in p['own_asset_SHA256'].items():assert sha(root/n)==h
  for n,h in p['implementation_source_SHA256'].items():assert sha(Path('/root/projects/comperror')/n)==h
  assert sha(p['own_frontend200'])==p['own_frontend200_SHA256'];self.front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];assert len(self.front)==200
  # The original saved auto path resolves to FP16 on the accelerator. Loading
  # on CPU here only reads/casts own tensors; there is no learned CPU forward.
  source=torch.load(root/'pytorch_model.bin',map_location='cpu',weights_only=True);assert isinstance(source,dict) and len(source)==484
  model,info=VisionEncoderDecoderModel.from_pretrained(root,torch_dtype=torch.float16,local_files_only=True,output_loading_info=True)
  assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys')
  state=model.state_dict();assert set(state)==set(source);load=[]
  for n,t in source.items():
   expected=t.half() if t.is_floating_point() else t;assert record(state[n])==record(expected),(n,'Own native CPU load/cast mismatch')
   load.append(dict(name=n,source=record(t),original_accelerator_cast=record(expected)))
  assert model.config.encoder.model_type=='donut-swin' and model.config.decoder.model_type=='mbart'
  assert len(model.decoder.model.decoder.layers)==4 and model.config.decoder.vocab_size==57566 and model.config.decoder.max_position_embeddings==1536
  assert model.config.decoder.tie_word_embeddings and fp(source['decoder.lm_head.weight'].half())==fp(source['decoder.model.decoder.embed_tokens.weight'].half())
  torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for n in ['conv','matmul']:getattr(torch.npu,n).allow_hf32=False
  self.model=model.to(self.device).eval();self.model.tie_weights();actual=self.model.state_dict();assert set(actual)==set(source)
  for row in load:
   n=row['name'];t=actual[n];assert t.device==self.device and not t.is_meta and record(t)==row['original_accelerator_cast'];row['actual_NPU']=record(t);row['device']=str(t.device)
  dec=self.model.decoder.model.decoder;assert self.model.decoder.lm_head.weight.data_ptr()==dec.embed_tokens.weight.data_ptr()
  self.processor=DonutProcessor.from_pretrained(root,local_files_only=True);self.tok=self.processor.tokenizer
  assert len(self.tok)==57566 and self.model.generation_config.decoder_start_token_id==57565 and self.model.generation_config.eos_token_id==2
  self.limit=min(int(self.model.config.decoder.max_position_embeddings),int(dec.embed_positions.num_embeddings)-int(dec.embed_positions.offset));assert self.limit==1536
  self.loading=self.home/'all484_own_source_CPUcast_actual_NPU_SHA256.json';self.loading.write_bytes(json.dumps(load,indent=2).encode())
  del source,state,actual;self.loaded=True
  owner=self
  class IdentityScoreObservation(LogitsProcessor):
   def __call__(self,input_ids,scores):
    assert scores.device==owner.device and scores.shape==(1,57566)
    owner.processed.append(cpu(scores));owner.processor_prefix_ids.append(input_ids.detach().cpu().tolist());return scores
  original_processors=self.model._get_logits_processor
  @functools.wraps(original_processors)
  def processors(*args,**kw):
   values=original_processors(*args,**kw)
   if self.phase=='generate':
    gc=kw.get('generation_config',args[0] if args else None);self.processor_generation_config={k:v for k,v in gc.to_dict().items() if not k.startswith('_')}
    self.processor_classes=[type(v).__name__ for v in values];values.append(IdentityScoreObservation())
   return values
  self.model._get_logits_processor=processors
  def vision(m,args,out):
   if self.phase=='generate':self.encoder_calls+=1;self.vision=cpu(out.last_hidden_state)
  def head(m,args,out):
   if self.phase=='generate':
    assert out.dtype==torch.float16 and out.device==self.device and out.shape[-1]==57566 and torch.isfinite(out).all()
    self.head_inputs.append(cpu(args[0][:,-1,:]));self.head_outputs.append(cpu(out[:,-1,:]))
  def self_input(i):
   def hook(m,args,kw):
    if self.phase!='generate':return
    x=kw.get('hidden_states',args[0] if args else None);assert x is not None and x.device==self.device and x.dtype==torch.float16
    self.self_inputs[i].append(cpu(x));mask=kw.get('attention_mask')
    self.masks[i].append(None if mask is None else cpu(mask))
   return hook
  def positions(m,args,kw,out):
   if self.phase!='generate':return
   ids=args[0] if args else kw.get('input_ids');past=args[1] if len(args)>1 else kw.get('past_key_values_length',0);pos=kw.get('position_ids')
   self.position_rows.append(dict(input_ids=ids.detach().cpu().tolist(),past_length=int(past),position_ids=None if pos is None else pos.detach().cpu().tolist(),output=record(out)));self.position_values.append(cpu(out))
  self.model.encoder.register_forward_hook(vision);self.model.decoder.lm_head.register_forward_hook(head);dec.embed_positions.register_forward_hook(positions,with_kwargs=True)
  for i,l in enumerate(dec.layers):l.self_attn.register_forward_pre_hook(self_input(i),with_kwargs=True)
  original_forward=self.model.forward;sig=inspect.signature(original_forward)
  @functools.wraps(original_forward)
  def forward(*a,**k):
   out=original_forward(*a,**k)
   if self.phase=='generate':
    cache=out.past_key_values;assert cache is not None
    for kind in ['self_attention_cache','cross_attention_cache']:
     c=getattr(cache,kind);assert len(c.layers)==self.model.config.decoder.num_hidden_layers
     for i,l in enumerate(c.layers):
      if i>=4:assert l.keys is None and l.values is None;continue
      for kv in ['keys','values']:
       t=getattr(l,kv);assert t.device==self.device and t.dtype==torch.float16 and t.shape[:2]==(1,16) and t.shape[-1]==64
       key=(kind,i,kv)
       if key not in self.cache_previous:self.cache_base[key]=cpu(t);self.cache_previous[key]=t.detach().clone();self.cache_appends[key]=[]
       else:
        prev=self.cache_previous[key]
        if kind=='cross_attention_cache':assert torch.equal(t,prev)
        else:assert t.shape[2]==prev.shape[2]+1 and torch.equal(t[:,:,:prev.shape[2]],prev);self.cache_appends[key].append(cpu(t[:,:,prev.shape[2]:]));self.cache_previous[key]=t.detach().clone()
       self.cache_final[key]=record(t)
    self.forwards+=1
   if self.profile_active:
    torch.npu.synchronize(self.device);self.prof.__exit__(None,None,None);self.profile_active=False;self.prof.export_chrome_trace(str(self.home/'npu_trace.json'));self.cwd.__exit__(None,None,None);self.profile_done=True
   return out
  self.model.forward=forward;assert inspect.signature(self.model.forward)==sig
  self.lineage=dict(hardware='Ascend910B',devices=self.devices,protocol_file=self.protocol_file,protocol_SHA256=sha(self.protocol_file),evidence_root=str(self.home),sample_trace=str(self.trace),runtime_evidence=str(self.home/'actual_runtime_source_and_shared_libraries.json'),all484_own_source_actual_NPU=str(self.loading),all484_own_source_actual_NPU_SHA256=sha(self.loading),own_declared_tied_head_restored_from_equal_own_source_only=True,current_native_attention=dict(root=self.model.config._attn_implementation,encoder=self.model.config.encoder._attn_implementation,decoder=self.model.config.decoder._attn_implementation),actual_decoder_layers=4,native_cache_allocated_slots=self.model.config.decoder.num_hidden_layers,full_vocabulary_size=57566,execution_dtype='FP16',original_requested_max_new_tokens=2048,own_position_limit=1536,original_raw_prompt_length_slicing_preserved=True,no_extra_logits_filter_or_precision_budget_change=True,profiler_trace=str(self.home/'npu_trace.json'),profiler_scope='Actual first vision and first native prefill only, not all generated steps',historical_A100_engine_or_tensor_capture_claimed=False,original_A100_reference_samples=8,original_MLU_BLEU_not_substituted_for_A100_ROUGE_L=True)
 def reset(self):
  self.encoder_calls=0;self.vision=None;self.head_inputs=[];self.head_outputs=[];self.processed=[];self.processor_prefix_ids=[];self.processor_classes=[];self.self_inputs={i:[] for i in range(4)};self.masks={i:[] for i in range(4)};self.position_rows=[];self.position_values=[];self.cache_previous={};self.cache_base={};self.cache_appends={};self.cache_final={};self.forwards=0
 def predict(self,prompt,media_path=None):
  i=self.calls;self.calls+=1;self.reset();start=time.monotonic();e=dict(index=i,prompt=prompt,media_path=media_path,error=None)
  try:
   assert shutil_disk_free(self.home)>=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes']+self.p['per_sample_lossless_space_estimate_bytes']
   ref=self.front[i];imagepath=Path(media_path).resolve();assert prompt==ref['prompt'] and str(imagepath)==ref['resolved_media_path'] and sha(imagepath)==ref['image_SHA256']
   with Image.open(imagepath) as im:im=im.convert('RGB');pv=self.processor.image_processor(images=im,return_tensors='pt')['pixel_values']
   assert record(pv)==ref['CPU_F32_pixels'];px=pv.half().to(self.device);assert record(px)==ref['original_FP16_pixels']
   ids=self.tok(prompt,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device);assert ids.cpu().tolist()==[ref['original_prompt_decoder_ids']]
   cap=max(1,self.limit-ids.shape[1]-1);effective=min(2048,cap);assert effective==ref['original_poscap_effective_max_new_tokens']
   g,_=self.model._prepare_generation_config(None,max_new_tokens=effective,do_sample=False,temperature=0.0)
   e.update(source_image_SHA256=sha(imagepath),CPU_F32_pixels=record(pv),NPU_FP16_pixels=record(px),original_decoder_prompt_ids=ids.cpu().tolist(),original_requested_max_new_tokens=2048,original_poscap_effective_max_new_tokens=effective,effective_generation_config=g.to_dict(),no_extra_OCR_charset_or_grammar_logits_processor=True)
   self.phase='generate'
   if not self.profile_done:
    self.cwd=contextlib.chdir(self.home);self.cwd.__enter__();self.prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False);self.prof.__enter__();self.profile_active=True
   with torch.inference_mode():out=self.model.generate(pixel_values=px,decoder_input_ids=ids,max_new_tokens=effective,do_sample=False,temperature=0.0,logits_processor=None)
   if hasattr(out,'sequences'):out=out.sequences
   seq=out[0].cpu().tolist();n=self.forwards;assert n==len(self.head_inputs)==len(self.head_outputs)==len(self.processed)==len(self.position_rows)==len(self.processor_prefix_ids) and 0<n<=effective and self.encoder_calls==1
   actual_prefix=self.processor_prefix_ids[0][0];assert seq[:len(actual_prefix)]==actual_prefix and len(seq)==len(actual_prefix)+n and all(seq[:len(actual_prefix)+j]==self.processor_prefix_ids[j][0] for j in range(n))
   assert all(seq[len(actual_prefix)+j]==int(self.processed[j].float().argmax(-1)) for j in range(n))
   decoded_ids=seq[ids.shape[1]:] if len(seq)>ids.shape[1] else seq;raw=self.processor.batch_decode([decoded_ids],skip_special_tokens=True)[0].strip()
   arrays={'pixels_FP16':cpu(px).numpy(),'encoder_FP16':self.vision.numpy(),'head_hidden_FP16':torch.cat(self.head_inputs,0).numpy(),'raw_full57566_FP16':torch.cat(self.head_outputs,0).numpy(),'processed_full57566_F32':torch.cat(self.processed,0).numpy()}
   for k,t in self.cache_base.items():
    name='_'.join(map(str,k));arrays[name+'_base']=t.numpy();xs=self.cache_appends[k];arrays[name+'_append']=torch.cat(xs,2).numpy() if xs else np.empty((1,16,0,64),dtype=np.float16)
   for k,vs in self.self_inputs.items():arrays[f'self_input_{k}']=torch.cat(vs,1).numpy()
   for j,v in enumerate(self.position_values):arrays[f'position_values_{j:05d}']=v.numpy()
   mask_rows={}
   for k,vs in self.masks.items():
    mask_rows[str(k)]=[]
    for j,v in enumerate(vs):
     name=f'self_mask_{k}_{j:05d}';mask_rows[str(k)].append(None if v is None else name)
     if v is not None:arrays[name]=v.numpy()
   archive=self.home/f'sample_{i:05d}.npz';np.savez_compressed(archive,**arrays)
   e.update(archive=str(archive),archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,actual_native_forwards=n,full_generation_sequence_ids=seq,actual_generation_prefix_ids=actual_prefix,new_model_token_ids=seq[len(actual_prefix):],original_raw_decode_ids=decoded_ids,raw_output=raw,original_budget_hit=(n==effective),vision_calls=1,vision_output=record(self.vision),original_processor_classes=self.processor_classes,actual_processor_generation_config=self.processor_generation_config,positions=self.position_rows,masks=mask_rows,native_cache_final={ '_'.join(map(str,k)):v for k,v in self.cache_final.items()},native_cache_prefix_and_cross_constant_checked_every_step=True,all_generated_head_and_KV_values_saved=True,actual_NPU_card=4,elapsed_seconds=time.monotonic()-start)
   log.info('Own Donut NPU sample=%s forwards=%s cap=%s raw=%r',i,n,effective,raw[:160]);return raw
  except BaseException:e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   if self.profile_active:
    try:self.prof.__exit__(None,None,None)
    finally:self.profile_active=False;self.cwd.__exit__(None,None,None)
   self.phase='idle';e['elapsed_seconds']=time.monotonic()-start;self.details[(prompt.strip(),media_path)]=e
   if self.calls==self.expected_count and self.loaded:
    r=self.home/'actual_runtime_source_and_shared_libraries.json';r.write_bytes(json.dumps(runtime_files(),ensure_ascii=False,indent=2).encode());e['final_runtime_evidence_SHA256']=sha(r)
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
 def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing actual execution'})
 def get_execution_lineage(self):
  return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

def shutil_disk_free(p):
 import shutil
 return shutil.disk_usage(p).free

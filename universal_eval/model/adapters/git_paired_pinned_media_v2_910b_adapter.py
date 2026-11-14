"""Native source-F32 GIT image-only caption, reviewed original paired128/300 generation and explicit cache-position compatibility."""
import os,contextlib,datetime,hashlib,json,logging,pathlib,struct,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.git_paired_caption_910b')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
@contextlib.contextmanager
def original_tensor_source(path,review):
 assert path.stat().st_size==review['size']
 if review['format']=='safetensors':
  with path.open('rb') as f:n=f.read(8);header=f.read(struct.unpack('<Q',n)[0])
  assert hashlib.sha256(n+header).hexdigest()==review['header_sha256']
  from safetensors import safe_open
  with safe_open(str(path),framework='pt',device='cpu') as f:yield f
 else:
  assert review['format']=='pytorch_weights_only_mmap' and sha(path)==review['file_sha256']
  tensors=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  assert isinstance(tensors,dict) and all(isinstance(k,str) and torch.is_tensor(v) for k,v in tensors.items())
  class ReadOnlySource:
   def keys(self):return tensors.keys()
   def get_tensor(self,key):return tensors[key]
  yield ReadOnlySource()
  del tensors

def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
class GitPairedPinnedMedia910BAdapter(IModel):
 def __init__(self,protocol=None,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  self.protocol=dict(protocol);self.devices=list(npu_devices or [14]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]))
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('GIT_paired_caption_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'visual_features').mkdir()
  self.trace=self.evidence/'sample_execution.jsonl';self.calls=0;self.details={};self.lineage={};self.profile_done=False
 @property
 def model_name(self):return getattr(self,'_model_name','unloaded')
 @property
 def model_type(self):return 'GitForCausalLM'
 @property
 def inference_framework(self):return 'PyTorch + torch_npu native GIT sourceF32'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')
 def load(self,model_path,**kwargs):
  from safetensors import safe_open
  from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,GitProcessor,GitForCausalLM,GenerationConfig
  p=self.protocol;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and wp.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels');self._model_name=wp.parent.name+'/'+wp.name;assert self.model_name==p['model_id'] and p['model_id'] in {'microsoft/git-base','microsoft/git-large','microsoft/git-large-textcaps'}
  for rel,digest in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(wp/rel)==digest
  for path,digest in p['runtime_sha256'].items():assert sha(path)==digest
  for key in ['reference_report','reference_log','historical_adapter','historical_parent']:assert sha(p[key])==p[key+'_sha256']
  self.dataset=pathlib.Path(p['dataset_path']);assert sha(self.dataset)==p['dataset_sha256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b'
  assert p['max_new_tokens']=={'microsoft/git-base':128,'microsoft/git-large':128,'microsoft/git-large-textcaps':300}[p['model_id']] and p['extract_answer'] is False and p['skip_text_input'] is True and p['source_dtype']=='F32'
  old=pathlib.Path(p['historical_adapter']).read_text();assert '_eff_max_new' in old and '_git_pos_fixed' in old and 'min_new_tokens' not in old
  assert sha(p['reference_fixture'])==p['reference_fixture_sha256'] and p['dataset_expected_count']==1459 and p['original_batch_size']==1
  for path,digest in p['original_source_sha256'].items():assert sha(path)==digest
  torch.npu.set_device(self.device)
  for name in ['conv','matmul']:getattr(torch.npu,name).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.architectures==['GitForCausalLM'] and cfg.tie_word_embeddings is False and not getattr(cfg,'quantization_config',None);cfg._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
  with torch.device('meta'):self.model=GitForCausalLM(cfg)
  params=dict(self.model.named_parameters(remove_duplicate=False));buffers=dict(self.model.named_buffers());records=[];loaded=set()
  for shard,review in p['weight_headers'].items():
   path=wp/shard
   with original_tensor_source(path,review) as f:
    for key in f.keys():
     assert key not in loaded and key in (set(params)|set(buffers));value=f.get_tensor(key);expected=(params|buffers)[key];assert tuple(value.shape)==tuple(expected.shape) and value.dtype==(torch.float32 if key in params else torch.int64)
     if key in buffers:assert torch.equal(value,torch.arange(value.shape[-1]).reshape(value.shape))
     target=value.to(self.device).contiguous();digest=fp(value);assert fp(target)==digest;parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False) if key in params else target);loaded.add(key)
     records.append({'source_file':shard,'source_key':key,'target_key':key,'kind':'parameter' if key in params else 'original_integer_buffer','shape':list(value.shape),'dtype':str(value.dtype),'device':str(target.device),'source_sha256':digest,'NPU_sha256':fp(target)})
  assert loaded==set(params)|set(buffers) and len(params)==p['learned_parameter_count'] and len(buffers)==p['original_position_buffer_count']==2
  assert all(t.device==self.device and t.dtype==torch.float32 for t in self.model.parameters()) and all(t.device==self.device and t.dtype==torch.int64 for t in self.model.buffers());self.model.eval()
  self.tokenizer=AutoTokenizer.from_pretrained(wp,local_files_only=True);ip=AutoImageProcessor.from_pretrained(wp,local_files_only=True,backend='pil');self.processor=GitProcessor(image_processor=ip,tokenizer=self.tokenizer)
  gen=GenerationConfig.from_pretrained(wp,local_files_only=True);self.model.generation_config=gen;self.gen,_=self.model._prepare_generation_config(None,max_new_tokens=p['max_new_tokens'],do_sample=False)
  assert self.gen.to_dict()==p['effective_generation_config'] and self.gen.num_beams==1 and self.gen.use_cache is True and self.gen.repetition_penalty==1. and self.gen.min_new_tokens in (None,0)
  self.nvis=(ip.crop_size['height']//cfg.vision_config.patch_size)*(ip.crop_size['width']//cfg.vision_config.patch_size)+1;assert self.nvis==p['visual_token_count'] and p['max_new_tokens']<=cfg.max_position_embeddings-self.nvis-4
  self._phase='idle';self._positions=[];self._masks=[];self._features=None;self._vision=None;self._prefix=None;self._vision_calls=0;self._bridge_calls=0
  def text_positions(module,args,kw):
   ids=kw.get('input_ids');past=int(kw.get('past_key_values_length',0));supplied=kw.get('position_ids');length=ids.shape[1] if ids is not None else kw['inputs_embeds'].shape[1];start=max(0,past-self.nvis)
   resolved=torch.arange(start,start+length,dtype=torch.int64,device=self.device).reshape(1,-1)
   if supplied is not None:
    adjusted=supplied-past if past else supplied;assert torch.equal(adjusted,resolved),(past,supplied,resolved)
   kw['position_ids']=resolved;self._positions.append({'phase':self._phase,'past_total_cache_length':past,'input_token_ids':ids.detach().cpu().tolist() if ids is not None else None,'supplied_position_ids':supplied.detach().cpu().tolist() if supplied is not None else None,'resolved_text_position_ids':resolved.cpu().tolist()});return args,kw
  def image_features(module,args,out):self._vision=out.last_hidden_state;self._vision_calls+=1
  def projected_features(module,args,out):assert out.device==self.device and out.dtype==torch.float32 and torch.isfinite(out).all();self._features=out
  def decoder_input(module,args,kw):
   states=args[0] if args else kw['hidden_states'];mask=kw['attention_mask'];assert torch.is_tensor(mask) and mask.device==self.device
   if states.shape[1]>1:
    assert self._features is not None and states.shape[1]==self.nvis+1 and torch.equal(states[:,:self.nvis],self._features)
    self._prefix=states[:,:self.nvis].detach().clone();self._bridge_calls+=1
   self._masks.append({'phase':self._phase,'mask':tr(mask)})
  self.model.git.embeddings.register_forward_pre_hook(text_positions,with_kwargs=True);self.model.git.image_encoder.register_forward_hook(image_features);self.model.git.visual_projection.register_forward_hook(projected_features);self.model.git.encoder.register_forward_pre_hook(decoder_input,with_kwargs=True)
  weight_path=self.evidence/'weight_loading.json';weight_path.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
  for path in list(p['runtime_sha256'])+list(p['original_source_sha256']):
   dest=snapshot/pathlib.Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(pathlib.Path(path).read_bytes())
  (snapshot/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2));(snapshot/'historical_reference_report.json').write_bytes(pathlib.Path(p['reference_report']).read_bytes());(snapshot/'historical_run.log').write_bytes(pathlib.Path(p['reference_log']).read_bytes())
  self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'source_dtype':'F32','protocol':'paired_GIT_original_image_only_caption','protocol_review':p,'precision':'All original sourceF32 learned tensors and2 originalI64 position buffers preserved with full CPU/NPU SHA identity; native vision/projector/BERT/LM head on NPU F32, HF32conv/matmul off, no learned-layer CPU offload or lowbit acceleration. Historical paired runtime explicitly used source F32; native NPU arithmetic may differ across hardware and is not claimed bitwise equal. Existing CANN GELU native approximate-none behavior remains unchanged and documented by original operator method note. No model/data/tokenizer download or replacement.','HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ['conv','matmul']},'all_expected_parameters_loaded':True,'weight_tensor_count':len(records),'learned_parameter_count':len(params),'original_position_buffer_count':2,'weight_loading_evidence':str(weight_path),'model_resource_root':str(wp),'model_resource_sha256':p['resource_sha256'],'runtime_source_sha256':p['runtime_sha256'],'weight_headers':p['weight_headers'],'effective_generation_config':self.gen.to_dict(),'effective_image_processor':ip.to_dict(),'visual_token_count':self.nvis,'per_sample_evidence':str(self.trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'architecture_source_snapshot':str(snapshot),'position_compatibility':p['canonical_text_position_compatibility'],'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
 def predict(self,prompt,media_path=None):
  self.calls+=1;started=time.monotonic();self._positions=[];self._masks=[];self._features=None;self._vision=None;self._prefix=None;self._vision_calls=0;self._bridge_calls=0;e={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
  try:
   media=pathlib.Path(media_path).resolve();assert str(media) in self.protocol['reference_first200_image_sha256'] and sha(media)==self.protocol['reference_first200_image_sha256'][str(media)]
   with Image.open(media) as im:image=im.convert('RGB')
   prepared=self.processor(images=[image],return_tensors='pt');assert set(prepared)=={'pixel_values'} and list(prepared['pixel_values'].shape)==[1,3,224,224]
   e.update(source_image_sha256=sha(media),CPU_pixel_values=tr(prepared['pixel_values']),model_receives_no_prompt_or_reference=True);inp={'pixel_values':prepared['pixel_values'].to(self.device)};self._phase='profile'
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
       out=self.model(**inp,input_ids=torch.tensor([[101]],device=self.device),use_cache=False,return_dict=True);assert torch.isfinite(out.logits).all();torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True
    self._phase='generate';out=self.model.generate(**inp,generation_config=self.gen,return_dict_in_generate=True,output_scores=False)
   seq=out.sequences[0].cpu().tolist();assert seq[0]==101 and len(seq)-1<=self.protocol['max_new_tokens'];raw=self.processor.batch_decode([seq],skip_special_tokens=True)[0].strip();truncated=len(seq)-1>=self.protocol['max_new_tokens'] and seq[-1]!=102
   positions=[r for r in self._positions if r['phase']=='generate'];assert len(positions)==len(seq)-1 and [r['resolved_text_position_ids'] for r in positions]==[[[i]] for i in range(len(seq)-1)]
   assert self._vision is not None and self._features is not None and self._prefix is not None and self._bridge_calls>0 and self._vision_calls>0 and torch.equal(self._features,self._prefix)
   cache=out.past_key_values;cache_length=cache.get_seq_length();assert cache_length==self.nvis+len(seq)-1
   cache_info=[{'key_shape':list(layer.keys.shape),'value_shape':list(layer.values.shape),'key_device':str(layer.keys.device),'value_device':str(layer.values.device)} for layer in cache.layers];assert len(cache_info)==self.model.config.num_hidden_layers and all(x['key_device']==x['value_device']==str(self.device) for x in cache_info)
   path=self.evidence/'visual_features'/('%06d.npz'%(self.calls-1));vision=self._vision.detach().cpu();features=self._features.detach().cpu();prefix=self._prefix.detach().cpu();np.savez_compressed(path,vision_output=vision.numpy(),projected_visual=features.numpy(),actual_decoder_visual_prefix=prefix.numpy())
   e.update(raw_output=raw,generation_sequence_ids=seq,generation_token_count=len(seq)-1,truncated=truncated,stop_reason='length' if truncated else 'eos' if seq[-1]==102 else 'other',text_positions=self._positions,attention_masks=self._masks,visual_feature_archive=str(path),visual_feature_archive_sha256=sha(path),vision_output=tr(vision),projected_visual=tr(features),actual_decoder_visual_prefix=tr(prefix),actual_visual_prefix_verified=True,vision_forward_calls=self._vision_calls,decoder_visual_bridge_calls=self._bridge_calls,final_cache_length=cache_length,cache_device_and_shapes=cache_info)
   log.info('GIT original NPU sample %s tokens=%s truncated=%s raw=%r',self.calls,len(seq)-1,truncated,raw[:180]);return raw
  except Exception:
   e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   e['elapsed_seconds']=time.monotonic()-started;self.details[(prompt.strip(),media_path)]=e
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._phase='idle';self._features=None;self._vision=None;self._prefix=None
 def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

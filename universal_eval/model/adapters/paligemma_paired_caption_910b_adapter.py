"""Native source-precision PaliGemma for the reviewed original paired/high7 caption protocols."""
import os,contextlib,pathlib,json,hashlib,struct,datetime,time,re,logging,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from jinja2 import Template
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.paligemma_paired_caption_910b')
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
def map_key(k):
 for old,new in [('vision_tower.vision_model.','model.vision_tower.'),('vision_tower.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.'),('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.')]:
  if k.startswith(old):return new+k[len(old):]
 return k
_ABBREV=re.compile(r'(?i)(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|No|Jr|Sr|U\.S|e\.g|i\.e|approx|Fig|Vol)$')
def original_first_sentence(text):
 if not text:return text
 buf=[]
 for i,ch in enumerate(text):
  buf.append(ch)
  if ch in '.!?' and (i==len(text)-1 or text[i+1].isspace()):
   token=''.join(buf).rstrip();last=token.split()[-1] if token.split() else ''
   if not _ABBREV.search(last.rstrip('.!?')):return token
 return text.strip()
class PaliGemmaPairedCaption910BAdapter(IModel):
 def __init__(self,protocol=None,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  self.p=dict(protocol);self.devices=list(npu_devices or [15]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.dtype={'F32':torch.float32,'BF16':torch.bfloat16}[self.p['source_dtype']]
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('PaliGemma_paired_caption_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'visual_features').mkdir()
  self.trace=self.evidence/'sample_execution.jsonl';self.calls=0;self.details={};self.lineage={};self.profile_done=False
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'PaliGemmaForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native PyTorch/torch_npu PaliGemma source'+self.p['source_dtype']
 def download(self,*a,**kw):raise RuntimeError('No model/data downloads permitted')
 def load(self,model_path,**kwargs):
  from safetensors import safe_open
  from transformers import AutoConfig,AutoTokenizer,AutoImageProcessor,PaliGemmaForConditionalGeneration,PaliGemmaProcessor,GenerationConfig
  from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
  from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
  p=self.p;assert p['prompt_mode']!='vllm_prefix_template','Original vLLM attention/position contract requires its own independent native implementation'
  wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and wp.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and wp.parent.name+'/'+wp.name==p['model_id']
  assert p['dataset_sha256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b' and sha(p['dataset_path'])==p['dataset_sha256'] and p['dataset_count']==1459
  for rel,digest in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(wp/rel)==digest
  for path,digest in dict(p['runtime_sha256'],**p['historical_sources']).items():assert sha(path)==digest
  for name in ['reference_report','reference_log']:assert sha(p[name])==p[name+'_sha256']
  assert not (wp/'adapter_config.json').exists() and not (wp/'adapter_model.safetensors').exists()
  if p.get('LoRA_backup'):raise RuntimeError('Merged LoRA requires a separate exact base/delta audit')
  previous=json.loads(pathlib.Path(p['reference_report']).read_text());assert previous['pipeline_config']['model']['name']==self.model_name and len(previous['detailed_results'])==200
  assert ('提取答案='+str(p['extract_answer'])) in pathlib.Path(p['reference_log']).read_text()
  self.dataset=pathlib.Path(p['dataset_path']);torch.npu.set_device(self.device)
  torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for name in ['conv','matmul']:getattr(torch.npu,name).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.architectures==['PaliGemmaForConditionalGeneration'] and cfg.tie_word_embeddings and not getattr(cfg,'quantization_config',None)
  cfg._attn_implementation='eager';cfg.text_config._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
  with torch.device('meta'):self.model=PaliGemmaForConditionalGeneration(cfg)
  params=dict(self.model.named_parameters());buffer_dtypes={n:t.dtype for n,t in self.model.named_buffers()};filled=set();loading=[];index=json.loads((wp/'model.safetensors.index.json').read_text())['weight_map']
  assert len(params)==p['learned_tensor_count'] and set(index.values())==set(p['weight_headers'])
  for shard,review in p['weight_headers'].items():
   path=wp/shard
   with path.open('rb') as f:n=f.read(8);header=f.read(struct.unpack('<Q',n)[0])
   assert path.stat().st_size==review['size'] and hashlib.sha256(n+header).hexdigest()==review['header_sha256']
   with safe_open(str(path),framework='pt',device='cpu') as f:
    assert set(f.keys())=={k for k,v in index.items() if v==shard}
    for key in f.keys():
     target_key=map_key(key);assert target_key in params and target_key not in filled;value=f.get_tensor(key);assert value.dtype==self.dtype and list(value.shape)==list(params[target_key].shape)
     target=value.to(self.device).contiguous();digest=fp(value);assert digest==fp(target);parent,leaf=target_key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));filled.add(target_key)
     loading.append({'source_file':shard,'source_key':key,'target_key':target_key,'shape':list(value.shape),'dtype':str(value.dtype),'device':str(target.device),'source_sha256':digest,'NPU_sha256':digest})
   log.info('PaliGemma paired loaded shard %s %s/%s',shard,len(filled),len(params))
  assert filled==set(params);self.model.tie_weights();assert self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight
  self.seq=(cfg.vision_config.image_size//cfg.vision_config.patch_size)**2;assert self.seq==p['image_seq_length'] and cfg.vision_config.image_size==p['expected_image_size']
  self.model.model.vision_tower.embeddings.position_ids=torch.arange(self.seq,dtype=torch.int64).reshape(1,-1).to(self.device)
  embedding=self.model.model.language_model.embed_tokens;embedding.embed_scale=torch.tensor(embedding.scalar_embed_scale,dtype=buffer_dtypes['model.language_model.embed_tokens.embed_scale']).to(self.device)
  rope={'gemma':GemmaRotaryEmbedding,'gemma2':Gemma2RotaryEmbedding}[cfg.text_config.model_type];self.model.model.language_model.rotary_emb=rope(cfg.text_config).to(self.device)
  assert all(t.device==self.device and t.dtype==self.dtype for t in self.model.parameters())
  assert all(not t.is_meta and t.device==self.device and t.dtype==buffer_dtypes[n] for n,t in self.model.named_buffers());self.model.eval()
  self.tok=AutoTokenizer.from_pretrained(wp,local_files_only=True);ip=AutoImageProcessor.from_pretrained(wp,local_files_only=True,backend='pil');ip.image_seq_length=self.seq;self.processor=PaliGemmaProcessor(image_processor=ip,tokenizer=self.tok);self.image_id=cfg.image_token_id;assert self.image_id==self.processor.image_token_id
  self.gen=GenerationConfig.from_dict(p['generation_config']);resolved,_=self.model._prepare_generation_config(self.gen);assert resolved.to_dict()==p['generation_config'] and self.gen.max_new_tokens==p['max_new_tokens'] and self.gen.use_cache is True and self.gen.num_beams==1 and self.gen.do_sample is False
  self.eos=self.gen.eos_token_id;self.eos=[self.eos] if isinstance(self.eos,int) else list(self.eos);self.template=Template(p['original_VLLM_template']) if p['prompt_mode']=='vllm_prefix_template' else None
  self._active=None;self._features=None;self._prefix=None;self._vision_calls=0;self._positions=None;self._masks=None
  def projected(module,args,out):
   assert out.device==self.device and out.dtype==self.dtype and torch.isfinite(out).all();self._features=out;self._prefix=None;self._vision_calls+=1
  def language_input(module,args,kw):
   embeds=kw.get('inputs_embeds')
   if self._prefix is not None or self._features is None or embeds is None or embeds.shape[:2]!=self._active.shape:return
   actual=embeds[self._active==self.image_id].reshape(self._features.shape);assert torch.equal(actual,self._features) and actual.abs().sum()>0;self._prefix=actual.detach().clone();self._positions=record(kw['position_ids']);mask=kw['attention_mask'];mask=mask if isinstance(mask,dict) else {'full_attention':mask};self._masks={}
   for name,value in mask.items():assert torch.is_tensor(value) and value.device==self.device and value.shape[-2:]==self._active.shape[-1:]*2 and (value==0).all();self._masks[name]=record(value)
  self.model.model.multi_modal_projector.register_forward_hook(projected);self.model.model.language_model.register_forward_pre_hook(language_input,with_kwargs=True)
  weights=self.evidence/'weight_loading.json';weights.write_text(json.dumps(loading,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
  for source in set(p['runtime_sha256'])|set(p['historical_sources']):
   path=pathlib.Path(source);dest=snapshot/path.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
  (snapshot/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2));(snapshot/'original_report.json').write_bytes(pathlib.Path(p['reference_report']).read_bytes());(snapshot/'original_run.log').write_bytes(pathlib.Path(p['reference_log']).read_bytes())
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'paired_paligemma_original_1459_caption','source_dtype':p['source_dtype'],'protocol_review':p,'all_expected_parameters_loaded':True,'weight_tensor_count':len(loading),'tied_output_head_restored':True,'weight_loading_evidence':str(weights),'per_sample_evidence':str(self.trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'image_seq_length':self.seq,'HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ['conv','matmul']},'analytic_buffers':{n:record(t) for n,t in self.model.named_buffers()},'text_inv_freq':self.model.model.language_model.rotary_emb.inv_freq.cpu().tolist(),'effective_image_processor':ip.to_dict(),'effective_generation_config':self.gen.to_dict(),'native_allocator_setting':'expandable_segments:True','native_computation':'Original sourceF32/BF16 learned tensors all NPU with full source/NPU SHA equality. Original native attention/softmax/matmul/cache and source weights retained, HF32off. CPU-constructed FP32 analytic RoPE and original native embedding-scale rounding preserved. Historical requested F16/BF16 may differ from sourceF32; explicit source preservation is disclosed, not a bitwise historical precision claim. No CPU learned-layer offload or low-bit acceleration. Allocator is process-local fragmentation policy only.','module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
 def predict(self,prompt,media_path=None):
  self.calls+=1;started=time.monotonic();self._features=None;self._prefix=None;self._vision_calls=0;self._positions=None;self._masks=None;e={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
  try:
   p=self.p;path=pathlib.Path(media_path).resolve();assert str(path) in p['reference_first200_image_sha256'] and sha(path)==p['reference_first200_image_sha256'][str(path)]
   with Image.open(path) as im:image=im.convert('RGB')
   text=prompt
   if p['strip_eval_instruction']:
    for suffix in [r'\n\n(?:请仔细分析图像和题目[^\n]*|Answer the question by providing only the option letter[^\n]*)\s*$',r'\n\nProvide only the final answer\.\s*$']:text=re.sub(suffix,'',text)
   text=p['prefix']+text+p['mc_instruction']
   if self.template is not None:text=self.template.render(messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'file://'+str(path)}},{'type':'text','text':text}]}])
   prepared=self.processor(text=text,images=image,return_tensors='pt');e.update(processor_text=text,source_image_sha256=sha(path),CPU_inputs={n:record(v) for n,v in prepared.items()},input_ids=prepared['input_ids'].tolist(),token_type_ids=prepared['token_type_ids'].tolist(),attention_mask=prepared['attention_mask'].tolist())
   labels=prepared.pop('labels',None);assert labels is None or (labels==-100).all();e['excluded_training_labels']=labels is not None
   assert (prepared['input_ids']==self.image_id).sum()==self.seq and (prepared['token_type_ids']==0).all()
   inputs={n:v.to(self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for n,v in prepared.items()};self._active=inputs['input_ids'];length=self._active.shape[-1];assert length<4096;e['actual_NPU_pixels']=record(inputs['pixel_values']);e['allocator_before']=torch.npu.memory_stats(self.device).get('num_ooms',0)
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
       outputs=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True);assert torch.isfinite(outputs.logits).all();torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True
    generated=self.model.generate(**inputs,generation_config=self.gen,return_dict_in_generate=True,output_scores=False)
   tokens=generated.sequences[0,length:].cpu().tolist();full=self.tok.decode(tokens,skip_special_tokens=True);raw=full if p['prompt_mode']=='vllm_prefix_template' else full.strip()
   if p['truncate_first_sentence']:raw=original_first_sentence(raw)
   truncated=len(tokens)>=p['max_new_tokens'] and (not tokens or tokens[-1] not in self.eos)
   assert self._prefix is not None and self._vision_calls>0 and torch.equal(self._features,self._prefix) and list(self._features.shape)==[1,self.seq,self.model.config.text_config.hidden_size]
   feature=self._features.detach().cpu();prefix=self._prefix.detach().cpu();archive=self.evidence/'visual_features'/('%06d.npz'%(self.calls-1));np.savez_compressed(archive,projected_visual_u8=feature.contiguous().view(torch.uint8).numpy(),actual_decoder_visual_prefix_u8=prefix.contiguous().view(torch.uint8).numpy())
   cache=generated.past_key_values;cache_length=int(cache.get_seq_length());assert cache_length==length+len(tokens)-1
   placements=[{'key_device':str(layer.keys.device),'value_device':str(layer.values.device),'key_shape':list(layer.keys.shape),'value_shape':list(layer.values.shape)} for layer in cache.layers];assert all(x['key_device']==x['value_device']==str(self.device) for x in placements)
   e.update(raw_output=raw,full_decoded_output_before_original_postprocess=full,generation_token_ids=tokens,generation_count=len(tokens),truncated=truncated,stop_reason='length' if truncated else 'eos' if tokens and tokens[-1] in self.eos else 'other',visual_feature_archive=str(archive),visual_feature_archive_sha256=sha(archive),projected_visual=record(feature),actual_decoder_visual_prefix=record(prefix),actual_visual_prefix_verified=True,vision_forward_calls=self._vision_calls,text_position_ids=self._positions,prefix_attention_masks=self._masks,final_cache_length=cache_length,cache_device_and_shapes=placements,allocator_after=torch.npu.memory_stats(self.device).get('num_ooms',0));assert e['allocator_after']==e['allocator_before']==0
   log.info('PaliGemma paired NPU sample %s tokens=%s truncated=%s raw=%r',self.calls,len(tokens),truncated,raw[:160]);return raw
  except Exception:e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   e['elapsed_seconds']=time.monotonic()-started;self.details[(prompt.strip(),media_path)]=e;clean=re.sub(r'\n\nProvide only the final answer\.\s*$','',prompt).strip();self.details[(clean,media_path)]=e
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._features=None;self._prefix=None
 def batch_predict(self,batch):return [self.predict(prompt,path) for prompt,path in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

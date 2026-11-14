"""Original Florence caption protocol on native NPU, retaining every original FP16-cast source tensor."""
import os,pathlib,json,hashlib,datetime,logging,time,traceback,contextlib,types,re
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.florence_oct4_original_ops_v2_910b import install_original_Oct4_equations,register_own_original_special_tokens
log=logging.getLogger('universal_eval.model.adapters.florence_original_caption_910b')
TAGS=re.compile(r'</s>|<s>|<pad>|<unk>|<mask>|<fake_token_around_image>|<image>|<[A-Z][A-Z0-9_]*>|<loc_\d+>|<seg>\d+</seg>')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
def clean(s):return ' '.join(TAGS.sub(' ',s or '').split()).strip()
class FlorenceOct4CaptionV3_910BAdapter(IModel):
 def __init__(self,protocol=None,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  self.p=dict(protocol);self.devices=list(npu_devices or [3]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False;self._phase='idle'
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('Florence_Oct4_caption_BF16_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'visual_features').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.gelu_trace=self.evidence/'native_GELU_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Florence2ForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native PyTorch/torch_npu original Oct4 Florence BF16 caption beam3'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoImageProcessor,AutoTokenizer,Florence2ForConditionalGeneration,GenerationConfig
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name=='microsoft/Florence-2-base'
  for rel,digest in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(wp/rel)==digest
  for source,digest in dict(p['runtime_sha256'],**p['historical_sources']).items():assert sha(source)==digest
  for name in ['reference_report','reference_log','dataset_path','original_dataset','mapping_pixel_review']:assert sha(p[name])==p[name+'_sha256']
  assert p['source_dtype']=='F16' and p['runtime_dtype']=='BF16' and p['original_num_beams']==3 and p['max_new_tokens']==1024 and p['extract_answer'] is False and p['original_task_prompt']==''
  torch.set_num_threads(min(8,torch.get_num_threads()));torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for n in ['conv','matmul']:getattr(torch.npu,n).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True,trust_remote_code=False);assert cfg.text_config.encoder_layers==cfg.text_config.decoder_layers==6 and cfg.text_config.d_model==768 and cfg.text_config.vocab_size==51289 and cfg.text_config.scale_embedding is False
  cfg.tie_word_embeddings=True;cfg.text_config.tie_word_embeddings=True
  for c in [cfg,cfg.text_config,cfg.vision_config]:c._attn_implementation='eager'
  with torch.device('meta'):self.model=Florence2ForConditionalGeneration(cfg)
  lm=self.model.model.language_model;lm.encoder.embed_tokens.weight=lm.shared.weight;lm.decoder.embed_tokens.weight=lm.shared.weight
  self.model.register_buffer('original_final_logits_bias',torch.empty((1,51289),device='meta'))
  expected=self.model.state_dict();parameters=set(dict(self.model.named_parameters(remove_duplicate=False)));records=[];seen=set();mapping={r['source']:r for r in p['source_target_mapping']};assert len(mapping)==666
  with safe_open(str(wp/'model.safetensors'),framework='pt',device='cpu') as f:
   assert set(f.keys())==set(mapping)
   for key in f.keys():
    row=mapping[key];v=f.get_tensor(key);assert v.dtype==torch.float16 and fp(v)==row['source_sha256'];cast=v.bfloat16();target=cast.T.contiguous() if row['operation']=='transpose_2d' else cast;name=row['target'];assert name in expected and name not in seen and target.shape==expected[name].shape and torch.isfinite(target).all();assert fp(cast)==row['original_BF16_cast_sha256'] and fp(target)==row['target_BF16_sha256']
    npu=cast.to(self.device).T if row['operation']=='transpose_2d' else target.to(self.device);assert fp(npu)==fp(target);parent,leaf=name.rsplit('.',1) if '.' in name else ('',name);module=self.model.get_submodule(parent) if parent else self.model
    if name in parameters:setattr(module,leaf,torch.nn.Parameter(npu,requires_grad=False))
    else:setattr(module,leaf,npu)
    records.append({'source':key,'target':name,'operation':row['operation'],'source_shape':list(v.shape),'target_shape':list(target.shape),'source_dtype':str(v.dtype),'runtime_dtype':str(target.dtype),'device':str(npu.device),'source_sha256':fp(v),'original_BF16_cast_sha256':fp(cast),'target_BF16_sha256':fp(target),'actual_NPU_sha256':fp(npu),'is_parameter':name in parameters,'nonzero_BF16_underflows':int(((v!=0)&(cast==0)).sum()),'FP16_roundtrip_changed_elements':int((v!=cast.half()).sum())});seen.add(name)
  missing=set(expected)-seen;assert missing=={'model.language_model.encoder.embed_tokens.weight','model.language_model.decoder.embed_tokens.weight','lm_head.weight'}
  lm.encoder.embed_tokens.weight=lm.shared.weight;lm.decoder.embed_tokens.weight=lm.shared.weight;self.model.lm_head.weight=lm.shared.weight;assert self.model.lm_head.weight is lm.shared.weight
  assert len(list(self.model.parameters()))==664 and len(list(self.model.buffers()))==2 and sum(r['nonzero_BF16_underflows'] for r in records)==0
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in list(self.model.parameters())+list(self.model.buffers()));self.model.eval()
  self.original_vision_equations=install_original_Oct4_equations(self.model)
  self.ip=AutoImageProcessor.from_pretrained(wp,local_files_only=True,trust_remote_code=False,backend='pil');self.tok=AutoTokenizer.from_pretrained(wp,local_files_only=True,trust_remote_code=False);assert json.loads(json.dumps(self.ip.to_dict()))==p['image_processor'];self.original_token_registration=register_own_original_special_tokens(wp,self.tok);assert self.original_token_registration==p['original_processor_registered_own_tokens'] and len(self.tok)==51289
  self.task_inputs=self.tok([p['expanded_prompt']],return_tensors='pt',padding=True,return_token_type_ids=False);assert self.task_inputs['input_ids'].tolist()==p['original_task_input_ids'];self.ids=self.task_inputs['input_ids'].to(self.device)
  self.gen,_=self.model._prepare_generation_config(GenerationConfig.from_dict(p['raw_inner_generation_config']),**p['original_generate_kwargs']);assert self.gen.to_dict()==p['generation_config'] and self.gen.num_beams==3 and self.gen.use_cache is True and self.gen.no_repeat_ngram_size==3 and self.gen.forced_bos_token_id is None and self.gen.forced_eos_token_id is None and self.gen.decoder_start_token_id==2
  values=torch.linspace(-8.,8.,20001,dtype=torch.float32).bfloat16();native=torch.nn.functional.gelu(values.to(self.device),approximate='none').cpu();erf=torch.nn.functional.gelu(values,approximate='none');tanh=torch.nn.functional.gelu(values,approximate='tanh');probe=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(probe,input_BF16_bytes=values.view(torch.uint8).numpy(),actual_NPU_BF16_bytes=native.view(torch.uint8).numpy(),CPU_original_erf_BF16_bytes=erf.view(torch.uint8).numpy(),CPU_tanh_diagnostic_BF16_bytes=tanh.view(torch.uint8).numpy());self.probe={'path':str(probe),'sha256':sha(probe),'device':str(self.device),'dtype':'torch.bfloat16','CPU_threads':torch.get_num_threads(),'max_abs_CPU_original_erf':float((native-erf).abs().max()),'max_abs_CPU_tanh_diagnostic':float((native-tanh).abs().max())}
  self._gelu_names=[n for n,m in self.model.named_modules() if type(m).__name__=='GELUActivation'];self._reset()
  def encoder_position(module,args,kw,out):
   if self._phase=='generate':
    n=self._merged_length;assert out.shape==(1,n,768);expected_position=module.weight[torch.arange(n,device=self.device)+module.offset].unsqueeze(0);assert torch.equal(out,expected_position);self._encoder_position=tr(out)
  def decoder_position(module,args,kw,out):
   if self._phase!='generate':return
   ids=args[0] if args else kw['input_ids'];past=int(args[1]) if len(args)>1 else int(kw.get('past_key_values_length',0));positions=kw.get('position_ids');assert ids.shape==(3,1) and positions is not None and positions.tolist()==[past] and past==len(self._positions)
   expect=module.weight[(positions+module.offset).unsqueeze(0)];assert torch.equal(out,expect) and torch.equal(out.expand(3,-1,-1),expect.expand(3,-1,-1));self._positions.append({'decoder_input_ids':ids.cpu().tolist(),'past_length':past,'position_ids':positions.cpu().tolist(),'actual_learned_positions':tr(out),'original_beam_broadcast_positions':tr(out.expand(3,-1,-1)),'device':str(out.device)})
  def encoder_mask(module,args,kw):
   if self._phase!='generate':return
   mask=kw.get('attention_mask') if 'attention_mask' in kw else (args[1] if len(args)>1 else None);assert mask is None or mask.device==self.device and torch.equal(mask,torch.zeros_like(mask));self._encoder_mask=None if mask is None else tr(mask)
  def decoder_mask(module,args,kw):
   if self._phase!='generate':return
   mask=kw.get('attention_mask');assert mask is None or mask.device==self.device and torch.equal(mask,torch.zeros_like(mask));self._masks.append(None if mask is None else tr(mask))
  def cross_input(index):
   def hook(module,args,kw):
    if self._phase!='generate':return
    x=kw.get('key_value_states');assert x is not None and x.shape==(3,self._merged_length,768) and x.device==self.device and x.dtype==torch.bfloat16;self._cross_calls[index]+=1
    if self._cross is None:self._cross=x.detach().clone();assert torch.equal(x,self._encoded.repeat_interleave(3,dim=0))
    else:assert torch.equal(x,self._cross)
   return hook
  lm.encoder.embed_positions.register_forward_hook(encoder_position,with_kwargs=True);lm.encoder.layers[0].self_attn.register_forward_pre_hook(encoder_mask,with_kwargs=True);lm.decoder.embed_positions.register_forward_hook(decoder_position,with_kwargs=True);lm.decoder.layers[0].self_attn.register_forward_pre_hook(decoder_mask,with_kwargs=True)
  for i,layer in enumerate(lm.decoder.layers):layer.encoder_attn.register_forward_pre_hook(cross_input(i),with_kwargs=True)
  def vision_output(module,args,out):
   if self._phase=='generate':self._actual_vision=out.last_hidden_state;self._vision_calls+=1
  def projector_output(module,args,out):
   if self._phase=='generate':self._actual_projected=out;self._projector_calls+=1
  def encoder_input(module,args,kw):
   if self._phase=='generate':
    actual=kw['inputs_embeds'];assert actual.data_ptr()==self._prepared_merged.data_ptr() and torch.equal(actual[:,:577],self._actual_projected);self._actual_merged=actual.detach().clone()
  self.model.model.vision_tower.register_forward_hook(vision_output);self.model.model.multi_modal_projector.register_forward_hook(projector_output);lm.encoder.register_forward_pre_hook(encoder_input,with_kwargs=True)
  def head_output(module,args,out):
   biased=out+self.model.original_final_logits_bias
   if self._phase=='generate':assert biased.device==self.device and biased.dtype==torch.bfloat16 and torch.isfinite(biased[:,-1,:]).all();self._head_calls+=1
   return biased
  self.model.lm_head.register_forward_hook(head_output)
  def count_gelu(name):
   def hook(module,args,out):
    if self._phase=='generate':assert out.device==self.device and out.dtype==torch.bfloat16;self._gelu_counts[name]+=1
   return hook
  for n,m in self.model.named_modules():
   if n in self._gelu_names:m.register_forward_hook(count_gelu(n))
  original_prepare=self.model.prepare_inputs_for_generation
  def prepare(model,input_ids,*args,**kwargs):
   if self._phase=='generate':self._full_beams.append(input_ids.detach().cpu().tolist())
   return original_prepare(input_ids,*args,**kwargs)
  self.model.prepare_inputs_for_generation=types.MethodType(prepare,self.model)
  weights=self.evidence/'weight_loading.json';weights.write_text(json.dumps(records,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
  for source in set(p['runtime_sha256'])|set(p['historical_sources']):
   src=pathlib.Path(source);dest=snapshot/src.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
  (snapshot/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2));(snapshot/'original_report.json').write_bytes(pathlib.Path(p['reference_report']).read_bytes());(snapshot/'original_run.log').write_bytes(pathlib.Path(p['reference_log']).read_bytes())
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'original_Oct4_Florence_caption_BF16_no_temporal_raw_tags','protocol_review':p,'source_dtype':'F16','runtime_dtype':'BF16','weight_loading_evidence':str(weights),'all_expected_parameters_loaded':True,'weight_tensor_count':666,'learned_parameter_count':664,'original_buffer_count':2,'source_BF16_underflow_total':0,'source_BF16_roundtrip_changed_elements':sum(r['FP16_roundtrip_changed_elements'] for r in records),'original_four_embedding_head_aliases_preserved':True,'original_processor_token_registration':self.original_token_registration,'actual_original_auto_BF16':True,'actual_original_projector_omits_temporal_addition':True,'original_final_logits_bias_executed_on_NPU':True,'effective_image_processor':self.ip.to_dict(),'effective_generation_config':self.gen.to_dict(),'native_GELU_probe':self.probe,'native_GELU_execution_trace':str(self.gelu_trace),'per_sample_evidence':str(self.trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ['conv','matmul']},'precision_disclosure':'Own666 FP16 tensors are converted exactly to the actual original adapter auto=BF16. Every sourceF16/castBF16/semantic target/actual NPU tensor SHA is retained. All664 learned tensors and2 buffers, original shared encoder/decoder/head aliases and zero final_logits_bias NPU addition preserved. Current native NPU BF16 DaViT/projector/BART, HF32off; original plain dataset prompt and actual577 visual prefix omit temporal addition according to original adapter patch. Original max2048 YAML is physically guarded at1024 in the original source; original beam3/cacheTrue/repetition1.0/noRepeat3/earlyStopping and forcedBOS-EOS=None retained. Original own1024 declared special tokens registered in memory in original order; no fabricated image token, no vocabulary resize or masked logits. Original full special-tag raw preserved with no TAGS cleanup. Exact conversion rounding is disclosed; not low-bit acceleration, CPU official generation or a claim of bitwise CPU/NPU agreement.','module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
 def _reset(self):
  self._encoded=None;self._cross=None;self._cross_calls=[0]*6;self._positions=[];self._masks=[];self._full_beams=[];self._encoder_position=None;self._encoder_mask=None;self._head_calls=0;self._gelu_counts={n:0 for n in getattr(self,'_gelu_names',[])};self._actual_vision=None;self._actual_projected=None;self._actual_merged=None;self._prepared_merged=None;self._vision_calls=0;self._projector_calls=0
 def _encode(self,pixels):
  visual=self.model.get_image_features(pixel_values=pixels);features=visual.pooler_output;assert features.shape==(1,577,768)
  text=torch.nn.functional.embedding(self.ids,self.model.model.language_model.shared.weight,padding_idx=1);merged=torch.cat([features,text],dim=1);mask=torch.ones(merged.shape[:2],device=self.device);self._merged_length=merged.shape[1];self._prepared_merged=merged
  encoder=self.model.model.language_model.encoder(inputs_embeds=merged,attention_mask=mask,return_dict=True);return visual.last_hidden_state,features,text,merged,mask,encoder
 def predict(self,prompt,media_path=None):
  self.calls+=1;self._reset();self.ids=self.tok([prompt],return_tensors='pt',padding=True,return_token_type_ids=False)['input_ids'].to(self.device);e={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None};started=time.monotonic()
  try:
   media=pathlib.Path(media_path).resolve();assert str(media) in self.p['reference_first200_image_sha256'] and sha(media)==self.p['reference_first200_image_sha256'][str(media)]
   with Image.open(media) as im:image=im.convert('RGB').resize((768,768),Image.Resampling.LANCZOS)
   prepared=self.ip(images=image,return_tensors='pt');assert set(prepared)=={'pixel_values'} and prepared['pixel_values'].shape==(1,3,768,768);pixels=prepared['pixel_values'].bfloat16().to(self.device);e.update(source_image_sha256=sha(media),CPU_original_PIL_pixels=tr(prepared['pixel_values']),NPU_original_BF16_pixels=tr(pixels),original_task_input_ids=self.ids.cpu().tolist(),original_input_prompt_ignored_in_favour_of_fixed_task=False,original_plain_dataset_prompt_used=True,no_reference_passed_to_model=True,allocator_before=torch.npu.memory_stats(self.device).get('num_ooms',0))
   with torch.inference_mode():
    if not self.profile_done:
     self._phase='profile'
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
       *_,mask,encoded=self._encode(pixels);warm=self.model(encoder_outputs=encoded,attention_mask=mask,decoder_input_ids=torch.tensor([[2]],device=self.device),use_cache=True,return_dict=True);assert torch.isfinite(warm.logits).all();torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True;del warm,encoded
    self._reset();self._phase='generate';vision,projected,text,merged,mask,encoded=self._encode(pixels);self._encoded=encoded.last_hidden_state.detach();out=self.model.generate(encoder_outputs=encoded,attention_mask=mask,generation_config=self.gen,return_dict_in_generate=True,output_scores=False)
   seq=out.sequences[0].cpu().tolist();steps=len(self._positions);assert seq[0]==2 and 1<=steps<=1024 and len(self._full_beams)==len(self._masks)==self._head_calls==steps and self._cross_calls==[steps]*6 and self._cross is not None
   eos=seq.index(2,1) if 2 in seq[1:] else len(seq)-1;assert 1<=eos<=1024;raw_before=self.tok.batch_decode([seq],skip_special_tokens=False)[0].strip();raw=raw_before
   for i in range(min(eos,steps)):assert seq[:i+1] in self._full_beams[i]
   cache=out.past_key_values;assert cache.get_seq_length()==steps;cache_info={}
   for name in ['self_attention_cache','cross_attention_cache']:
    cache_info[name]=[]
    for i,layer in enumerate(getattr(cache,name).layers):
     assert layer.keys is not None and layer.values is not None and layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.bfloat16
     cache_info[name].append({'slot':i,'key_shape':list(layer.keys.shape),'value_shape':list(layer.values.shape),'key_device':str(layer.keys.device),'value_device':str(layer.values.device),'dtype':str(layer.keys.dtype)})
   assert self._vision_calls==self._projector_calls==1 and vision.data_ptr()==self._actual_vision.data_ptr() and projected.data_ptr()==self._actual_projected.data_ptr() and torch.equal(merged,self._actual_merged)
   tensors={'vision_spatial':self._actual_vision,'projected_visual':self._actual_projected,'original_unscaled_text_embedding':text,'actual_BART_encoder_inputs':self._actual_merged,'actual_BART_encoder_output':self._encoded,'actual_decoder_cross_attention_input':self._cross};assert all(torch.isfinite(v).all() for v in tensors.values());archive=self.evidence/'visual_features'/('%06d.npz'%(self.calls-1));np.savez_compressed(archive,**{k:v.detach().cpu().contiguous().view(torch.uint8).numpy() for k,v in tensors.items()});e.update(actual_vision_forward_calls=self._vision_calls,actual_projector_forward_calls=self._projector_calls)
   e.update(full_decoded_output_original_special_tags=raw_before,original_TAGS_cleanup_applied=False,raw_output=raw,generation_sequence_ids=seq,selected_generation_token_count=eos,actual_beam_search_steps=steps,beam_search_budget_reached=steps==1024,selected_budget_reached=eos==1024,truncated=eos==1024,budget_reached=eos==1024,forced_eos_at_budget=False,actual_beam_prefixes=self._full_beams,decoder_positions=self._positions,decoder_self_attention_masks=self._masks,encoder_positions=self._encoder_position,encoder_bidirectional_mask=self._encoder_mask,actual_visual_encoder_prefix_verified=True,actual_decoder_cross_attention_equal_to_beam_expanded_encoder=True,visual_feature_archive=str(archive),visual_feature_archive_sha256=sha(archive),visual_tensor_records={k:tr(v) for k,v in tensors.items()},native_GELU_call_counts=dict(self._gelu_counts),finite_actual_next_token_logit_vectors=self._head_calls,original_zero_final_logits_bias_addition_calls=self._head_calls,cache_device_and_shapes=cache_info,final_cache_length=int(cache.get_seq_length()),allocator_after=torch.npu.memory_stats(self.device).get('num_ooms',0));assert e['allocator_before']==e['allocator_after']==0
   with self.gelu_trace.open('a',encoding='utf8') as f:f.write(json.dumps({'attempt':self.calls,'device':str(self.device),'dtype':'torch.bfloat16','module_call_counts':dict(self._gelu_counts),'beam_search_steps':steps,'error':None})+'\n')
   log.info('Florence original NPU sample %s beam_steps=%s selected_tokens=%s raw=%r',self.calls,steps,eos,raw[:180]);return raw
  except Exception:e['error']=traceback.format_exc();log.error(e['error']);raise
  finally:
   self._phase='idle';e['elapsed_seconds']=time.monotonic()-started;self.details[(prompt.strip(),media_path)]=e
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   self._encoded=None;self._cross=None
 def batch_predict(self,batch):return [self.predict(p,m) for p,m in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

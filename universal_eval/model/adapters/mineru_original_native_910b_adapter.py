"""Frozen MinerUStrictBatch4 protocol (actually serial1), own native Qwen2-VL on NPU."""
import os,pathlib,json,hashlib,datetime,time,logging,contextlib,collections,re,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
from scripts.mineru_original_independent_cpu_v2_910b import sha,fp,tr,clean,INSTRUCTION
log=logging.getLogger('universal_eval.model.adapters.mineru_original_native_910b')
def append(path,obj):
 with pathlib.Path(path).open('a',encoding='utf8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
def cpu(t):return t.detach().cpu().contiguous()
def save_arrays(path,values):
 np.savez_compressed(path,**{k:cpu(v).reshape(-1).view(torch.uint8).numpy() for k,v in values.items()})
 return {k:tr(v) for k,v in values.items()}
class MinerUOriginalNative910BAdapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256
  self.devices=list(npu_devices or [5]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
  self.evidence=pathlib.Path(evidence_dir).resolve()/('MinerU_original_serial_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'samples').mkdir()
  self.trace=self.evidence/'sample_execution.jsonl';self.activation_trace=self.evidence/'native_activation_execution.jsonl';self.progress=self.evidence/'generation_progress.jsonl';self.finiteness=self.evidence/'next_token_logits_finiteness.jsonl';self.details={};self.calls=0;self.lineage={};self.profile_done=False;self.prof=None
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Qwen2VLForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers Qwen2-VL BF16 torch_npu original serial protocol'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoProcessor,Qwen2VLForConditionalGeneration
  from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding,Qwen2VLRotaryEmbedding
  from safetensors import safe_open
  p=self.p;root=pathlib.Path(model_path).resolve();assert str(root)==p['weight_path'] and p['model_id']=='opendatalab/MinerU2.5-Pro-2604-1.2B'
  for rel,h in p['resource_sha256'].items():assert pathlib.Path(rel).name==rel and sha(root/rel)==h
  for path,h in dict(p['runtime_sha256'],**p['historical_sources'],**p['evidence_sha256']).items():assert sha(path)==h,path
  assert p['original_max_new_tokens']==512 and p['original_actual_batch_size']==1 and p['original_generate_kwargs']=={'max_new_tokens':512,'do_sample':False,'num_beams':1,'use_cache':True}
  torch.set_num_threads(4);assert torch.npu.is_available();torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
  cfg=AutoConfig.from_pretrained(root,local_files_only=True);assert cfg.text_config.tie_word_embeddings;cfg.tie_word_embeddings=True
  old=torch.get_default_dtype()
  try:
   torch.set_default_dtype(torch.bfloat16)
   with torch.device('meta'):self.model=Qwen2VLForConditionalGeneration(cfg)
  finally:torch.set_default_dtype(old)
  assert {'root':cfg._attn_implementation,'text':cfg.text_config._attn_implementation,'vision':cfg.vision_config._attn_implementation}==p['original_resolved_attention']
  params=dict(self.model.named_parameters());assert len(params)==681;expected={r['source_key']:r for r in p['source_tensor_records']};records=[];seen=set();path=root/'model.safetensors';assert sha(path)==p['weight_file_sha256']
  with safe_open(str(path),framework='pt',device='cpu') as h:
   assert set(h.keys())==set(expected)
   for key in h.keys():
    r=expected[key];name=r['target_key'];x=h.get_tensor(key);assert x.dtype==torch.bfloat16 and tr(x)=={'shape':r['shape'],'dtype':r['source_dtype'],'sha256':r['source_sha256']} and r['original_BF16_sha256']==fp(x)
    assert name in params and name not in seen and list(params[name].shape)==list(x.shape)
    target=x.to(self.device);digest=fp(target);assert digest==r['source_sha256'];parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(target,requires_grad=False));records.append(dict(r,device=str(self.device),actual_NPU_sha256=digest));seen.add(name)
  assert seen==set(params);self.model.lm_head.weight=self.model.model.language_model.embed_tokens.weight
  assert self.model.lm_head.weight is self.model.model.language_model.embed_tokens.weight
  self.model.model.visual.rotary_pos_emb=VisionRotaryEmbedding(cfg.vision_config.embed_dim//cfg.vision_config.num_heads//2).to(self.device)
  self.model.model.language_model.rotary_emb=Qwen2VLRotaryEmbedding(cfg.text_config,device='cpu').to(self.device)
  assert all(t.device==self.device and t.dtype==torch.bfloat16 for t in self.model.parameters()) and all(not t.is_meta and t.device==self.device and t.dtype==torch.float32 for t in self.model.buffers())
  self.model.eval();self.processor=AutoProcessor.from_pretrained(root,local_files_only=True,trust_remote_code=True);self.tokenizer=self.processor.tokenizer
  gen,unused=self.model._prepare_generation_config(None,**p['original_generate_kwargs']);assert not unused and gen.to_dict()==p['resolved_original_generation_config'];self.resolved_gen=gen
  assert gen.eos_token_id==151645 and gen.pad_token_id==151643 and gen.repetition_penalty==1 and not gen.bad_words_ids and gen.forced_eos_token_id is None
  self.rows=json.loads(pathlib.Path(p['source_preflight']).read_text())['samples'];self.lookup={(r['prepared_prompt'],str(pathlib.Path(r['path']).resolve())):r for r in self.rows}
  self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['QuickGELUActivation','GELU','SiLUActivation']};assert collections.Counter(self.activation_modules.values())=={'QuickGELUActivation':32,'GELU':1,'SiLUActivation':24},self.activation_modules
  self._install_observers();weights=self.evidence/'weight_loading.json';weights.write_text(json.dumps(records,indent=2))
  values=torch.linspace(-8,8,20001).bfloat16();probe_arrays={'input_BF16_bytes':values.view(torch.uint8).numpy()}
  for mode in ['none','tanh']:probe_arrays['NPU_'+mode+'_BF16_bytes']=torch.nn.functional.gelu(values.to(self.device),approximate=mode).cpu().view(torch.uint8).numpy();probe_arrays['CPU_'+mode+'_BF16_bytes']=torch.nn.functional.gelu(values,approximate=mode).view(torch.uint8).numpy()
  probe=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(probe,**probe_arrays)
  vocab=self.tokenizer.backend_tokenizer;unmapped=[i for i in range(cfg.text_config.vocab_size) if vocab.id_to_token(i) is None]
  self.lineage={'hardware':'Ascend910B','protocol':'MinerU_original_serial_BF16','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'devices':self.devices,'weight_loading_evidence':str(weights),'weight_tensor_count':681,'all_expected_parameters_loaded':True,'original_tied_output_head_preserved':True,'per_sample_evidence':str(self.trace),'activation_execution_trace':str(self.activation_trace),'generation_progress':str(self.progress),'next_token_logits_finiteness':str(self.finiteness),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':{'path':str(probe),'sha256':sha(probe),'CPU_threads':4,'diagnostic_only':True},'native_activation_modules':self.activation_modules,'original_resolved_attention':p['original_resolved_attention'],'original_actual_batch_size':1,'original_max_new_tokens':512,'original_use_cache':True,'resolved_original_generation_config':gen.to_dict(),'own_generation_json_not_used_by_original_constructor':True,'model_vocab_size':cfg.text_config.vocab_size,'tokenizer_unique_vocab_size':vocab.get_vocab_size(),'original_model_IDs_without_token_string':unmapped,'unmapped_padded_IDs_not_removed_resized_masked_or_filled':True,'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'text_inv_freq':tr(self.model.model.language_model.rotary_emb.inv_freq),'vision_inv_freq':tr(self.model.model.visual.rotary_pos_emb.inv_freq),'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'precision_disclosure':'Own681 stored BF16 learned tensors and the exact original tied head; every source/cast/actual NPU tensor has full SHA. CPU FP32 analytic RoPE constants transferred unchanged. Native SDPA/QuickGELU/GELU/SiLU/cache remain unchanged; all learned vision/merger/language/head layers on NPU BF16, HF32off. Original constructor ignores own generation_config.json; original config-derived greedy512/eos151645/pad151643/cacheTrue is retained. Original class named Batch4 actually invokes serial1. Read-only observers save true vision injection, positions, native mask/SDPA causal semantics, all24 cache layers and complete151936 next-token logits. No native low-bit acceleration, CPU official outputs, substitute resource, option constraints or CPU/NPU bitwise arithmetic claim. Strict explicit MCv2 with the full denominator rejects ordinary articles; original historical Accuracy is retained only as a separate diagnostic.'}
 def _install_observers(self):
  def root_pre(module,args,kw):
   ids=kw.get('input_ids',args[0] if args else None);assert ids is not None and ids.device==self.device;self._active_ids=ids
   cache=kw.get('past_key_values');past=int(cache.get_seq_length()) if cache is not None else 0;mask=kw.get('attention_mask');assert torch.is_tensor(mask)
   self._step={'step':len(self._steps),'input_ids':ids.cpu().tolist(),'past_length_before':past,'use_cache':kw.get('use_cache'),'attention_mask':mask.cpu().tolist()}
   if not self.profile_done and self.prof is None:
    self.prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True);self.prof.start()
   if len(self._steps)%128==0:append(self.progress,{'index':self._index,'decoder_step_entered':len(self._steps)+1,'generated_before_this_step':len(self._steps),'complete':False})
  def root_post(module,args,out):
   logits=out.logits[:,-1,:];assert logits.shape==(1,151936) and logits.dtype==torch.bfloat16 and logits.device==self.device;finite=bool(torch.isfinite(logits).all());assert finite
   self._logits.append(cpu(logits));cache=out.past_key_values;assert cache is not None;self._step['cache_length_after']=int(cache.get_seq_length());self._step['cache_layers']=[]
   for i,layer in enumerate(cache.layers):
    assert layer.keys.device==layer.values.device==self.device and layer.keys.dtype==layer.values.dtype==torch.bfloat16
    self._step['cache_layers'].append({'layer':i,'keys_shape':list(layer.keys.shape),'values_shape':list(layer.values.shape),'device':str(self.device),'dtype':'torch.bfloat16'})
   assert len(self._step['cache_layers'])==24;self._steps.append(self._step);append(self.finiteness,{'index':self._index,'step':len(self._steps)-1,'shape':[1,151936],'device':str(self.device),'dtype':'torch.bfloat16','all_finite':finite})
   if not self.profile_done:
    torch.npu.synchronize(self.device);self.prof.stop();self.prof.export_chrome_trace(str(self.evidence/'npu_trace.json'));self.prof=None;self.profile_done=True
  def visual_pre(module,args,kw):self._arrays['actual_vision_pixels']=cpu(args[0] if args else kw['hidden_states']);self._arrays['actual_grid']=cpu(kw['grid_thw'])
  def encoder_post(module,args,out):self._arrays['actual_last_vision_block']=cpu(out)
  def visual_post(module,args,out):self._features=out.pooler_output;assert self._features.device==self.device;self._arrays['actual_visual_features']=cpu(self._features);self._vision_calls+=1
  def vision_rope_pre(module,args):self._arrays['actual_vision_position_ids']=cpu(args[0])
  def language_pre(module,args,kw):
   x=kw['inputs_embeds'];assert x.device==self.device and x.dtype==torch.bfloat16
   self._step['language_position_ids']=kw['position_ids'].cpu().tolist()
   if not self._steps:
    mask=self._active_ids==151655;actual=x[mask];assert self._features is not None and torch.equal(actual,self._features.reshape(-1,896))
    expected=module.embed_tokens(self._active_ids);assert torch.equal(x[~mask],expected[~mask]);self._arrays['actual_prefill_inputs_embeds']=cpu(x);self._arrays['actual_injected_visual']=cpu(actual)
  def rope(module,args,kw,out):
   positions=kw.get('position_ids',args[1] if len(args)>1 else None);self._step['mrope_position_ids']=positions.cpu().tolist();self._arrays['rope_cos_%04d'%len(self._steps)]=cpu(out[0]);self._arrays['rope_sin_%04d'%len(self._steps)]=cpu(out[1])
  def attention(module,args,kw):
   mask=kw.get('attention_mask');self._step['actual_attention_mask']=None if mask is None else tr(mask);self._step['native_is_causal_attribute']=bool(module.is_causal);self._step['native_sdpa_implicit_causal']=bool(self._active_ids.shape[-1]>1 and mask is None and module.is_causal)
   if mask is not None:self._arrays['causal_mask_%04d'%len(self._steps)]=cpu(mask)
  def activation(name):
   def hook(module,args,out):assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   return hook
  self.model.register_forward_pre_hook(root_pre,with_kwargs=True);self.model.register_forward_hook(root_post);v=self.model.model.visual;v.register_forward_pre_hook(visual_pre,with_kwargs=True);v.register_forward_hook(visual_post);v.blocks[-1].register_forward_hook(encoder_post);v.rotary_pos_emb.register_forward_pre_hook(vision_rope_pre)
  text=self.model.model.language_model;text.register_forward_pre_hook(language_pre,with_kwargs=True);text.rotary_emb.register_forward_hook(rope,with_kwargs=True);text.layers[0].self_attn.register_forward_pre_hook(attention,with_kwargs=True)
  for n,m in self.model.named_modules():
   if n in self.activation_modules:m.register_forward_hook(activation(n))
 def predict(self,prompt,media_path=None):
  start=time.monotonic();media=pathlib.Path(media_path).resolve();record=self.lookup[(re.sub(r'<image\d+>\s*','',prompt).strip(),str(media))];self._index=record['index'];assert self._index==self.calls and sha(media)==record['original_image_sha256'];self.calls+=1
  self._steps=[];self._logits=[];self._arrays={};self._counts={n:0 for n in self.activation_modules};self._features=None;self._vision_calls=0;entry={'index':self._index,'attempt':self.calls,'prompt':prompt,'media_path':media_path,'source_image_sha256':sha(media),'error':None};prefix=self.evidence/'samples'/('%04d'%self._index)
  try:
   with Image.open(media) as image:image=image.convert('RGB')
   text=re.sub(r'<image\d+>\s*','',prompt).strip();chat=self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True);assert chat==record['chat']
   prepared=self.processor(text=[chat],images=[image],return_tensors='pt');assert set(prepared)==set(record['inputs'])
   for k,v in prepared.items():assert tr(v)=={kk:record['inputs'][k][kk] for kk in ['shape','dtype','sha256']},k
   length=prepared['input_ids'].shape[1];inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in prepared.items()}
   with torch.inference_mode(),contextlib.chdir(self.evidence):out=self.model.generate(**inputs,**self.p['original_generate_kwargs'],return_dict_in_generate=True,output_scores=False)
   generated=out.sequences[0,length:].cpu().tolist();assert 0<len(generated)<=512 and len(generated)==len(self._steps) and self._vision_calls==1
   assert out.sequences[0,:length].cpu().tolist()==prepared['input_ids'][0].tolist()
   for name,count in self._counts.items():assert count==(len(generated) if '.language_model.' in name else 1),(name,count)
   logits=torch.stack(self._logits);assert logits.float().argmax(-1)[:,0].tolist()==generated
   raw=self.processor.batch_decode(out.sequences[:,length:],skip_special_tokens=True)[0].strip();visual=prefix.with_name(prefix.name+'_visual.npz');full=prefix.with_name(prefix.name+'_full_logits.npz');vr=save_arrays(visual,self._arrays);lr=save_arrays(full,{'actual_next_token_full_logits':logits})
   sample={'index':self._index,'actual_steps':self._steps,'decoder_steps':len(generated),'CPU_inputs':{k:tr(v) for k,v in prepared.items()},'input_ids':prepared['input_ids'].tolist(),'chat':chat,'visual_archive':str(visual),'visual_archive_sha256':sha(visual),'visual_tensor_records':vr,'full_logits_archive':str(full),'full_logits_archive_sha256':sha(full),'full_logits_tensor_records':lr,'actual_vision_calls':1,'native_activation_call_counts':dict(self._counts),'raw_output':raw,'generation_token_ids':generated,'error':None};evidence=prefix.with_suffix('.json');evidence.write_text(json.dumps(sample,ensure_ascii=False,indent=2))
   entry.update(raw_output=raw,generation_token_ids=generated,generation_count=len(generated),truncated=len(generated)>=512,budget_reached=len(generated)>=512,empty=not bool(raw),invalid=explicit_choice(raw) is None,first_EOS_index=generated.index(151645) if 151645 in generated else None,actual_generation_batch_size=1,actual_visual_injection_verified=True,sample_evidence=str(evidence),sample_evidence_sha256=sha(evidence),visual_archive=str(visual),visual_archive_sha256=sha(visual),full_logits_archive=str(full),full_logits_archive_sha256=sha(full))
   append(self.activation_trace,{'index':self._index,'decoder_steps':len(generated),'dtype':'torch.bfloat16','device':str(self.device),'module_call_counts':dict(self._counts)});append(self.progress,{'index':self._index,'sample_complete':True,'decoder_steps_completed':len(generated),'raw_output_empty':not bool(raw)})
   log.info('MinerU original NPU sample %s/200 tokens=%s truncated=%s raw=%r',self.calls,len(generated),entry['truncated'],raw[:120]);return raw
  except Exception:entry['error']=traceback.format_exc();log.error(entry['error']);raise
  finally:
   entry['elapsed_seconds']=time.monotonic()-start;append(self.trace,entry);self.details[(clean(record['prompt']),media_path)]=entry;self.details[(prompt,media_path)]=entry;self._arrays={};self._logits=[];self._features=None
 def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing MinerU execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

"""Cosmos Reason2 original native BF16/eager true size-grouped generation."""
import os,pathlib,json,datetime,logging,time,traceback,contextlib,collections,functools
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
from scripts.cosmos_reason2_original_cpu_v2_910b import sha,fp,tr
log=logging.getLogger('universal_eval.model.adapters.cosmos_reason2_original_910b')

class CosmosReason2Original910BAdapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha=protocol_sha256;self.devices=list(npu_devices or [0]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.group_calls=0;self.details={};self.lineage={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('CosmosReason2_original_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.activation_trace=self.evidence/'native_activation_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'Qwen3VLForConditionalGeneration'
 @property
 def inference_framework(self):return 'Native Transformers / torch_npu original BF16 eager true size-grouped greedy4096'
 def download(self,*a,**k):raise RuntimeError('Downloads prohibited')
 def load(self,model_path,**kwargs):
  from transformers import AutoConfig,AutoProcessor,Qwen3VLForConditionalGeneration,GenerationConfig
  from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding,Qwen3VLVisionRotaryEmbedding
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert self.model_name=='nvidia/Cosmos-Reason2-2B' and str(wp)==p['weight_path'] and p['original_runtime_dtype']=='BF16' and p['original_max_new_tokens']==4096
  for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
  for src,digest in dict(p['native_runtime_sha256'],**p['historical_sources']).items():assert sha(src)==digest
  assert sha(p['dataset_path'])==p['dataset_sha256'] and p['all200_independent_pixels_template_tokens_original_true_group_padding_bitwise_equal'] and p['original_outer_batch_size']==10
  assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for k in ['conv','matmul']:getattr(torch.npu,k).allow_hf32=False
  cfg=AutoConfig.from_pretrained(wp,local_files_only=True);assert cfg.architectures==['Qwen3VLForConditionalGeneration'] and cfg.tie_word_embeddings and cfg.text_config.tie_word_embeddings and cfg.text_config.vocab_size==151936 and cfg.text_config.num_hidden_layers==28 and cfg.vision_config.depth==24 and not getattr(cfg,'quantization_config',None)
  cfg._attn_implementation='eager';cfg.text_config._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
  with torch.device('meta'):self.model=Qwen3VLForConditionalGeneration(cfg)
  expected={r['source_key']:r for r in p['source_tensor_records']};params=dict(self.model.named_parameters());assert len(params)==len(expected)==625 and set(params)==set(expected);filled=set();loading=[]
  for shard,header in p['source_files'].items():
   q=wp/shard;assert pathlib.Path(shard).name==shard and q.stat().st_size==header['bytes'] and sha(q)==header['sha256']
   with safe_open(str(q),framework='pt',device='cpu') as f:
    for name in f.keys():
     r=expected[name];x=f.get_tensor(name);assert name not in filled and x.dtype==torch.bfloat16 and list(x.shape)==r['shape']==list(params[name].shape) and fp(x)==r['full_sha256'] and torch.isfinite(x).all();y=x.to(self.device);assert fp(y)==r['full_sha256'];parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(y,requires_grad=False));filled.add(name);loading.append(dict(r,target_key=name,source_sha256=r['full_sha256'],original_BF16_cast_sha256=fp(x.bfloat16()),actual_NPU_sha256=fp(y),device=str(self.device),runtime_dtype=str(y.dtype),original_cast_changes=0))
   log.info('Own CosmosReason2 loaded %s/625 source parameters',len(loading))
  assert filled==set(params);self.model.lm_head.weight=self.model.model.language_model.embed_tokens.weight;assert self.model.lm_head.weight.shape==(151936,2048)
  text=self.model.model.language_model;vision=self.model.model.visual;text.rotary_emb=Qwen3VLTextRotaryEmbedding(cfg.text_config,device='cpu').to(self.device);vision.rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(vision.rotary_pos_emb.dim,vision.rotary_pos_emb.theta).to(self.device)
  assert all(v.device==self.device and v.dtype==torch.bfloat16 for v in self.model.parameters()) and all(not v.is_meta and v.device==self.device and v.dtype==torch.float32 for v in self.model.buffers());self.model.eval()
  self.processor=AutoProcessor.from_pretrained(wp,local_files_only=True);self.tokenizer=self.processor.tokenizer;assert self.tokenizer.padding_side=='left' and len(self.tokenizer)==151669 and self.processor.chat_template==(wp/'chat_template.jinja').read_text() and json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['own_image_processor']
  self.model.generation_config=GenerationConfig.from_pretrained(wp,local_files_only=True);self.generation_config,_=self.model._prepare_generation_config(None,max_new_tokens=4096,do_sample=False,temperature=0.0);assert self.generation_config.to_dict()==p['original_resolved_generation_config'];assert self.generation_config.use_cache and self.generation_config.num_beams==1 and not self.generation_config.do_sample and self.generation_config.eos_token_id==[151645,151643]
  self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if n.endswith('act_fn')};assert self.activation_modules==p['native_activation_modules'] and len(self.activation_modules)==56;self._reset();self._install_observers()
  values=torch.linspace(-8,8,20001,dtype=torch.float32).bfloat16();probe={'input':values.view(torch.uint8).numpy()}
  for kind in ['none','tanh']:probe['NPU_'+kind]=torch.nn.functional.gelu(values.to(self.device),approximate=kind).cpu().view(torch.uint8).numpy();probe['CPU_'+kind]=torch.nn.functional.gelu(values,approximate=kind).view(torch.uint8).numpy()
  pq=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(pq,**probe);wq=self.evidence/'weight_loading.json';wq.write_text(json.dumps(loading,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir();(snapshot/'original_protocol.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
  for src in set(p['native_runtime_sha256'])|set(p['historical_sources']):
   path=pathlib.Path(src);dest=snapshot/path.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
  self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'CosmosReason2_original_BF16_eager_true_size_grouped_free_generation','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha,'weight_tensor_count':625,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(wq),'per_sample_evidence':str(self.trace),'per_group_evidence':str(self.evidence/'groups'),'activation_execution_trace':str(self.activation_trace),'activation_modules':self.activation_modules,'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':{'path':str(pq),'sha256':sha(pq),'CPU_threads':4,'only_diagnostic':True},'actual_native_video_component_never_executed':str(type(self.processor.video_processor)),'original_max_new_tokens':4096,'original_outer_batch_size':10,'actual_batching':'Original stable (original_width//64,original_height//64) groups within each original outer10; singleton or real group including10; no serial fallback after group failure','generation_config':self.generation_config.to_dict(),'runtime_dtype':'BF16','original_tied_full151936_head_preserved':True,'own_tokenizer_ID_count':151669,'all267_original_padded_head_IDs_preserved':True,'original_eager_attention_all_text_and_vision':True,'image_processor':self.processor.image_processor.to_dict(),'analytic_F32_buffers':{n:tr(b) for n,b in self.model.named_buffers()},'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'precision_disclosure':'Own625 original BF16 tensors/source-cast-actual NPU SHA; all vision24, three DeepStack mergers plus final merger, text28, tied full151936 head on NPU BF16. CPU-derived original analytic F32 inverse frequencies transferred exactly; HF32 off. Native eager attention, native cacheTrue, original4096 greedy size groups/own template and left padding. Full original output tokens retained including EOS padding and original267 untokenized head IDs. No single-letter constraint, CPU learned-layer offload, replacement resources or native low-bit claim. Native arithmetic/GELU differences retained; CPU/FP32 controls diagnostic only. First actual prefill profiler includes read-only observer checks, not entire200 capture.'}
 def _reset(self):
  self._arrays={};self._steps=[];self._step=None;self._counts={n:0 for n in getattr(self,'activation_modules',{})};self._visual=None;self._vision_calls=0;self._deep=[];self._previous_cache=None;self._last_prefix=None
 def _store(self,name,x):
  assert x.device==self.device;self._arrays[name]=x.detach().cpu().contiguous().clone();return tr(self._arrays[name])
 def _install_observers(self):
  vision=self.model.model.visual;text=self.model.model.language_model;cfg=self.model.config;original_forward=self.model.forward
  @functools.wraps(original_forward)
  def profiled(*args,**kw):
   if self.profile_done:return original_forward(*args,**kw)
   with contextlib.chdir(self.evidence):
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:out=original_forward(*args,**kw);torch.npu.synchronize(self.device)
    prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
   self.profile_done=True;return out
  self.model.forward=profiled;original_prepare=self.model.prepare_inputs_for_generation
  @functools.wraps(original_prepare)
  def prepare(input_ids,*a,**kw):self._last_prefix=input_ids.detach().cpu().clone();return original_prepare(input_ids,*a,**kw)
  self.model.prepare_inputs_for_generation=prepare
  def pre(module,args,kw):
   assert self._step is None and self._last_prefix is not None;self._step={'step':len(self._steps)+1,'full_prefix':tr(self._last_prefix),'input_ids':kw['input_ids'].cpu().tolist(),'position_ids4':kw['position_ids'].cpu().tolist(),'attention_mask2':kw['attention_mask'].cpu().tolist(),'pixels_present':kw.get('pixel_values') is not None,'all28_attention':[]}
  def post(module,args,out):
   step=self._step;n=step['step'];logits=out.logits[:,-1];assert logits.shape==(len(self._items),151936) and logits.dtype==torch.bfloat16 and logits.device==self.device and torch.isfinite(logits).all();self._store('%04d_full_vocab_logits'%n,logits);step['actual_logits_shape']=list(out.logits.shape);cache=out.past_key_values;assert len(cache.layers)==28;step['cache28']=[];now=[]
   for i,c in enumerate(cache.layers):
    k,v=c.keys,c.values;assert k.device==v.device==self.device and k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape and list(k.shape[:2])==[len(self._items),8] and k.shape[-1]==128;prefix=True
    if self._previous_cache is not None:
     pk,pv=self._previous_cache[i];prefix=torch.equal(k[...,:pk.shape[-2],:],pk) and torch.equal(v[...,:pv.shape[-2],:],pv);assert prefix and k.shape[-2]==pk.shape[-2]+1
    step['cache28'].append({'layer':i,'shape':list(k.shape),'device':str(k.device),'dtype':str(k.dtype),'original_prefix_NPU_bitwise_unchanged':prefix});now.append((k.detach(),v.detach()))
   self._previous_cache=now;assert len(step['all28_attention'])==28;self._steps.append(step);self._step=None
   if n==1 or n%128==0:
    with (self.evidence/'generation_progress.jsonl').open('a') as f:f.write(json.dumps({'group':self.group_calls,'batch_size':len(self._items),'completed_decoder_forward':n,'generated_before_this_forward':n-1})+'\n')
  def vispre(module,args,kw):
   from transformers.vision_utils import get_vision_bilinear_indices_and_weights
   self._store('actual_vision_pixels',kw.get('hidden_states',args[0] if args else None));grid=kw.get('grid_thw',args[1] if len(args)>1 else None);indices,weights=get_vision_bilinear_indices_and_weights(grid,num_grid_per_side=module.num_grid_per_side,spatial_merge_size=module.config.spatial_merge_size);self._bilinear_indices=indices;self._bilinear_weights=weights;self._store('actual_visual_bilinear_weights',weights)
  def vispost(module,args,out):
   self._vision_calls+=1;self._visual=out.pooler_output;self._store('actual_encoder',out.last_hidden_state);self._store('actual_final_visual',out.pooler_output);self._deep=list(out.deepstack_features);assert len(self._deep)==3
   for i,v in enumerate(self._deep):self._store('actual_deepstack_%s'%i,v)
  def position(module,args,out):assert torch.equal(args[0],self._bilinear_indices);self._position_gather=out;self._store('actual_visual_bilinear_indices',args[0]);self._store('actual_visual_position_gather',out)
  def patch(module,args,out):self._actual_patch=out;self._store('actual_visual_patch_embeddings',out)
  def visual0(module,args,kw):
   x=kw.get('hidden_states',args[0] if args else None);position=(self._position_gather*self._bilinear_weights[:,:,None]).sum(0).to(self._actual_patch.dtype);assert torch.equal(x,self._actual_patch+position);self._store('actual_visual_block0_input',x)
  def vrot(module,args,out):self._store('actual_vision_position_ids2',args[0]);self._store('actual_vision_rotary_frequencies',out)
  def lang(module,args,kw):
   if self._step['step']!=1:return
   x=kw['inputs_embeds'];mask=self._active_ids==cfg.image_token_id;assert self._visual is not None and torch.equal(x[mask],self._visual.reshape(-1,2048)) and torch.equal(x[~mask],module.embed_tokens(self._active_ids)[~mask]);assert torch.equal(kw['visual_pos_masks'],mask)
   assert len(kw['deepstack_visual_embeds'])==3 and all(torch.equal(a,b) for a,b in zip(kw['deepstack_visual_embeds'],self._deep));self._store('actual_prefill_embeds',x);self._step['original_final_visual_injection_bitwise']=True
  original_deep=text._deepstack_process
  @functools.wraps(original_deep)
  def deep(hidden,mask,visual):
   result=original_deep(hidden,mask,visual);i=sum(k.startswith('actual_deepstack_received_') for k in self._arrays);assert i<3 and torch.equal(visual,self._deep[i]) and torch.equal(result[mask],hidden[mask]+visual) and torch.equal(result[~mask],hidden[~mask]);self._store('actual_deepstack_received_%s'%i,visual);self._step.setdefault('deepstack_actual_additions',[]).append({'index':i,'NPU_unmodified_original_result_verified':True});return result
  text._deepstack_process=deep
  def rope(module,args,kw,out):
   pos=kw.get('position_ids',args[1] if len(args)>1 else None);assert pos.cpu().tolist()==self._step['position_ids4'][1:];self._store('%04d_text_cos'%self._step['step'],out[0]);self._store('%04d_text_sin'%self._step['step'],out[1])
  def attention(i):
   def hook(module,args,kw):
    mask=kw.get('attention_mask');assert kw['position_ids'].cpu().tolist()==self._step['position_ids4'][0];record=None if mask is None else tr(mask)
    if i==0:
     self._step['causal_mask']=record
     if mask is not None:self._store('%04d_causal_mask'%self._step['step'],mask)
    else:assert self._step['causal_mask']==record
    self._step['all28_attention'].append({'layer':i,'mask':record,'device':str(next(module.parameters()).device)})
   return hook
  def activation(name):
   def hook(module,args,out):assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   return hook
  self.model.register_forward_pre_hook(pre,with_kwargs=True);self.model.register_forward_hook(post);vision.register_forward_pre_hook(vispre,with_kwargs=True);vision.register_forward_hook(vispost);vision.pos_embed.register_forward_hook(position);vision.patch_embed.register_forward_hook(patch);vision.blocks[0].register_forward_pre_hook(visual0,with_kwargs=True);vision.rotary_pos_emb.register_forward_hook(vrot);text.register_forward_pre_hook(lang,with_kwargs=True);text.rotary_emb.register_forward_hook(rope,with_kwargs=True)
  for i,layer in enumerate(text.layers):layer.self_attn.register_forward_pre_hook(attention(i),with_kwargs=True)
  for n,m in self.model.named_modules():
   if n in self.activation_modules:m.register_forward_hook(activation(n))
 def _generate_group(self,items):
  self.group_calls+=1;self._reset();self._items=items;home=self.evidence/'groups'/('%04d'%self.group_calls);home.mkdir();rec={'group':self.group_calls,'indices':[x['index'] for x in items],'batch_size':len(items),'error':None};started=time.monotonic()
  try:
   images=[];texts=[]
   for x in items:
    assert sha(x['media_path'])==self.p['input_records'][x['index']]['media_sha256']
    with Image.open(x['media_path']) as im:images.append(im.convert('RGB'))
    texts.append(self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':x['model_prompt']}]}],add_generation_prompt=True,tokenize=False))
   inp=self.processor(text=texts[0] if len(items)==1 else texts,images=[images[0]] if len(items)==1 else [[im] for im in images],return_tensors='pt',padding=True)
   for slot,x in enumerate(items):
    row=self.p['input_records'][x['index']];assert texts[slot]==row['rendered'] and inp['input_ids'][slot][inp['attention_mask'][slot].bool()].tolist()==row['unpad_ids']
   rec.update(CPU_inputs={k:tr(v) for k,v in inp.items()},input_ids=inp['input_ids'].tolist(),attention_mask=inp['attention_mask'].tolist(),image_grid_thw=inp['image_grid_thw'].tolist(),mm_token_type_ids=inp['mm_token_type_ids'].tolist(),rendered=texts)
   device_inputs={k:v.to(self.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device) for k,v in inp.items()};self._active_ids=device_inputs['input_ids'];length=self._active_ids.shape[-1]
   with torch.inference_mode():result=self.model.generate(**device_inputs,max_new_tokens=4096,do_sample=False,temperature=0.0)
   generated=result[:,length:].cpu();tokens=generated.tolist();raw=[s.strip() for s in self.processor.batch_decode(generated,skip_special_tokens=True)];assert self._vision_calls==1 and self._steps[0]['original_final_visual_injection_bitwise'] and len(self._steps[0]['deepstack_actual_additions'])==3 and len(self._steps)==generated.shape[1]<=4096
   for n,c in self._counts.items():assert c==(1 if '.visual.' in n else len(self._steps)),(n,c)
   q=home/'actual_visual_positions_cache_full_vocab.npz';np.savez_compressed(q,**{k:v.reshape(-1).view(torch.uint8).numpy() for k,v in self._arrays.items()});rec.update(steps=self._steps,arrays={k:tr(v) for k,v in self._arrays.items()},archive=str(q),archive_sha256=sha(q),generated_padded_token_ids=tokens,raw_outputs=raw,vision_calls=1,activation_counts=self._counts,elapsed_seconds=time.monotonic()-started);jq=home/'batch_execution.json';jq.write_text(json.dumps(rec,ensure_ascii=False,indent=2))
   from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
   with self.trace.open('a',encoding='utf8') as f:
    for slot,x in enumerate(items):
     eos=next((j for j,t in enumerate(tokens[slot]) if t in [151645,151643]),None);count=len(tokens[slot]) if eos is None else eos+1;entry={'index':x['index'],'attempt':x['index']+1,'prompt':x['prompt'],'actual_model_prompt':x['model_prompt'],'media_path':x['media_path'],'source_image_sha256':self.p['input_records'][x['index']]['media_sha256'],'group':self.group_calls,'group_batch_size':len(items),'group_slot':slot,'group_execution':str(jq),'group_execution_sha256':sha(jq),'generation_token_ids':tokens[slot],'first_EOS_index':eos,'generation_count':count,'raw_output':raw[slot],'error':None,'empty':not raw[slot],'invalid':explicit_choice(raw[slot]) is None,'truncated':count>=4096,'unsupported_original_head_token_IDs':[t for t in tokens[slot] if t>=151669]};self.details[(x['prompt'].strip(),x['media_path'])]=entry;f.write(json.dumps(entry,ensure_ascii=False)+'\n')
   with self.activation_trace.open('a') as f:f.write(json.dumps({'group':self.group_calls,'indices':rec['indices'],'batch_size':len(items),'decoder_forwards':len(self._steps),'module_call_counts':self._counts,'device':str(self.device),'dtype':'torch.bfloat16','error':None})+'\n')
   log.info('CosmosReason2 real NPU group%s batch%s indices%s steps%s',self.group_calls,len(items),rec['indices'],len(self._steps));return raw
  except Exception:rec['error']=traceback.format_exc();(home/'failed_batch_execution.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));log.error(rec['error']);raise
  finally:self._reset()
 def batch_predict(self,batch):
  assert 1<=len(batch)<=10;lookup={(x['model_prompt'],x['media_path']):x for x in self.p['input_records']};groups=collections.OrderedDict();result=[None]*len(batch)
  for slot,(prompt,media) in enumerate(batch):
   key=(prompt,str(pathlib.Path(media).resolve()));assert key in lookup;row=lookup[key]
   with Image.open(media) as im:group=(im.width//64,im.height//64)
   groups.setdefault(group,[]).append(dict(row,media_path=media,slot=slot))
  for items in groups.values():
   outputs=self._generate_group(items)
   for x,out in zip(items,outputs):result[x['slot']]=out
  assert all(isinstance(v,str) for v in result);return result
 def predict(self,prompt,media_path=None):raise RuntimeError('Use original true size-grouped batch API; no serial fallback after a failed group')
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

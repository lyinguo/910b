"""Own FigCodifier BF16/eager InternVL2-InternLM2 original manual greedy NPU run."""
import os,pathlib,json,hashlib,datetime,logging,time,traceback,contextlib,collections,importlib
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.figcodifier_original_runtime_910b import classes,tokenizer
log=logging.getLogger('universal_eval.model.adapters.figcodifier_original_manual_910b')

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':fp(x)}
def pixels(im):
 from torchvision import transforms
 ratios=sorted({(i,j) for n in range(1,13) for i in range(1,n+1) for j in range(1,n+1) if 1<=i*j<=12},key=lambda x:x[0]*x[1]);best=(1,1);dist=float('inf');w,h=im.size
 for r in ratios:
  d=abs(w/h-r[0]/r[1])
  if d<dist:best,dist=r,d
  elif d==dist and w*h>.5*448*448*r[0]*r[1]:best=r
 full=im.resize((best[0]*448,best[1]*448));tiles=[full.crop((x*448,y*448,(x+1)*448,(y+1)*448)) for y in range(best[1]) for x in range(best[0])]
 if len(tiles)>1:tiles.append(im.resize((448,448)))
 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);return torch.stack([transform(t) for t in tiles]),list(best)

class FigCodifierOriginalManual910BAdapter(IModel):
 def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  assert sha(protocol_file)==protocol_sha256;self.p=json.loads(pathlib.Path(protocol_file).read_text());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha256=protocol_sha256;self.devices=list(npu_devices or [11]);assert len(self.devices)==1;self.device=torch.device('npu:'+str(self.devices[0]));self.calls=0;self.details={};self.lineage={};self.profile_done=False
  stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f');self.evidence=pathlib.Path(evidence_dir).resolve()/('FigCodifier_original_manual_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'samples').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.act_trace=self.evidence/'native_activation_execution.jsonl'
 @property
 def model_name(self):return self.p['model_id']
 @property
 def model_type(self):return 'InternVLChatModel'
 @property
 def inference_framework(self):return 'Own InternVL/InternLM2 native torch_npu BF16 eager manual greedy with original tuple cache'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')
 def load(self,model_path,**kwargs):
  from safetensors import safe_open
  p=self.p;wp=pathlib.Path(model_path).resolve();assert str(wp)==p['weight_path'] and self.model_name=='MathLLMs/FigCodifier'
  for n,d in p['resource_sha256'].items():assert pathlib.Path(n).name==n and sha(wp/n)==d
  for n,d in dict(p['runtime_sha256'],**p['evidence_sha256']).items():assert sha(n)==d
  assert sha(p['input_audit'])==p['input_audit_sha256'];audit=json.loads(pathlib.Path(p['input_audit']).read_text());self.records={r['index']:r for r in audit['records']};assert sorted(self.records)==list(range(200));self.by_media={r['media_path']:r for r in self.records.values()}
  assert torch.npu.is_available();torch.set_num_threads(4);torch.npu.set_device(self.device);torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
  for k in ['conv','matmul']:getattr(torch.npu,k).allow_hf32=False
  module=classes(wp);C=importlib.import_module(module.__package__+'.configuration_internvl_chat').InternVLChatConfig;cfg=C.from_dict(json.loads((wp/'config.json').read_text()));assert cfg.llm_config.architectures==['InternLM2ForCausalLM'] and not cfg.llm_config.tie_word_embeddings and cfg.template=='internlm2-chat' and cfg.select_layer==-1
  cfg.vision_config.use_flash_attn=False;cfg.llm_config.attn_implementation='eager';cfg.llm_config._attn_implementation='eager'
  with torch.device('meta'):self.model=module.InternVLChatModel(cfg,use_flash_attn=False)
  params=dict(self.model.named_parameters());expected={r['source_key']:r for r in p['source_tensor_records']};assert len(params)==len(expected)==573 and set(params)==set(expected)
  index=json.loads((wp/'model.safetensors.index.json').read_text())['weight_map'];assert set(index)==set(params);records=[]
  for shard,header in p['source_files'].items():
   assert sha(wp/shard)==header['sha256'] and (wp/shard).stat().st_size==header['bytes']
   with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
    assert set(f.keys())=={k for k,v in index.items() if v==shard}
    for key in f.keys():
     x=f.get_tensor(key);r=expected[key];assert x.dtype==torch.bfloat16 and list(x.shape)==list(params[key].shape)==r['shape'];digest=fp(x);assert digest==r['source_full_sha256']==r['source_BF16_cast_full_sha256'] and r['original_BF16_cast_changes']==0
     y=x.to(self.device).contiguous();assert fp(y)==digest and torch.isfinite(y).all();parent,leaf=key.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(y,requires_grad=False));records.append({'source_key':key,'target_key':key,'source_file':shard,'shape':list(x.shape),'source_dtype':str(x.dtype),'runtime_dtype':str(y.dtype),'device':str(y.device),'source_sha256':digest,'original_BF16_cast_sha256':digest,'actual_NPU_sha256':fp(y),'original_cast_changes':0})
   log.info('Own FigCodifier loaded %s/%s original parameters',len(records),573)
  assert self.model.language_model.output.weight is not self.model.language_model.model.tok_embeddings.weight and fp(self.model.language_model.output.weight)!=fp(self.model.language_model.model.tok_embeddings.weight)
  rope_records=[];cpu_rot=None
  for i,layer in enumerate(self.model.language_model.model.layers):
   rot=layer.attention.rotary_emb
   if cpu_rot is None:
    cpu_rot=type(rot)(rot.dim,max_position_embeddings=rot.max_position_embeddings,base=rot.base,device='cpu',scaling_factor=rot.scaling_factor)
    assert cpu_rot.inv_freq.dtype==cpu_rot.cos_cached.dtype==cpu_rot.sin_cached.dtype==torch.float32
    shared={n:b.to(self.device) for n,b in cpu_rot.named_buffers()}
   for name,b in shared.items():setattr(rot,name,b)
   rot.max_seq_len_cached=cpu_rot.max_seq_len_cached
   assert type(rot).__name__=='InternLM2DynamicNTKScalingRotaryEmbedding' and rot.max_seq_len_cached==32768 and rot.base==1000000 and rot.scaling_factor==2
   rope_records.append({'layer':i,'class':type(rot).__name__,'dim':rot.dim,'base':rot.base,'factor':rot.scaling_factor,'max_seq_len_cached':rot.max_seq_len_cached,'buffers':{n:tr(b) for n,b in rot.named_buffers()}})
  assert all(x.device==self.device and x.dtype==torch.bfloat16 for x in self.model.parameters()) and all(not x.is_meta and x.device==self.device for x in self.model.buffers())
  self.model.eval();self.tok=tokenizer(wp,module.__package__);assert len(self.tok)==92557 and self.tok.bos_token_id==1 and self.tok.convert_tokens_to_ids('<IMG_CONTEXT>')==92546 and self.tok.convert_tokens_to_ids('<|im_end|>')==92542 and self.model.num_image_token==256
  self.model.img_context_token_id=92546;self.module=module;self._reset();self._observers();loading=self.evidence/'weight_loading.json';loading.write_text(json.dumps(records,indent=2));self.probe=self._probe()
  self.lineage={'protocol':'FigCodifier_original_BF16_eager_manual_serial_generation','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha256,'devices':self.devices,'hardware':'Ascend910B','runtime_dtype':'BF16','weight_tensor_count':573,'all_expected_parameters_loaded':True,'original_untied_output_head_preserved':True,'own_model_architecture':type(self.model).__name__,'own_LM_architecture':type(self.model.language_model).__name__,'own_language_attention':type(self.model.language_model.model.layers[0].attention).__name__,'original_manual_stop_EOS_ID':92542,'source_config_EOS2_not_manual_stop':True,'original_manual_max_new_tokens':2048,'original_manual_explicit_use_cache':True,'actual_language_sample_batch_size':1,'outer_algorithm_metadata_batch_size':10,'own_system_message':self.model.system_message,'own_vocab_size':92557,'own_BOS_ID':1,'image_tokens_per_tile':256,'weight_loading_evidence':str(loading),'per_sample_evidence':str(self.trace),'activation_execution_trace':str(self.act_trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'native_GELU_probe':self.probe,'activation_modules':self.activation_modules,'original_32_analytic_rotary_buffers':rope_records,'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'local_compatibility':['Original checkpoint Python is read-only; isolated import retains own InternVision/InternLM2 learned code','Original eager/naive attention and original tuple native KV cache, no HF generate fallback substitution','DropPath is exact identity in eval; training rejected. CPU schedule permits meta construction','Original FP32 analytic rotary frequencies/cos/sin restored on CPU and transferred unchanged; identical nonlearned tables shared by32 layers','Own deprecated tokenizer is loaded directly from exact local source, no fake or borrowed vocabulary'],'precision_disclosure':'Own573 source BF16->original runtime BF16, full source/cast/actual NPU SHA; own untied head preserved; all learned InternVision/projector/InternLM2/output on NPU11 BF16, HF32off. Own eager attention, FP32 language softmax and own output float conversion, original2048 serial manual greedy, cacheTrue, stop92542. Original FP32 analytic RoPE; no CPU learned-layer offload, precision reduction, quantized replacement or low-bit acceleration claim. Actual CANN arithmetic differences retained; CPU/FP32 controls diagnostic only.'}
 def _probe(self):
  x=torch.linspace(-8,8,20001);arrays={};out={'path':str(self.evidence/'native_GELU_semantics.npz'),'device':str(self.device),'CPU_threads':torch.get_num_threads(),'modes':{}}
  for dtype,label in [(torch.float32,'FP32'),(torch.bfloat16,'BF16')]:
   z=x.to(dtype);arrays[label+'_input_bytes']=z.view(torch.uint8).numpy();out['modes'][label]={}
   for mode in ['none','tanh']:
    c=torch.nn.functional.gelu(z,approximate=mode);n=torch.nn.functional.gelu(z.to(self.device),approximate=mode).cpu();arrays[label+'_CPU_'+mode+'_bytes']=c.view(torch.uint8).numpy();arrays[label+'_NPU_'+mode+'_bytes']=n.view(torch.uint8).numpy();out['modes'][label][mode]={'max_abs_CPU_same_declared':float((c-n).abs().max()),'finite':bool(torch.isfinite(n).all())}
  np.savez_compressed(out['path'],**arrays);out['sha256']=sha(out['path']);return out
 def _reset(self):
  self._arrays={};self._steps=[];self._step=None;self._counts={n:0 for n in getattr(self,'activation_modules',{})};self._encoder=None;self._features=None;self._pixels=None;self._mask_object=None;self._positions_object=None;self._oldcache={};self._vision_calls=0;self._connector_calls=0;self._bridge=False;self._actual_rope_pair=None
 def _store(self,name,tensor):
  assert tensor.device==self.device;self._arrays[name]=tensor.detach().cpu().contiguous().clone();return tr(self._arrays[name])
 def _observers(self):
  self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if type(m).__name__ in ['GELUActivation','SiLUActivation','GELU']};assert len(self.activation_modules)==57
  for name in self.activation_modules:
   def h(m,args,out,name=name):
    assert out.device==self.device and out.dtype==torch.bfloat16;self._counts[name]+=1
   self.model.get_submodule(name).register_forward_hook(h)
  def vis_pre(m,args,kw):self._store('actual_vision_pixels',kw['pixel_values']);assert torch.equal(kw['pixel_values'],self._pixels)
  def vis_post(m,args,out):self._vision_calls+=1;self._encoder=out.last_hidden_state;self._store('actual_encoder',self._encoder)
  def con_pre(m,args):self._store('actual_connector_input',args[0])
  def con_post(m,args,out):self._connector_calls+=1;self._features=out;self._store('actual_connector',out)
  self.model.vision_model.register_forward_pre_hook(vis_pre,with_kwargs=True);self.model.vision_model.register_forward_hook(vis_post);self.model.mlp1.register_forward_pre_hook(con_pre);self.model.mlp1.register_forward_hook(con_post)
  def lm_pre(m,args,kw):
   assert kw['use_cache'] is True and kw['return_dict'] is True and self._step is None
   n=len(self._steps)+1;self._step={'step':n,'attention_mask_2D':kw['attention_mask'].cpu().tolist(),'layers':[],'cache':[]};self._mask_object=None;self._positions_object=None
   if n==1:
    emb=kw['inputs_embeds'];self._store('actual_prefill_inputs_embeds',emb);self._store('actual_injected_visual',emb[self._input_ids==92546]);assert torch.equal(emb[self._input_ids==92546],self._features.reshape(-1,4096));self._bridge=True
   else:assert kw['input_ids'].shape==(1,1) and len(kw['past_key_values'])==32
  def lm_post(m,args,out):
   st=self._step;assert st is not None and len(st['layers'])==32 and len(st['cache'])==32 and len(out.past_key_values)==32
   logits=out.logits[:,-1,:];assert logits.shape==(1,92557) and logits.dtype==torch.float32 and torch.isfinite(logits).all();st['full_next_token_logits']=self._store('%04d_logits'%st['step'],logits);st['next_token_ID']=int(logits.argmax(-1).item());self._steps.append(st);self._step=None
  self.model.language_model.register_forward_pre_hook(lm_pre,with_kwargs=True);self.model.language_model.register_forward_hook(lm_post)
  for i,layer in enumerate(self.model.language_model.model.layers):
   def rope_post(m,args,out,i=i):
    st=self._step;pos=self._positions_object.reshape(-1);pair=(out[0][pos],out[1][pos])
    if i==0:
     self._actual_rope_pair=pair;st['actual_cos']=self._store('%04d_cos'%st['step'],pair[0]);st['actual_sin']=self._store('%04d_sin'%st['step'],pair[1])
    else:assert all(torch.equal(a,b) for a,b in zip(pair,self._actual_rope_pair))
   layer.attention.rotary_emb.register_forward_hook(rope_post)
   def pre(m,args,kw,i=i):
    st=self._step;mask=kw['attention_mask'];pos=kw['position_ids'];past=kw['past_key_value'];assert kw['use_cache'] is True and not kw['output_attentions'];assert args[0].device==self.device and mask.device==pos.device==self.device
    if i==0:
     self._mask_object=mask;self._positions_object=pos;st['actual_causal_mask']=tr(mask);st['actual_position_ids']=pos.cpu().tolist()
    else:assert mask is self._mask_object and pos is self._positions_object
    st['layers'].append({'layer':i,'device':str(args[0].device),'dtype':str(args[0].dtype),'position_ids_same_actual_object':True,'mask_same_actual_object':True,'past_length':0 if past is None else past[0].shape[2]});self._oldcache[i]=past
   def post(m,args,out,i=i):
    k,v=out[1];assert k.device==v.device==self.device and k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape and list(k.shape[:2])==[1,8] and k.shape[-1]==128
    past=self._oldcache.pop(i);prefix=True if past is None else bool(torch.equal(k[:,:,:past[0].shape[2]],past[0]) and torch.equal(v[:,:,:past[1].shape[2]],past[1]));assert prefix
    self._step['cache'].append({'layer':i,'key_shape':list(k.shape),'value_shape':list(v.shape),'device':str(k.device),'dtype':str(k.dtype),'original_tuple_type':isinstance(out[1],tuple),'actual_NPU_full_prior_key_value_prefix_equal':prefix})
   layer.register_forward_pre_hook(pre,with_kwargs=True);layer.register_forward_hook(post)
 def _query(self,prompt,n):return ('<|im_start|>system\n'+self.model.system_message+'<|im_end|><|im_start|>user\n<image>\n'+prompt+'<|im_end|><|im_start|>assistant\n').replace('<image>','<img>'+'<IMG_CONTEXT>'*(256*n)+'</img>',1)
 def predict(self,prompt,media_path=None):
  self.calls+=1;started=time.monotonic();self._reset();r=self.by_media[str(pathlib.Path(media_path).resolve())];entry={'attempt':self.calls,'index':r['index'],'prompt':r['prompt'],'actual_original_model_prompt':prompt,'media_path':str(pathlib.Path(media_path).resolve()),'error':None}
  try:
   assert prompt==r['actual_original_model_prompt'] and sha(media_path)==r['source_image_sha256']
   with Image.open(media_path) as im:x,grid=pixels(im.convert('RGB'))
   assert tr(x)==r['original_pixel_values'] and grid==r['original_grid'];query=self._query(prompt,len(x));assert query==r['original_packed_chat_text'];inputs=self.tok(query,return_tensors='pt');assert inputs['input_ids'].tolist()==[r['original_input_ids']] and inputs['attention_mask'].tolist()==r['original_attention_mask']
   ids=inputs['input_ids'].to(self.device);mask=inputs['attention_mask'].to(self.device);self._input_ids=ids;self._pixels=x.to(device=self.device,dtype=torch.bfloat16)
   def first():
    vit=self.model.extract_feature(self._pixels);emb=self.model.language_model.get_input_embeddings()(ids);flat=emb.reshape(-1,4096);flat[(ids.reshape(-1)==92546)]=vit.reshape(-1,4096).to(emb.device);emb=flat.reshape(1,ids.shape[1],4096)
    return self.model.language_model(inputs_embeds=emb,attention_mask=mask,use_cache=True,return_dict=True)
   generated=[]
   with torch.inference_mode():
    if not self.profile_done:
     with contextlib.chdir(self.evidence):
      with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:out=first();torch.npu.synchronize(self.device)
      prof.export_chrome_trace(str(self.evidence/'npu_trace.json'))
     self.profile_done=True
    else:out=first()
    for step in range(2048):
     past=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1).unsqueeze(-1);generated.append(next_id);assert int(next_id.item())==self._steps[-1]['next_token_ID']
     if int(next_id.item())==92542:break
     if step+1==2048:break
     mask=torch.cat([mask,torch.ones_like(mask[:,:1])],1);out=self.model.language_model(input_ids=next_id,past_key_values=past,attention_mask=mask,use_cache=True,return_dict=True)
   tokens=torch.cat(generated,1)[0].cpu().tolist();decoded=self.tok.batch_decode([tokens],skip_special_tokens=True)[0];raw=decoded.split('<|im_end|>')[0].strip();assert self._bridge and self._vision_calls==self._connector_calls==1 and len(tokens)==len(self._steps)
   archive=self.evidence/'samples'/('%03d.npz'%r['index']);np.savez_compressed(archive,**{k:v.view(torch.uint8).numpy() for k,v in self._arrays.items()});records={k:tr(v) for k,v in self._arrays.items()};detail={'index':r['index'],'prompt':r['prompt'],'actual_original_model_prompt':prompt,'rendered_text':query,'CPU_pixel_values':tr(x),'input_ids':ids.cpu().tolist(),'original_grid':grid,'actual_tile_batch_size':len(x),'actual_language_sample_batch_size':1,'native_activation_call_counts':self._counts,'actual_vision_calls':self._vision_calls,'actual_connector_calls':self._connector_calls,'actual_visual_injection_verified':self._bridge,'steps':self._steps,'generation_token_ids':tokens,'full_decoded_output_before_original_strip':decoded,'raw_output':raw,'truncated':len(tokens)>=2048,'error':None,'archive':str(archive),'archive_sha256':sha(archive),'tensor_records':records};q=self.evidence/'samples'/('%03d.json'%r['index']);q.write_text(json.dumps(detail,ensure_ascii=False,indent=2));entry.update(source_image_sha256=sha(media_path),sample_execution=str(q),sample_execution_sha256=sha(q),generation_token_ids=tokens,generation_count=len(tokens),raw_output=raw,truncated=len(tokens)>=2048,empty=not raw,actual_visual_injection_verified=True)
   log.info('FigCodifier NPU sample %d index=%d tiles=%d generated=%d truncated=%s raw=%r',self.calls,r['index'],len(x),len(tokens),entry['truncated'],raw[:140]);return raw
  except Exception:entry['error']=traceback.format_exc();log.error(entry['error']);raise
  finally:
   entry['elapsed_seconds']=time.monotonic()-started
   self.details[(r['prompt'],media_path)]=entry
   with self.trace.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
   with self.act_trace.open('a') as f:f.write(json.dumps({'index':r['index'],'device':str(self.device),'dtype':'torch.bfloat16','language_sample_batch_size':1,'tile_batch_size':len(self._pixels) if self._pixels is not None else None,'decoder_forwards':len(self._steps),'module_call_counts':self._counts,'error':entry['error']})+'\n')
   self._arrays={};self._features=None;self._encoder=None;self._pixels=None
 def batch_predict(self,batch):
  groups=collections.OrderedDict()
  for i,(prompt,path) in enumerate(batch):
   with Image.open(path) as im:key=(im.width//64,im.height//64)
   groups.setdefault(key,[]).append((i,prompt,path))
  result=[None]*len(batch)
  for group in groups.values():
   for i,prompt,path in group:result[i]=self.predict(prompt,path)
  return result
 def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'No actual recorded execution'})
 def get_execution_lineage(self):return self.lineage
 def unload(self):self.model=None;torch.npu.empty_cache()

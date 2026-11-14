"""Independent source/input/schedule/KV/mask/full-head/caption delivery audit.

No model, IO or scheduler under test is imported. CPU work verifies evidence
and diagnostics, and never supplies an official language-model answer.
"""
import os,sys,json,pathlib,hashlib,re,bisect,collections,math
from decimal import Decimal
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch,numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from PIL import Image,ImageOps
from torchvision.transforms.v2 import functional as TV
from torchvision.transforms import InterpolationMode
R=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R))
from universal_eval.metric.rouge_910b_porter import PorterStemmer
torch.set_num_threads(4);external={}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def checked(path,digest=None):
 path=pathlib.Path(path).resolve();assert path.is_relative_to(R/'workspace/diagnostics') or path.is_relative_to(R/'scripts') or path.is_relative_to(R/'universal_eval')
 h=sha(path)
 if digest is not None:assert h==digest,(str(path),h,digest)
 external[str(path)]=h;return path
stem=PorterStemmer()
def words(s):return [stem.stem(w) if len(w)>3 else w for w in re.findall('[a-z0-9]+',str(s).lower())]
def rouge(a,b):
 a=words(a);b=words(' '.join(b) if isinstance(b,list) else b)
 if not a or not b:return 0.
 pos={}
 for i,w in enumerate(b):pos.setdefault(w,[]).append(i)
 tail=[]
 for w in a:
  for i in reversed(pos.get(w,[])):
   j=bisect.bisect_left(tail,i)
   if j==len(tail):tail.append(i)
   else:tail[j]=i
 return 2*len(tail)/(len(a)+len(b))
patterns=[r'<answer>\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*</answer>',r'\\boxed\{([^}]+)\}',r'(?:the\s+)?answer\s*(?:is|:)\s*([^\n.]+)',r'correct\s+option\s+is\s*\**(\w+)\**',r'correct\s+answer\s*(?:is|:)\s*([^\n.]+)',r'(?:the\s+)?(?:number|result|value|amount|length)\s+is\s+([^\n.]+)',r'(?:therefore|so|hence|thus|conclusion)[,.]?\s+.+?\s+is\s+([^\n.]+?)(?:\.|\n|$)',r'\*\*([^*]+?)\*\*',r'=\s*(\d+(?:\.\d+)?)\s*$',r'(?:^|\n)\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*$',r'([A-Ea-e])\s*$']
flags=[re.I,0,re.I,re.I,re.I,re.I,re.I,0,re.M,re.M,re.M]
def caption_extract(raw):
 if not raw or not isinstance(raw,str):return raw
 s=raw.strip().replace('\u0120',' ').replace('\u010a',' ')
 for pattern,flag in zip(patterns,flags):
  matches=re.findall(pattern,s,flag)
  if matches:
   answer=matches[-1].strip().rstrip('*').strip()
   if answer:return answer
 return raw
def decode(rust,prompt,ids,finish):
 if finish=='stop':assert ids[-1]==1;ids=ids[:-1]
 stream=DecodeStream(ids=prompt,skip_special_tokens=True);text=''
 for token in ids:
  try:piece=stream.step(rust,int(token))
  except (TypeError,OverflowError):piece=None
  except Exception as exc:
   if not str(exc).startswith('Invalid prefix encountered'):raise
   stream=DecodeStream(skip_special_tokens=True);piece=stream.step(rust,int(token))
  text+=piece or ''
 return text
def jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in (3,10,200)
names={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {q.name for q in folder.iterdir()}==names
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];p=json.loads(checked(kw['protocol_file'],kw['protocol_sha256']).read_text());w=pathlib.Path(p['weight_path']);vd='npu:'+str(kw['npu_devices'][0])
line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution']
assert cfg['model']['name']==p['model_id']=='google/paligemma2-3b-pt-896' and cfg['model']['model_path']==str(w)
assert folder.parent.parent.name=='google_paligemma2-3b-pt-896' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1)
assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
assert cfg['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs={'extract_answer':True})
assert cfg['metrics']['override']==['ROUGE-L'] and cfg['metrics']['include_failed_predictions']
assert not cfg['model']['download'] and not cfg['data']['download'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template']
assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0
assert line['runtime_dtype']=='BF16' and line['weight_tensor_count']==727 and line['physical_packed_parameter_count']==541
assert line['original_max_new_tokens']==1024 and line['original_outer_request_batch_size']==10 and line['original_max_model_len']==8192
assert line['original_primary_eos_token_id']==1 and line['original_additional_stop_token_ids']==[] and line['original_min_tokens']==0
assert line['original_full_vocabulary']==257216 and line['original_full_head_dtype']=='BF16' and line['original_MM_prefix_inclusive_range']==[0,4095]
for name,digest in p['required_code_sha256'].items():assert sha(R/name)==digest,name
for path,digest in p['static_evidence_SHA256'].items():checked(path,digest)
loaded=json.loads(checked(line['weight_loading_evidence']).read_text());physical=json.loads(checked(line['physical_packed_weights']).read_text());assert len(loaded)==727 and len(physical)==541
by={r['source']:r for r in loaded};phys={r['name']:r for r in physical};raws={};mapping=collections.defaultdict(list)
for name,rec in p['source_files'].items():assert sha(w/name)==rec['SHA256']
for name in sorted(n for n in p['source_files'] if n.endswith('.safetensors')):
 with safe_open(str(w/name),framework='pt',device='cpu') as f:
  for key in f.keys():
   value=f.get_tensor(key);e=p['source_tensors'][key];r=by[key];h=fp(value)
   assert value.dtype==torch.bfloat16 and list(value.shape)==e['shape']==r['source_shape'] and h==e['SHA256']==r['source_SHA256']==r['cast_SHA256']==r['actual_NPU_SHA256'] and r['device']==vd
   target=key;offset=None
   for word,n in [('q_proj',0),('k_proj',1),('v_proj',2)]:
    if '.'+word+'.' in key:
     target=key.replace('.'+word+'.','.qkv_proj.');q=p['source_tensors'][key.replace('.'+word+'.','.q_proj.')]['shape'][0];k=p['source_tensors'][key.replace('.'+word+'.','.k_proj.')]['shape'][0];offset=[0,q,q+k][n];break
   for word,n in [('gate_proj',0),('up_proj',1)]:
    if '.'+word+'.' in key:target=key.replace('.'+word+'.','.gate_up_proj.');offset=n*value.shape[0]
   assert r['packed_parameter']==target and r['packed_slice']==(None if offset is None else [offset,offset+value.shape[0]])
   mapping[target].append((0 if offset is None else offset,key));raws[key]=value
assert len(raws)==727 and set(mapping)==set(phys)
for name,parts in mapping.items():
 parts.sort();end=0
 for offset,key in parts:assert offset==end;end+=raws[key].shape[0]
 tensor=raws[parts[0][1]] if len(parts)==1 else torch.cat([raws[key] for _,key in parts]);rec=phys[name]
 assert rec['shape']==list(tensor.shape) and rec['dtype']=='torch.bfloat16' and rec['device']==vd and rec['SHA256']==fp(tensor)
embedding=raws['language_model.model.embed_tokens.weight'];assert embedding.shape==(257216,2304) and not any(k.endswith('lm_head.weight') for k in raws)
source_control={k:{'BF16':fp(v),'FP32':fp(v.float())} for k,v in raws.items() if not k.startswith('language_model.')};assert len(source_control)==439
del raws
dataset=jsonl(pathlib.Path(p['dataset']));assert sha(p['dataset'])==p['dataset_SHA256'] and len(dataset)==1459
inputs=jsonl(folder/'sample_inputs.jsonl');outputs=jsonl(folder/'sample_outputs.jsonl');details=report['detailed_results']
for rows in [inputs,outputs,details]:assert len(rows)==count and [r['index'] for r in rows]==list(range(count))
entries=jsonl(checked(line['per_sample_evidence']));assert len(entries)==count and [r['index'] for r in entries]==list(range(count));by_index={r['index']:r for r in entries}
groups=json.loads(checked(line['completed_groups_manifest']).read_text());assert [g['request_count'] for g in groups]==[min(10,count-i) for i in range(0,count,10)]
rust=Tokenizer.from_file(str(w/'tokenizer.json'));all_steps=0;argmax_rows=0;KV_count=0;encoder_waves=[]
for gi,manifest in enumerate(groups):
 group=json.loads(checked(manifest['path'],manifest['SHA256']).read_text());steps=jsonl(checked(manifest['trace'],manifest['trace_SHA256']));assert len(steps)==group['actual_model_forward_count']==manifest['actual_forward_count']
 requests=group['requests'];assert len(requests)==manifest['request_count'];states={};expected_pixels={};logical={};logical_buffers={};logical_lengths={};pixels_seen=set()
 for local,row in enumerate(requests):
  i=gi*10+local;ex=p['input_records'][i];assert row['index']==i and row['request_id']==str(local) and row['prompt']==ex['model_prompt']
  assert sha(row['media_path'])==row['source_image_SHA256']==ex['media_SHA256']
  rendered='<image>'*4096+'<bos>'+'Describe this image in detail.\n\nProvide only the final answer.'+'\n';ids=rust.encode(rendered,add_special_tokens=False).ids
  assert rendered==row['rendered'] and hashlib.sha256(rendered.encode()).hexdigest()==ex['rendered_SHA256'] and ids==ex['ids']==row['input_ids']
  assert len(ids)==4111 and ids[:4096]==[257152]*4096 and ids[4096]==2 and ids.count(2)==1 and row['computed_prefix_hit_tokens']==0
  with Image.open(row['media_path']) as source:
   image=ImageOps.exif_transpose(source)
   if image.mode in ('RGBA','LA','PA') or 'transparency' in image.info:
    rgba=image.convert('RGBA');image=Image.new('RGB',rgba.size,(255,255,255));image.paste(rgba,mask=rgba.getchannel('A'))
   elif image.mode!='RGB':image=image.convert('RGB')
   rgb=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)
  resized=TV.resize(rgb,[896,896],interpolation=InterpolationMode.BICUBIC,antialias=True);pixels=((resized.float()-127.5)/127.5).bfloat16()
  assert fp(pixels)==row['CPU_pixel_SHA256']==ex['pixels_SHA256'];expected_pixels[str(local)]=pixels
  states[str(local)]=dict(past=0,tokens=[],ended=False,finish=None,finished=False,held=False,forward_count=0,row=row)
 with np.load(checked(manifest['archive'],manifest['archive_SHA256']),allow_pickle=False) as arrays:
  visited=set()
  def tensor(rec):
   key=rec['array'];visited.add(key);a=arrays[key];assert rec['device']==vd and a.dtype==np.uint8 and a.ndim==1 and hashlib.sha256(a.tobytes()).hexdigest()==rec['SHA256']
   dtype={'torch.bfloat16':torch.bfloat16,'torch.bool':torch.bool,'torch.int64':torch.int64,'torch.float32':torch.float32}[rec['dtype']]
   return torch.from_numpy(a.copy()).view(dtype).reshape(rec['shape'])
  visual={};wave_by_step={};waves=iter(group['vision_waves']);waiting=list(states);running=[];used=0;previous=set()
  for n,step in enumerate(steps):
   assert step['ordinal']==n;budget=8192;plan=[];enc=[]
   for rid in running:
    st=states[rid];q=min(4111-st['past'] if st['past']<4111 else 1,budget)
    if q:plan.append((rid,st['past'],q));budget-=q
   while waiting and budget and used+4096<=8192:
    rid=waiting.pop(0);states[rid]['held']=True;used+=4096;q=min(4111,budget);plan.append((rid,0,q));budget-=q;enc.append(rid);running.append(rid)
   plan.sort(key=lambda item:(item[2]!=1,item[2]));assert [x[0] for x in plan]==step['request_ids'] and [x[2] for x in plan]==step['query_lengths'] and enc==step['encoder_requests']
   if enc:
    wave=next(waves);assert wave['request_ids']==enc;encoder_waves.append(len(enc));tensors=wave['tensors'];actual=tensor(tensors['vision_actual_pixels']);assert torch.equal(actual,torch.stack([expected_pixels[r] for r in enc]))
    encoder=tensor(tensors['vision_encoder_output']);projector=tensor(tensors['visual_projector_output']);normalized=tensor(tensors['original_Pali_normalized_visual_embeddings'])
    assert encoder.shape==(len(enc),4096,1152) and projector.shape==normalized.shape==(len(enc),4096,2304)
    assert torch.equal(normalized,projector*(2304**-.5));visual.update({r:v for r,v in zip(enc,normalized)})
    assert len(wave['activations'])==27
    for layer,act in enumerate(wave['activations']):assert act['module']=='vision_activation_%02d'%layer and act['shape']==[len(enc),4096,4304] and act['dtype']=='torch.bfloat16' and act['device']==vd and len(act['SHA256'])==64
   for slot,(rid,past,q) in enumerate(plan):
    st=states[rid];ids=st['row']['input_ids'][past:past+q] if past<4111 else [st['tokens'][-1]];assert step['input_ids'][slot]==ids and step['positions'][slot]==list(range(past,past+q))
   assert len(step['activations'])==26 and len(step['KV_deltas'])==52*len(plan) and len(step['attention'])==26*len(plan)
   for layer,act in enumerate(step['activations']):assert act['module']=='language_activation_%02d'%layer and act['shape']==[sum(q for _,_,q in plan),9216] and act['dtype']=='torch.bfloat16' and act['device']==vd
   if n==0:
    embeds=tensor(step['input_embeddings']);expected=[]
    for rid,past,q in plan:
     ids=torch.tensor(step['input_ids'][len(expected)]);val=embedding[ids]*torch.tensor(48.,dtype=torch.bfloat16);image=ids==257152;val[image]=visual[rid][torch.arange(past,past+q)[image]];expected.append(val)
    assert torch.equal(embeds,torch.cat(expected))
   for delta in step['KV_deltas']:
    value=tensor(delta);layer=int(delta['name'].rsplit('_',1)[1]);rid=delta['request_id'];kind='k' if '_key_' in delta['name'] else 'v';slot=step['request_ids'].index(rid);q=step['query_lengths'][slot]
    assert value.shape==(q,4,256) and delta['positions']==step['positions'][slot] and delta['prefix_token_count']==0 and delta['prefix_source_request_id'] is None
    key=(layer,rid,kind);past=logical_lengths.get(key,0)
    assert delta['positions'][0]==past and past+q<=5135
    if key not in logical_buffers:logical_buffers[key]=torch.empty((5135,4,256),dtype=torch.bfloat16)
    logical_buffers[key][past:past+q].copy_(value);logical_lengths[key]=past+q;logical[key]=logical_buffers[key][:past+q];KV_count+=1
   for attn in step['attention']:
    mask=tensor(attn);rid=attn['request_id'];layer=attn['layer'];slot=step['request_ids'].index(rid);position=torch.tensor(step['positions'][slot]);total=logical[(layer,rid,'k')].shape[0];keys=torch.arange(total)
    allowed=position[:,None]>=keys[None,:]
    if layer%2==0:allowed &= position[:,None]-keys[None,:]<4096
    allowed |= (position[:,None]<4096)&(keys[None,:]<4096)
    assert torch.equal(mask,allowed) and attn['past_length']==int(position[0]) and attn['key_length']==total and attn['q_heads']==8 and attn['kv_heads']==4 and attn['layer_type']==('sliding_attention' if layer%2==0 else 'full_attention')
   logits=tensor(step['full_logits']);assert logits.dtype==torch.bfloat16 and logits.shape==(len(plan),257216) and torch.isfinite(logits).all();best=logits.argmax(-1).tolist();assert best==group['steps'][n]['argmax'];argmax_rows+=len(best)
   released=[]
   for (rid,past,q),token,action in zip(plan,best,step['sample_actions']):
    st=states[rid];st['past']+=q;st['forward_count']+=1;assert action['request_id']==rid
    if st['past']<4111:assert action['kind']=='discard_partial_prefill_head' and action['actual_argmax']==token
    elif st['ended']:
     assert action['kind']=='original_unused_async_lookahead' and action['actual_argmax']==token;st['finished']=True;released.append(rid)
    else:
     st['tokens'].append(token);finish='stop' if token==1 else ('length' if len(st['tokens'])>=1024 or st['past']+1>=8192 else None)
     assert action['kind']=='reported_sample' and action['token_id']==token and action['generation_count']==len(st['tokens']) and action['finish_reason']==finish and action['finished']==bool(finish)
     st['ended']=bool(finish);st['finish']=finish
     if finish and (len(st['tokens'])>=1024 or st['past']+1>=8192):st['finished']=True;released.append(rid)
   freed=[]
   if n:
    for rid in running:
     st=states[rid]
     if rid in previous and st['held'] and (st['past']>=4097 or st['finished']):st['held']=False;used-=4096;freed.append(rid)
   for rid in released:
    st=states[rid]
    if st['held']:st['held']=False;used-=4096;freed.append(rid)
    running.remove(rid);entry=by_index[st['row']['index']];assert entry['generation_token_ids']==st['tokens'] and entry['finish_reason']==st['finish'] and entry['actual_request_forward_count']==st['forward_count']
    assert len(entry['final_all26_KV'])==26
    for layer,cache in enumerate(entry['final_all26_KV']):
     assert cache['layer']==layer and cache['device']==vd and cache['dtype']=='torch.bfloat16'
     for kind,label in [('k','key'),('v','value')]:
      key=(layer,rid,kind);value=logical.pop(key);assert cache[label+'_shape']==list(value.shape) and cache[label+'_SHA256']==fp(value);del logical_buffers[key],logical_lengths[key]
   assert freed==step['encoder_freed_after_async_previous_output']
   for rid in freed:visual.pop(rid,None)
   previous=set(step['request_ids']);all_steps+=1
  assert not waiting and not running and not logical and not logical_buffers and not logical_lengths and not visual and used==0 and visited==set(arrays.files)
  assert next(waves,None) is None
 for row in requests:
  e=by_index[row['index']];assert e['error'] is None and decode(rust,row['input_ids'],e['generation_token_ids'],e['finish_reason'])==e['raw_output']
scores=[];empty=[];truncated=[];invalid=[]
for i,(a,b,c) in enumerate(zip(inputs,outputs,details)):
 e=by_index[i];raw=e['raw_output'];pred=caption_extract(raw);ref=dataset[i]['reference']
 assert a['prompt']==c['prompt']==dataset[i]['prompt'] and a['reference']==b['reference']==c['reference']==ref
 assert b['raw_output']==c.get('raw_output','')==raw and b['prediction']==c['prediction']==pred and b['error'] is None and c['extra']['execution']==e
 scores.append(rouge(pred,ref))
 if not raw:empty.append(i)
 if e['truncated']:truncated.append(i)
 if not re.search('[a-z0-9]',str(pred).lower()):invalid.append(i)
score=sum(scores)/count;assert report['metrics']=={'ROUGE-L':score} and report['summary']['current_score']==score and report['summary']['primary_metric']=='ROUGE-L'
kernels={}
for kind,path in line['profiler_traces'].items():
 data=json.loads(checked(path).read_text());events=data if isinstance(data,list) else data['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
 values=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(values)>100;kernels[kind]=len(values)
rr=line['actual_NPU_RoPE_and_scale']
with np.load(checked(rr['path'],rr['SHA256']),allow_pickle=False) as arrays:
 assert set(arrays.files)=={'rope','scale'};rope=torch.from_numpy(arrays['rope'].copy()).view(torch.bfloat16).reshape(8192,256);scale=torch.from_numpy(arrays['scale'].copy()).view(torch.bfloat16);assert scale.numel()==1 and scale.item()==48.
 assert fp(rope)==line['original_full_RoPE_buffer']['SHA256'] and torch.isfinite(rope).all() and rope.abs().max()<=1
 frequency=1/(10000.**(torch.arange(0,256,2,dtype=torch.float32)/256));angle=torch.arange(8192,dtype=torch.float32)[:,None]*frequency[None,:];reference=torch.cat([angle.cos(),angle.sin()],-1).bfloat16();difference=(rope.float()-reference.float()).abs();rope_diag={'actual_NPU_SHA256':fp(rope),'CPU_original_formula_BF16_SHA256':fp(reference),'max_abs':float(difference.max()),'unequal_elements':int((rope!=reference).sum()),'not_forced_bitwise_equal':True}
relative=abs(Decimal(str(score))-Decimal(p['A100_D']))/abs(Decimal(p['A100_D']))
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'metrics':{'ROUGE-L':score},'per_sample_ROUGE_L':scores,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'historical_record_reuse':False,'NEW_real_NPU_inference':True,'all727_source_cast_actual_NPU_SHA_and541_physical_packed_SHA_verified':True,'all53_semantic_activations_verified':True,'all26_KV_layer_delta_records_verified':KV_count,'all_actual_forward_steps_verified':all_steps,'full257216_BF16_argmax_rows_recomputed':argmax_rows,'original_batch10_request_counts':[g['request_count'] for g in groups],'actual_encoder_wave_sizes':encoder_waves,'original_full_algorithm_caption_instruction_and_extractTrue_verified':True,'original_partial_prefill_and_discarded_sampling_and_EOS_lookahead_and_budget_boundary_verified':True,'all439_original_vision_parameter_control_SHA':source_control,'NPU_profiler_kernel_events':sum(kernels.values()),'kernel_counts':kernels,'profiler_capture_scope':'First actual visual wave and first native decoder prefill only','report_sha256':{x.name:sha(x) for x in folder.iterdir()},'external_evidence_sha256':external,'new_independent_verifier_SHA256':sha(pathlib.Path(__file__)),'A100_D':p['A100_D'],'relative_error_to_A100_table_D':float(relative),'formal_alignment_within_5pct':count==200 and relative<=Decimal('.05'),'pre_numerical_review_real3_not_approved':not bool(kw.get('numerical_review')),'precision_disclosure':line['precision_disclosure'],'own_full8192_RoPE_independent_formula_diagnostic':rope_diag}
if kw.get('numerical_review'):
 nr=json.loads(checked(kw['numerical_review'],kw['numerical_review_sha256']).read_text());assert nr['passed'] and nr['diagnostic_only'] and nr['original_BF16_A100_vllm_protocol_preserved']
 for path,digest in nr['evidence_SHA256'].items():checked(path,digest)
 proof['numerical_review']=nr
proof['external_evidence_sha256']=external;dest=pathlib.Path(line['weight_loading_evidence']).parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False,indent=2))

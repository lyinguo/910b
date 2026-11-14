"""Independent actual-A100 Nougat audit: full own source/cast, pixels/raw/dual metrics and real NPU execution."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import bisect,collections,decimal,hashlib,json,math,pathlib,re,sys
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import AutoConfig,VisionEncoderDecoderModel,NougatProcessor,GenerationConfig
R=pathlib.Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts'));torch.set_num_threads(4)
from universal_eval.metric.rouge_910b_porter import PorterStemmer
from nougat_original_cpu_reference_v2_910b import independent_nougat_pixels
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
stem=PorterStemmer()
def words(s):return [stem.stem(v) if len(v)>3 else v for v in re.findall('[a-z0-9]+',str(s).lower())]
def rouge(a,b):
 a=words(a);b=words(' '.join(b) if isinstance(b,list) else b)
 if not a or not b:return 0.
 pos={}
 for i,v in enumerate(b):pos.setdefault(v,[]).append(i)
 tail=[]
 for v in a:
  for n in reversed(pos.get(v,[])):
   at=bisect.bisect_left(tail,n)
   if at==len(tail):tail.append(n)
   else:tail[at]=n
 return 2.*len(tail)/(len(a)+len(b))
def bleu(pred,reference):
 h=str(pred or '').strip().split();refs=[reference] if isinstance(reference,str) else reference;refs=[str(v).strip().split() for v in refs]
 if not h or not refs or any(not v for v in refs):return 0.
 ps=[]
 for n in range(1,5):
  hc=collections.Counter(zip(*(h[k:] for k in range(n))));rc=[collections.Counter(zip(*(v[k:] for k in range(n)))) for v in refs];c=sum(min(x,max(v[g] for v in rc)) for g,x in hc.items())
  if n==1 and c==0:return 0.
  ps.append((c if c else .1)/max(1,len(h)-n+1))
 rlen=min((len(v) for v in refs),key=lambda x:(abs(x-len(h)),x));bp=1. if len(h)>rlen else math.exp(1.-rlen/len(h));return bp*math.exp(math.fsum(.25*math.log(v) for v in ps))

folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200];required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
r=json.loads((folder/'eval_report.json').read_text());cfg=r['pipeline_config'];p=cfg['model']['kwargs']['protocol'];device='npu:'+str(cfg['model']['kwargs']['npu_devices'][0]);assert cfg['model']['name']==p['model_id']=='MohamedRashad/arabic-large-nougat' and cfg['task_type']==folder.parent.name=='image-to-text' and folder.parent.parent.name=='MohamedRashad_arabic-large-nougat' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['model']['download'] is False and cfg['data']['download'] is False and cfg['data']['filter_empty'] is False
assert cfg['algorithm']=={'name':'greedy_inference','batch_size':1,'kwargs':{'extract_answer':False,'truncate_first_sentence':False}} and cfg['metrics']['include_failed_predictions'] is True and cfg['metrics']['override']==['ROUGE-L','BLEU-4']
assert p['execution_dtype']=='FP16' and p['source_dtype']=='BF16' and p['max_new_tokens']==4096 and p['extract_answer'] is False and p['truncate_first_sentence'] is False
lin=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert lin['protocol_review']==p and lin['source_dtype']=='BF16' and lin['execution_dtype']=='FP16' and lin['all_expected_parameters_loaded'] and lin['weight_tensor_count']==640 and lin['learned_parameter_count']==620 and lin['original_buffer_count']==20 and lin['untied_original_output_head_preserved']
assert lin['actual_attention']==p['actual_A100_attention']=={'root':'sdpa','encoder':'eager','decoder':'sdpa'} and set(lin['module_placements'].values())=={device} and lin['HF32_enabled']=={'conv':False,'matmul':False}
assert 'First actual encoder and unmodified first native decoder prefill' in lin['profiler_scope'] and lin['CPU_threads_for_probe']==4
for path,h in p['runtime_sha256'].items():assert sha(path)==h,path
for name in ['reference_report','reference_log','dataset_path','CPU200_reference','source_cast_manifest','original_A100_real3_diagnostic']:assert sha(p[name])==p[name+'_sha256'],name
W=pathlib.Path(cfg['model']['model_path']).resolve();assert str(W)==p['weight_path'] and W.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels') and W.parent.name+'/'+W.name==p['model_id']
for name,h in p['resource_sha256'].items():assert sha(W/name)==h,name
mc=AutoConfig.from_pretrained(W,local_files_only=True)
with torch.device('meta'):blue=VisionEncoderDecoderModel(mc)
params=dict(blue.named_parameters());buffers=dict(blue.named_buffers());assert len(params)==620 and len(buffers)==20 and not mc.tie_word_embeddings and not mc.decoder.tie_word_embeddings
loading=json.loads(pathlib.Path(lin['weight_loading_evidence']).read_text());cast={v['source_key']:v for v in json.loads(pathlib.Path(p['source_cast_manifest']).read_text())};assert len(loading)==len(cast)==640 and {v['source_key'] for v in loading}==set(dict(params,**buffers))
aa=json.loads(pathlib.Path(p['original_A100_real3_diagnostic']).read_text());assert aa['all3_historical_raw_equal'] and aa['actual_attention']==lin['actual_attention'] and aa['original_report_SHA256']==p['actual_A100_original_report_sha256'] and aa['original_caption_raw_equals_prediction_count']==200
lp={v['source_key']:v for v in loading};source_positions=None
for name,w in p['weights'].items():
 file=W/name;assert file.stat().st_size==w['size'] and sha(file)==w['sha256']
 with safe_open(file,framework='pt',device='cpu') as handle:
  for key in handle.keys():
   x=handle.get_tensor(key);y=x.half() if key in params else x;c=cast[key];a=lp[key];assert key==a['target_key'] and list(x.shape)==c['shape']==a['shape'] and x.dtype==(torch.bfloat16 if key in params else torch.int64)
   assert fp(x)==c['source_sha256']==a['source_sha256'] and fp(y)==c['original_FP16_sha256']==a['actual_NPU_sha256'] and a['device']==device
   assert a['actual_NPU_sha256']==(aa['actual_parameter_sha256'][key] if key in params else aa['actual_buffers'][key]['sha256'])
   assert c['roundtrip_changed_elements']==(int((x!=y.to(x.dtype)).sum()) if key in params else 0) and c['nonzero_underflow']==(int(((x!=0)&(y==0)).sum()) if key in params else 0)
   if key=='decoder.model.decoder.embed_positions.weight':source_positions=y
assert source_positions is not None and source_positions.shape==(8194,1024) and sum(v['roundtrip_changed_elements'] for v in loading)==lin['source_cast_changed']==36981 and sum(v['nonzero_underflow'] for v in loading)==lin['source_cast_underflow']==186
blue.generation_config=GenerationConfig.from_pretrained(W,local_files_only=True);gen,_=blue._prepare_generation_config(None,max_new_tokens=4096,do_sample=False,temperature=0.0);assert gen.to_dict()==lin['effective_generation_config']==p['effective_generation_config']==aa['effective_generation_config'] and gen.use_cache is True and gen.num_beams==1 and gen.forced_eos_token_id==2
processor=NougatProcessor.from_pretrained(W,local_files_only=True);ip=processor.image_processor;assert json.loads(json.dumps(ip.to_dict()))==p['image_processor'];rust=Tokenizer.from_file(str(W/'tokenizer.json'))
original=json.loads(pathlib.Path(p['reference_report']).read_text());oldrows=original['detailed_results'];assert len(oldrows)==200 and all(v['raw_output']==v['prediction'] for v in oldrows)
oldmetrics={'ROUGE-L':sum(rouge(v['prediction'],v['reference']) for v in oldrows)/200,'BLEU-4':sum(bleu(v['prediction'],v['reference']) for v in oldrows)/200}
for k,v in oldmetrics.items():assert abs(v-original['metrics'][k])<1e-12,(k,v)
cpu=json.loads(pathlib.Path(p['CPU200_reference']).read_text());data=[json.loads(v) for v in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert len(data)==1459 and sha(p['dataset_path'])=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b'
inputs=[json.loads(v) for v in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(v) for v in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=r['detailed_results'];tracepath=pathlib.Path(lin['per_sample_evidence']);trace=[json.loads(v) for v in tracepath.read_text().splitlines()];gelupath=pathlib.Path(lin['native_GELU_execution_trace']);gelus=[json.loads(v) for v in gelupath.read_text().splitlines()]
for rows in [inputs,outputs,details]:assert len(rows)==count and [v['index'] for v in rows]==list(range(count))
assert len(trace)==len(gelus)==count
probe=lin['native_GELU_probe'];assert probe['dtype']=='torch.float16' and probe['CPU_threads']==4 and sha(probe['path'])==probe['sha256']
with np.load(probe['path'],allow_pickle=False) as z:
 x=torch.from_numpy(z['input_FP16_bits'].copy()).view(torch.float16);npu=torch.from_numpy(z['NPU_FP16_bits'].copy()).view(torch.float16);erf=torch.from_numpy(z['CPU_erf_FP16_bits'].copy()).view(torch.float16);tanh=torch.from_numpy(z['CPU_tanh_diagnostic_FP16_bits'].copy()).view(torch.float16)
 assert torch.equal(x,torch.linspace(-8.,8.,20001).half()) and torch.equal(erf,torch.nn.functional.gelu(x,approximate='none')) and torch.equal(tanh,torch.nn.functional.gelu(x,approximate='tanh')) and torch.isfinite(npu).all()
 assert float((npu-erf).abs().max())==probe['max_abs_CPU_erf'] and float((npu-tanh).abs().max())==probe['max_abs_CPU_tanh_diagnostic']
external={lin[k]:sha(lin[k]) for k in ['weight_loading_evidence','per_sample_evidence','native_GELU_execution_trace','profiler_trace']};external[probe['path']]=probe['sha256']
for key in ['reference_report','reference_log','CPU200_reference','source_cast_manifest','original_A100_real3_diagnostic']:external[p[key]]=sha(p[key])
empty=[];invalid=[];truncated=[];rouges=[];bleus=[];steps=0
for i,(inp,out,d,e,g) in enumerate(zip(inputs,outputs,details,trace,gelus)):
 ref=cpu[i];assert inp['prompt']==ref['prompt']==oldrows[i]['prompt']==data[i]['prompt']==e['prompt'] and inp['reference']==out['reference']==d['reference']==ref['reference']==data[i]['reference'];assert out['extra']['execution']==e and e['error'] is None and out['error'] is None and e['model_receives_original_ten_token_prompt_and_no_reference'] and e['original_index']==i
 media=pathlib.Path(inp['media_path']).resolve();assert str(media)==ref['resolved_media_path'] and sha(media)==ref['source_image_sha256']==e['source_image_sha256']
 with Image.open(media) as im:image=im.convert('RGB')
 manual=independent_nougat_pixels(image,p['image_processor']);native=ip(images=image,return_tensors='pt')['pixel_values'];assert torch.equal(manual,native);check(manual,ref['CPU_pixels']);check(manual,e['CPU_pixels']);check(manual.half(),e['NPU_FP16_pixels'])
 seq=e['generation_sequence_ids'];initial=e['actual_prepared_initial_decoder_ids'];n=e['generation_token_count'];ids=rust.encode(inp['prompt'],add_special_tokens=False).ids
 assert len(ids)==10 and e['original_decoder_prompt_ids_no_special_tokens']==ref['original_decoder_prompt_ids_no_special_tokens']==[ids] and initial==ids and seq[:10]==initial and len(seq)==10+n and seq[-1]==2 and 1<=n<=4096
 assert lin['original_decoder_prompt_len']==lin['original_returned_sequence_slice_start']==e['original_output_slice_start']==10 and e['original_decoded_output_ids']==seq[10:]
 raw=rust.decode(seq[10:],skip_special_tokens=True)
 if processor.tokenizer.clean_up_tokenization_spaces:
  for a,b in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(" ' ","'"),(" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")]:raw=raw.replace(a,b)
 raw=raw.strip();assert raw==out.get('raw_output','')==d.get('raw_output','')==e['raw_output']==out['prediction']==d['prediction']
 assert len(e['text_positions'])==len(e['self_attention_masks'])==len(e['full_vocabulary_logits_execution'])==n
 for j,(pos,mask,logits) in enumerate(zip(e['text_positions'],e['self_attention_masks'],e['full_vocabulary_logits_execution'])):
  expected_positions=list(range(10)) if j==0 else [10+j-1];expected_ids=[ids] if j==0 else [[seq[10+j-1]]];past=0 if j==0 else 10+j-1
  assert pos['input_ids']==expected_ids and pos['past_length']==past and pos['position_ids']==expected_positions and pos['device']==device;check(source_positions[torch.tensor(expected_positions)+2].unsqueeze(0),pos['actual_learned_position_output'])
  if mask is not None:
   dtype={'torch.float16':torch.float16,'torch.float32':torch.float32,'torch.bool':torch.bool}[mask['dtype']];q=10 if j==0 else 1;k=10+j
   if dtype==torch.bool:expected_mask=torch.ones((q,k),dtype=dtype).tril(diagonal=k-q)
   else:expected_mask=torch.full((q,k),torch.finfo(dtype).min,dtype=dtype).triu(diagonal=k-q+1)
   check(expected_mask[None,None],mask)
  assert logits['step']==j and logits['shape']==[1,86000] and logits['dtype']=='torch.float16' and logits['device']==device and logits['all_finite'] and re.fullmatch('[0-9a-f]{64}',logits['full_last_logits_SHA256']) and 0<=logits['actual_full86000_argmax_id']<86000 and seq[10+j]==(2 if j==4095 else logits['actual_full86000_argmax_id'])
 archive=pathlib.Path(e['visual_feature_archive']);assert sha(archive)==e['visual_feature_archive_sha256'];external[str(archive)]=sha(archive)
 with np.load(archive,allow_pickle=False) as z:a=torch.from_numpy(z['encoder_FP16_bits'].copy()).view(torch.float16);b=torch.from_numpy(z['actual_cross_FP16_bits'].copy()).view(torch.float16)
 assert a.shape==b.shape==(1,588,1024) and torch.equal(a,b) and torch.isfinite(a).all();check(a,e['encoder_output']);check(b,e['actual_decoder_cross_attention_input']);assert e['vision_forward_calls']==1 and e['decoder_cross_attention_calls']==[n]*10 and e['final_cache_length']==10+n-1
 for name,rows in e['cache_device_and_shapes'].items():
  assert name in ['self_attention_cache','cross_attention_cache'] and len(rows)==mc.decoder.num_hidden_layers==lin['native_cache_allocated_slots'] and lin['actual_decoder_layers']==10
  for slot,row in enumerate(rows):
   assert row['slot']==slot and row['initialized']==(slot<10)
   if slot<10:assert row['device']==device and row['dtype']=='torch.float16' and row['key_shape']==row['value_shape']==[1,16,10+n-1 if name=='self_attention_cache' else 588,64]
   else:assert row=={'slot':slot,'initialized':False}
 assert e['allocator_before']==e['allocator_after']==0 and e['truncated']==e['forced_eos_at_original_budget']==(n==4096)
 assert g['index']==i and g['error'] is None and g['dtype']=='torch.float16' and g['device']==device and g['generation_token_count']==n and g['module_call_counts']==e['native_GELU_call_counts'];counts=g['module_call_counts'];assert len(counts)==30 and sum(k.startswith('encoder.') for k in counts)==20 and all(v==(1 if k.startswith('encoder.') else n) for k,v in counts.items())
 if not raw:empty.append(i)
 elif not any(x.isalnum() for x in raw):invalid.append(i)
 if n==4096:truncated.append(i)
 rouges.append(rouge(raw,out['reference']));bleus.append(bleu(raw,out['reference']));steps+=n
metrics={'ROUGE-L':sum(rouges)/count,'BLEU-4':sum(bleus)/count}
for k,v in metrics.items():assert abs(v-r['metrics'][k])<1e-12,(k,v,r['metrics'][k])
assert r['summary']['primary_metric']=='ROUGE-L' and r['statistics']['total_samples']==count and r['statistics']['error_count']==0
prof=json.loads(pathlib.Path(lin['profiler_trace']).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hw={v['pid'] for v in events if v.get('name')=='process_name' and v.get('args',{}).get('name')=='Ascend Hardware'};kernels=[v for v in events if v.get('pid') in hw and v.get('ph')=='X' and str(v.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
D=decimal.Decimal(str(p['A100_D']));score=decimal.Decimal(str(metrics['ROUGE-L']));assert D!=0;relative=abs(D-score)/abs(D)
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'historical_record_reuse':False,'metrics':metrics,'ROUGE-L':metrics['ROUGE-L'],'source_dtype':'BF16','execution_dtype':'FP16','original_decoder_prompt_len':10,'original_output_slice_start':10,'source_tensor_count':640,'learned_parameter_count':620,'original_buffer_count':20,'actual_activation_module_count':30,'actual_native_cache_layers':10,'actual_native_cache_allocated_slots':mc.decoder.num_hidden_layers,'actual_attention':lin['actual_attention'],'actual_full_vocabulary_size':86000,'full_vocab_greedy_steps':steps,'source_cast_changed':36981,'source_cast_underflow':186,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'independent_CPU_pixel_replays':count,'original_raw_no_firstsentence_or_extraction_replayed':True,'original_A100_all200_dual_metrics_recomputed':oldmetrics,'NPU_profiler_kernel_events':len(kernels),'profiler_scope':lin['profiler_scope'],'external_evidence_sha256':external,'sample_ROUGE_L_F1':rouges,'sample_BLEU_4':bleus,'native_GELU_probe':probe,'relative_error_to_A100_table_D':float(relative),'formal_alignment_within_5pct':bool(count==200 and relative<=decimal.Decimal('.05')),'A100_D':p['A100_D'],'report_sha256':{v.name:sha(v) for v in folder.iterdir()},'protocol':'Actual A100 original FP16/encoder eager and decoderSDPA/nativecache/original-ten-token-decoder-prompt/own896x672 crop,uint8bilinear,bicubicthumbnail,pad,fusednormalization;original4096/forcedEOS2/fullraw,noextraction/noFirstSentence;full86000 actualNPUargmax/own untiedhead/640source-cast-CUDA-NPU hashes/all30activations/positions+2/cross/nativecache and full-denominator dualmetrics independently verified. No model math,precision,mask,budget or reference substitution.'}
num_path=pathlib.Path(p['numerical_disclosure']);assert sha(num_path)==p['numerical_disclosure_sha256'];num=json.loads(num_path.read_text());assert num['diagnostic_evidence_verified'] and num['no_NPU_model_math_precision_mask_budget_change'] and lin['Arabic_A100_original_numerical_disclosure']==num
assert lin['Arabic_A100_original_numerical_disclosure_sha256']==p['numerical_disclosure_sha256'];static=dict(num['all_static_diagnostic_evidence_sha256']);static[str(num_path)]=sha(num_path)
for f,h in static.items():assert sha(f)==h,f
audit=json.loads(pathlib.Path(num['original_numerical_audit']).read_text());assert audit['CPU_diagnostics_only'] and audit['no_CPU_decoder_answers_or_official_scores'] and audit['same_source_FP16_encoder_all_parameter_and_buffer_SHA_count']==audit['independent_FP32_CPU_NPU_same_rounded_encoder_tensor_SHA_count']==373
for f in audit['evidence_sha256']:
 if not f.endswith('.npz'):continue
 with np.load(f,allow_pickle=False) as zz:
  if 'independent_same_FP16_rounded_FP32' in f:
   a=torch.from_numpy(zz['CPU_FP32_bits'].copy()).view(torch.float32);b=torch.from_numpy(zz['NPU_FP32_bits'].copy()).view(torch.float32);idx=int(pathlib.Path(f).stem.rsplit('_',1)[-1]);assert a.shape==b.shape==(1,588,1024) and torch.isfinite(a).all() and torch.isfinite(b).all() and float((a-b).abs().max())==audit['rows'][idx]['independent_same_rounded_FP32_CPU_vs_NPU5']['max_abs']
  elif 'CPU_FP16_encoder_bits' in zz:
   a=torch.from_numpy(zz['CPU_FP16_encoder_bits'].copy()).view(torch.float16);assert a.shape==(1,588,1024) and torch.isfinite(a).all()
raw={}
for f in sorted(tracepath.parent.rglob('*')):
 if f.is_file() and f.name!='independent_report_verification.json':raw[str(f)]=sha(f)
for f,h in raw.items():
 if f in external:assert external[f]==h
 external[f]=h
external.update(static);external[p['native_GELU_method_note']]=p['native_GELU_method_note_sha256'];assert sha(p['native_GELU_method_note'])==p['native_GELU_method_note_sha256']
proof['external_evidence_sha256']=external;proof['Arabic_A100_original_numerical_diagnostic_evidence_sha256']=static;proof['raw_NPU_source_snapshot_CANN_trace_file_count']=len(raw);proof['all_raw_and_static_diagnostic_evidence_SHA_verified']=True;proof['diagnostic_static_controls_referenced_not_rerun_per_report']=True;proof['numerical_disclosure_rows']=audit['rows'];proof['reference_metric_qualification']=p['metric_reference_qualification']
dest=tracepath.parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False,indent=2))

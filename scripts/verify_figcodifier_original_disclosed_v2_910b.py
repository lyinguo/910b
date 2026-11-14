"""Independent complete evidence verifier, authoritative count3 or fixed200."""
import sys,os,pathlib,json,re,collections,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from figcodifier_independent_cpu_910b import sha,fp,tr,arrays,independent_pixels,OwnSentencePieceReference
from idefics2_original_accuracy_reference_910b import original_accuracy_sample
from strict_mc_independent_reference_910b import independent_choice
torch.set_num_threads(4)
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256'];p=json.loads(pathlib.Path(kw['protocol_file']).read_text());wp=pathlib.Path(p['weight_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];dev='npu:'+str(kw['npu_devices'][0]);assert p['model_id']==cfg['model']['name']=='MathLLMs/FigCodifier' and cfg['model']['model_path']==str(wp) and folder.parent.parent.name=='MathLLMs_FigCodifier' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['batch_size']==10 and cfg['algorithm']['adapter']=='universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2' and cfg['algorithm']['kwargs']=={'extract_answer':True} and cfg['metrics']['override']==['Strict_MC_Accuracy'] and cfg['metrics']['adapters']==['universal_eval/algorithm/tinyllava_strict_mc.py:StrictMCAccuracy'] and cfg['metrics']['include_failed_predictions']
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True} and cfg['data']['preprocess'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template'] and not cfg['data']['download'] and not cfg['model']['download']
assert line['protocol']=='FigCodifier_original_BF16_eager_manual_serial_generation' and line['protocol_sha256']==kw['protocol_sha256'] and line['runtime_dtype']=='BF16' and line['weight_tensor_count']==573 and line['all_expected_parameters_loaded'] and line['original_untied_output_head_preserved'] and line['own_LM_architecture']=='InternLM2ForCausalLM' and line['own_language_attention']=='InternLM2Attention' and line['original_manual_stop_EOS_ID']==92542 and line['original_manual_max_new_tokens']==2048 and line['original_manual_explicit_use_cache'] and line['actual_language_sample_batch_size']==1 and line['outer_algorithm_metadata_batch_size']==10 and line['own_vocab_size']==92557 and line['own_BOS_ID']==1 and line['image_tokens_per_tile']==256 and set(line['module_placements'].values())=={dev} and line['HF32_enabled']=={'conv':False,'matmul':False}
for n,d in dict(p['runtime_sha256'],**p['evidence_sha256']).items():assert sha(n)==d
for n,d in p['resource_sha256'].items():assert pathlib.Path(n).name==n and sha(wp/n)==d
expected={r['source_key']:r for r in p['source_tensor_records']};loading=pathlib.Path(line['weight_loading_evidence']);loaded={r['source_key']:r for r in json.loads(loading.read_text())};assert len(expected)==len(loaded)==573
for file,header in p['source_files'].items():
 assert sha(wp/file)==header['sha256'] and (wp/file).stat().st_size==header['bytes']
 with safe_open(str(wp/file),framework='pt',device='cpu') as f:
  for k in f.keys():
   x=f.get_tensor(k);r=loaded[k];e=expected[k];digest=fp(x);assert x.dtype==torch.bfloat16 and str(x.dtype)==r['source_dtype']==r['runtime_dtype'] and r['device']==dev and list(x.shape)==r['shape']==e['shape'] and r['source_file']==e['source_file']==file and r['target_key']==k and digest==e['source_full_sha256']==e['source_BF16_cast_full_sha256']==r['source_sha256']==r['original_BF16_cast_sha256']==r['actual_NPU_sha256'] and r['original_cast_changes']==0
assert loaded['language_model.output.weight']['actual_NPU_sha256']!=loaded['language_model.model.tok_embeddings.weight']['actual_NPU_sha256']
inv=1.0/(1000000**(torch.arange(0,128,2).float()/128));freq=torch.outer(torch.arange(32768).float(),inv);emb=torch.cat((freq,freq),-1);cos=emb.cos();sin=emb.sin();assert len(line['original_32_analytic_rotary_buffers'])==32
for i,r in enumerate(line['original_32_analytic_rotary_buffers']):assert r['layer']==i and r['class']=='InternLM2DynamicNTKScalingRotaryEmbedding' and r['dim']==128 and r['base']==1000000 and r['factor']==2 and r['max_seq_len_cached']==32768 and r['buffers']=={'inv_freq':tr(inv),'cos_cached':tr(cos),'sin_cached':tr(sin)}
assert sha(p['input_audit'])==p['input_audit_sha256'];audit=json.loads(pathlib.Path(p['input_audit']).read_text());records={r['index']:r for r in audit['records']};assert sorted(records)==list(range(200));own=OwnSentencePieceReference(wp)
data=[json.loads(x) for x in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert len(data)==4329 and sha(p['dataset_path'])==p['dataset_sha256'];old=json.loads(pathlib.Path(p['reference_report']).read_text());assert sha(p['reference_report'])==p['reference_report_sha256'] and len(old['detailed_results'])==200
ins=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];trace_path=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(x) for x in trace_path.read_text().splitlines()];acts_path=pathlib.Path(line['activation_execution_trace']);acts=[json.loads(x) for x in acts_path.read_text().splitlines()];assert len(ins)==len(outs)==len(details)==len(trace)==len(acts)==count
for rows in [ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
assert sorted(x['index'] for x in trace)==list(range(count));by={x['index']:x for x in trace};actby={x['index']:x for x in acts};expected_order=[]
for start in range(0,count,10):
 groups=collections.OrderedDict()
 for i in range(start,min(start+10,count)):groups.setdefault(tuple(v//64 for v in records[i]['original_size']),[]).append(i)
 for group in groups.values():expected_order+=group
assert [x['index'] for x in trace]==expected_order and len(line['activation_modules'])==57
external={str(q):sha(q) for q in [loading,trace_path,acts_path,pathlib.Path(line['profiler_trace']),pathlib.Path(line['native_GELU_probe']['path'])]};correct=legacy=0;empty=[];invalid=[];truncated=[];steps_total=0;scores=[];tilehist=collections.Counter()
for i,(a,b,r) in enumerate(zip(ins,outs,details)):
 e=by[i];rec=records[i];assert e['error'] is None and e['actual_original_model_prompt']==rec['actual_original_model_prompt'] and sha(rec['media_path'])==rec['source_image_sha256']==e['source_image_sha256'] and a['prompt']==r['prompt']==rec['prompt'] and a['reference']==b['reference']==r['reference']==rec['reference']
 q=pathlib.Path(e['sample_execution']);assert sha(q)==e['sample_execution_sha256'];g=json.loads(q.read_text());external[str(q)]=sha(q);assert sha(g['archive'])==g['archive_sha256'];external[g['archive']]=sha(g['archive']);t=arrays(g['archive'],g['tensor_records'])
 with Image.open(rec['media_path']) as im:px,grid=independent_pixels(im.convert('RGB'))
 assert tr(px)==rec['original_pixel_values']==g['CPU_pixel_values'] and grid==rec['original_grid']==g['original_grid'];M=px.shape[0];tilehist[M]+=1;assert g['actual_tile_batch_size']==M and g['actual_language_sample_batch_size']==1 and torch.equal(t['actual_vision_pixels'],px.bfloat16()) and t['actual_encoder'].shape==(M,1025,1024) and t['actual_connector'].shape==(M,256,4096)
 # Independent 2x2 spatial packing, row-major channels, original ps_versionv2.
 enc=t['actual_encoder'][:,1:,:].reshape(M,32,32,1024);packed=enc.reshape(M,16,2,16,2,1024).permute(0,1,3,2,4,5).reshape(M,256,4096);assert torch.equal(packed,t['actual_connector_input'])
 rendered=rec['original_packed_chat_text'];ids=own.encode(rendered);assert ids==rec['original_input_ids'] and g['rendered_text']==rendered and g['input_ids']==[ids] and ids[0]==1 and ids.count(92546)==256*M
 mask=torch.tensor([v==92546 for v in ids]);assert t['actual_prefill_inputs_embeds'].shape==(1,len(ids),4096) and torch.equal(t['actual_prefill_inputs_embeds'][0,mask],t['actual_injected_visual']) and torch.equal(t['actual_injected_visual'],t['actual_connector'].reshape(-1,4096)) and g['actual_visual_injection_verified'] and g['actual_vision_calls']==g['actual_connector_calls']==1
 tokens=g['generation_token_ids'];steps=g['steps'];assert 1<=len(tokens)==len(steps)<=2048;assert all(v!=92542 for v in tokens[:-1]) and (tokens[-1]==92542 or len(tokens)==2048)
 act=actby[i];assert act['error'] is None and act['device']==dev and act['dtype']=='torch.bfloat16' and act['language_sample_batch_size']==1 and act['tile_batch_size']==M and act['decoder_forwards']==len(tokens) and act['module_call_counts']==g['native_activation_call_counts'] and set(act['module_call_counts'])==set(line['activation_modules'])
 for name,calls in act['module_call_counts'].items():assert calls==(len(tokens) if name.startswith('language_model.') else 1)
 for n,(step,token) in enumerate(zip(steps,tokens),1):
  total=len(ids)+n-1;Q=len(ids) if n==1 else 1;pos=torch.arange(total-Q,total);assert step['step']==n and step['actual_position_ids']==[pos.tolist()] and step['attention_mask_2D']==[[1]*total]
  cm=torch.where(pos[:,None]>=torch.arange(total)[None,:],torch.tensor(0.,dtype=torch.bfloat16),torch.tensor(torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16))[None,None];assert tr(cm)==step['actual_causal_mask']
  assert torch.equal(t['%04d_cos'%n],cos[pos].bfloat16()) and torch.equal(t['%04d_sin'%n],sin[pos].bfloat16());logits=t['%04d_logits'%n];assert logits.shape==(1,92557) and logits.dtype==torch.float32 and torch.isfinite(logits).all() and tr(logits)==step['full_next_token_logits'] and int(logits.argmax(-1))==token==step['next_token_ID']
  assert len(step['layers'])==len(step['cache'])==32
  for layer,cache in zip(step['layers'],step['cache']):assert layer['layer']==cache['layer'] and layer['device']==cache['device']==dev and layer['dtype']==cache['dtype']=='torch.bfloat16' and layer['past_length']==(0 if n==1 else total-1) and layer['position_ids_same_actual_object'] and layer['mask_same_actual_object'] and cache['key_shape']==cache['value_shape']==[1,8,total,128] and cache['original_tuple_type'] and cache['actual_NPU_full_prior_key_value_prefix_equal']
 decoded=own.decode(tokens);raw=decoded.split('<|im_end|>')[0].strip();choice=independent_choice(raw);prediction=choice if choice is not None else raw;assert decoded==g['full_decoded_output_before_original_strip'] and raw==g['raw_output']==e['raw_output']==r['raw_output']==b['raw_output'] and tokens==e['generation_token_ids'] and r['prediction']==b['prediction']==prediction and b['error'] is None and r['extra']['execution']==e
 hit=int(choice is not None and choice==str(r['reference']).strip().upper());scores.append(hit);correct+=hit;legacy+=original_accuracy_sample(prediction,r['reference']);steps_total+=len(tokens)
 if not raw:empty.append(i)
 if choice is None:invalid.append(i)
 if len(tokens)>=2048:truncated.append(i)
 assert e['truncated']==g['truncated']==(len(tokens)>=2048)
score=correct/count;assert report['metrics']['Strict_MC_Accuracy']==report['summary']['current_score']==score and report['summary']['primary_metric']=='Strict_MC_Accuracy' and report['statistics']['error_count']==0 and report['statistics']['total_samples']==count and report['statistics']['batch_size']==10 and len(external)==2*count+5
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'] and probe['device']==dev;torch.set_num_threads(probe['CPU_threads'])
with np.load(probe['path'],allow_pickle=False) as z:
 for dtype,label in [(torch.float32,'FP32'),(torch.bfloat16,'BF16')]:
  x=torch.linspace(-8,8,20001).to(dtype);assert np.array_equal(z[label+'_input_bytes'],x.view(torch.uint8).numpy())
  for mode in ['none','tanh']:
   cpu=torch.nn.functional.gelu(x,approximate=mode);native=torch.from_numpy(z[label+'_NPU_'+mode+'_bytes'].copy()).view(dtype);assert np.array_equal(z[label+'_CPU_'+mode+'_bytes'],cpu.view(torch.uint8).numpy()) and bool(torch.isfinite(native).all()) and float((native-cpu).abs().max())==probe['modes'][label][mode]['max_abs_CPU_same_declared']
profile=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'sample_Accuracy':scores,'correct':correct,'original_legacy_Accuracy_diagnostic_only':legacy/count,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'historical_record_reuse':False,'inference_reused_for_verifier_repair':'--reuse-for-verifier-repair' in sys.argv,'all573_source_cast_actual_NPU_SHA_verified':True,'all_expected_visual_features_actually_injected':True,'all32_original_tuple_cache_positions_masks_RoPE_and_full92557_argmax_steps_verified':steps_total,'actual_activation_module_count':57,'all57_activation_module_counts_verified':True,'original_language_sample_batch_size':1,'original_outer10_size_group_execution_order':expected_order,'original_tile_batch_histogram':dict(tilehist),'original_manual_EOS92542_and_budget2048_preserved':True,'own_SentencePiece_and_cleanup_output_decode_verified':True,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_numerical_review_real3_not_approved':True}
assert 'native_numerical_review' in kw
nrpath=pathlib.Path(kw['native_numerical_review']);assert sha(nrpath)==kw['native_numerical_review_sha256'];nr=json.loads(nrpath.read_text());assert nr==line['native_numerical_review'] and nr['official_inference_remains_original_BF16_eager_manual_serial'] and nr['CPU_FP32_controls_diagnostic_only'] and nr['all_control_CPU_NPU_parameter_SHA_equal'] and nr['native_operator_formulas_unchanged']
ev=dict(nr['evidence_sha256']);ev[str(nrpath)]=sha(nrpath)
for key,target,hashkey,linekey in [('metric_contract_file',None,'metric_contract_sha256','original_metric_contract'),('frozen_extraction_audit_file',None,'frozen_extraction_audit_sha256','original_frozen_extraction_audit')]:
 q=pathlib.Path(kw[key]);assert sha(q)==kw[hashkey] and json.loads(q.read_text())==line[linekey];ev[str(q)]=sha(q)
for path,digest in ev.items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
progress_path=pathlib.Path(line['generation_progress']);progress=[json.loads(x) for x in progress_path.read_text().splitlines()];complete=[x for x in progress if x['event']=='sample_complete'];assert [x['index'] for x in complete]==expected_order and len(complete)==count
for e in trace:
 active=[x for x in progress if x['index']==e['index'] and x['event']=='native_decoder_forward_completed'];assert active and active[0]['decoder_forward_completed']==1 and active[-1]['decoder_forward_completed']==e['generation_count']
 for x in active:assert x['tokens_already_appended_before_this_return']==x['decoder_forward_completed']-1 and x['actual_language_batch_size']==1 and x['original_budget']==2048 and x['actual_NPU_device']==dev
 assert next(x for x in complete if x['index']==e['index'])['generation_token_count']==e['generation_count']
external[str(progress_path)]=sha(progress_path);assert len(external)==2*count+6
proof.update(external_evidence_sha256=external,native_numerical_difference_retained_and_disclosed=True,pre_numerical_review_real3_not_approved=False,native_numerical_review=nr,figcodifier_numerical_diagnostic_evidence_sha256=ev,static_CPU_controls_from_prior_real3_not_recomputed_per_report=True,original_same_historical_raw_strict_MC_diagnostic=line['original_metric_contract']['same_historical_raw_strict_MC_v2_diagnostic'],ordinary_articles_never_credited=True)

dest=loading.parent/'independent_report_verification.json';assert not dest.exists() or json.loads(dest.read_text())==proof
if not dest.exists():dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))

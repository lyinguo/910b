"""Independent exact source, true groups, all logits/positions/cache/vision and metric replay."""
import pathlib,sys,os,json,re,collections,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent));sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from idefics2_grouped_independent_cpu_910b import *
from idefics2_original_accuracy_reference_910b import original_accuracy_sample,safe_credit_reason
from transformers import AutoConfig,Idefics2ForConditionalGeneration,GenerationConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
torch.set_num_threads(4)
def check(t,r):assert list(t.shape)==r['shape'] and str(t.dtype)==r['dtype'] and fp(t)==r['sha256']
DTYPES={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64,'torch.bool':torch.bool}
def load_arrays(path,records):
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(records)
  tensors={k:torch.from_numpy(z[k].copy()).view(DTYPES[r['dtype']]).reshape(r['shape']) for k,r in records.items()}
 for k,t in tensors.items():check(t,records[k]);assert torch.isfinite(t).all()
 return tensors
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200];required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {x.name for x in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256'];p=json.loads(pathlib.Path(kw['protocol_file']).read_text());wp=pathlib.Path(p['weight_path'])
assert cfg['model']['name']==p['model_id'] in ['HuggingFaceM4/idefics2-8b','HuggingFaceM4/idefics2-8b-chatty'] and cfg['model']['model_path']==str(wp)
assert folder.parent.name==cfg['task_type']=='image-text-to-text' and folder.parent.parent.name==p['model_id'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['batch_size']==10 and cfg['algorithm']['kwargs']=={'extract_answer':True} and cfg['algorithm']['adapter']=='universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2' and cfg['metrics']=={'override':['Accuracy'],'include_failed_predictions':True}
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True} and cfg['data']['download'] is False and cfg['data']['filter_empty'] is False and cfg['model']['download'] is False
line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];device='npu:'+str(kw['npu_devices'][0])
assert line['protocol']=='idefics2_original_grouped_generation_BF16' and line['protocol_sha256']==kw['protocol_sha256'] and line['all_expected_parameters_loaded'] and line['independent_lm_head_preserved'] and line['source_dtype']=='F32' and line['runtime_dtype']=='BF16' and line['native_attention_implementation']=='eager' and line['original_cache'] is True and line['original_max_new_tokens']==2048 and line['original_algorithm_batch_size']==10 and line['image_splitting'] is True and line['image_tokens_per_sample']==320 and line['model_vocab_size']==32003 and line['HF32_enabled']=={'conv':False,'matmul':False} and set(line['module_placements'].values())=={device}
for key in ['reference_report','reference_log','dataset_path','original_dataset','CPU_grouped_input_preflight','original_protocol_review']:assert sha(p[key])==p[key+'_sha256']
for source,digest in dict(p['runtime_sha256'],**p['historical_sources'],**p['frozen_before_original_generation_sources']).items():assert sha(source)==digest
for name,digest in p['resource_sha256'].items():assert sha(wp/name)==digest
data=[json.loads(s) for s in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert len(data)==4329
old=json.loads(pathlib.Path(p['reference_report']).read_text());assert len(old['detailed_results'])==200 and old['metrics']['Accuracy']==p['original_recomputed_Accuracy']
for i,r in enumerate(old['detailed_results']):assert r['index']==i and r['prompt']==original_clean_prompt(data[i]['prompt']) and r['reference']==data[i]['reference'] and sha(r['media_path'])==p['reference_first200_image_sha256'][str((pathlib.Path(p['dataset_path']).parent/data[i]['media_path']).resolve())]
mc=AutoConfig.from_pretrained(wp,local_files_only=True)
with torch.device('meta'):meta=Idefics2ForConditionalGeneration(mc)
assert mc.tie_word_embeddings is False and mc.text_config.vocab_size==32003
gen=GenerationConfig.from_pretrained(wp,local_files_only=True);meta.generation_config=gen;resolved,_=meta._prepare_generation_config(None,**p['original_generate_kwargs']);assert resolved.to_dict()==p['resolved_original_generation_config'] and resolved.bad_words_ids==[[32000],[32001]] and resolved.eos_token_id==[2,32002] and resolved.use_cache and resolved.num_beams==1 and not resolved.do_sample
parameters=dict(meta.named_parameters());expected_act={n:type(m).__name__ for n,m in meta.named_modules() if type(m).__name__ in ['GELUTanh','SiLUActivation']};assert len(parameters)==763 and len(expected_act)==63 and line['activation_modules']==expected_act
weights_path=pathlib.Path(line['weight_loading_evidence']);loading=json.loads(weights_path.read_text());actual={r['source_key']:r for r in loading};source={r['source_key']:r for r in p['source_tensor_records']};assert len(loading)==763 and set(actual)==set(source)==set(parameters)
embedding=None;head=None
for shard,header in p['weight_headers'].items():
 assert sha(wp/shard)==header['file_sha256']
 with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
  for key in f.keys():
   value=f.get_tensor(key);cast=value.bfloat16();r=actual[key];s=source[key];assert value.dtype==torch.float32 and r['device']==device and r['target_key']==key and r['shape']==s['shape']==list(value.shape)==list(parameters[key].shape)
   assert fp(value)==r['source_sha256']==s['source_sha256'] and fp(cast)==r['actual_NPU_sha256']==r['original_BF16_cast_sha256']==s['original_BF16_cast_sha256']
   if key=='lm_head.weight':head=cast.clone()
   if key=='model.text_model.embed_tokens.weight':embedding=cast.clone()
assert head is not None and embedding is not None and not torch.equal(head,embedding)
rope=MistralRotaryEmbedding(mc.text_config,device='cpu');check(rope.inv_freq,line['text_inv_freq'])
trace_path=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(s) for s in trace_path.read_text().splitlines()];inputs=[json.loads(s) for s in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(s) for s in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
assert len(trace)==len(inputs)==len(outputs)==len(details)==count and [r['attempt'] for r in trace]==list(range(1,count+1))
for rows in [inputs,outputs,details]:assert [r['index'] for r in rows]==list(range(count))
external={};groups=groups_for(data,p['dataset_path'],count);assert len(set(e['group_evidence'] for e in trace))==len(groups)
acts=[json.loads(s) for s in pathlib.Path(line['activation_execution_trace']).read_text().splitlines()];finite=[json.loads(s) for s in pathlib.Path(line['next_token_logits_finiteness']).read_text().splitlines()];progress=[json.loads(s) for s in pathlib.Path(line['generation_progress']).read_text().splitlines()];assert len(acts)==len(groups)
for name in ['weight_loading_evidence','per_sample_evidence','activation_execution_trace','profiler_trace','generation_progress','next_token_logits_finiteness']:external[line[name]]=sha(line[name])
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'];external[probe['path']]=probe['sha256']
with np.load(probe['path'],allow_pickle=False) as z:
 x=torch.from_numpy(z['input_BF16_bytes'].copy()).view(torch.bfloat16)
 for approximate in ['none','tanh']:
  c=torch.nn.functional.gelu(x,approximate=approximate);n=torch.from_numpy(z['NPU_'+approximate+'_BF16_bytes'].copy()).view(torch.bfloat16);assert torch.equal(c,torch.from_numpy(z['CPU_'+approximate+'_BF16_bytes'].copy()).view(torch.bfloat16)) and torch.isfinite(n).all() and float((c-n).abs().max())==probe[approximate]['max_abs_CPU_same_declared']
rust=own_rust(wp);rope_differences=[];total_steps=0;all_samples={}
for group_number,indices in enumerate(groups,1):
 selected=[e for e in trace if e['actual_group']==group_number];assert sorted(e['index'] for e in selected)==sorted(indices) and {e['actual_generation_batch_size'] for e in selected}=={len(indices)}
 record_path=pathlib.Path(selected[0]['group_evidence']);assert len({e['group_evidence_sha256'] for e in selected})==1 and sha(record_path)==selected[0]['group_evidence_sha256'];external[str(record_path)]=sha(record_path);g=json.loads(record_path.read_text());assert g['indices']==indices and g['group']==group_number and g['batch_size']==len(indices) and g['error'] is None
 own,texts,sizes=independent_group(wp,p,data,indices);B=len(indices);L=own['input_ids'].shape[1];assert g['input_ids']==own['input_ids'].tolist() and g['attention_mask']==own['attention_mask'].tolist() and g['padded_input_length']==L
 for k,t in own.items():check(t,g['CPU_inputs'][k])
 for sample,text,index in zip(g['samples'],texts,indices):assert sample['index']==index and sample['chat_text']==text and sample['prompt']==original_clean_prompt(data[index]['prompt'])+INSTRUCTION and sample['source_image_sha256']==sha(sample['resolved_media_path'])
 for name in ['visual_archive','full_logits_archive']:assert sha(g[name])==g[name+'_sha256'];external[g[name]]=sha(g[name])
 tensors=load_arrays(g['visual_archive'],g['visual_tensor_records']);logits=load_arrays(g['full_logits_archive'],g['logits_tensor_record'])['actual_next_token_full_logits'];steps=g['decoder_steps'];assert logits.shape==(steps,B,32003) and logits.dtype==torch.bfloat16 and len(g['actual_steps'])==steps and 0<steps<=2048
 assert torch.equal(tensors['actual_vision_pixels'],own['pixel_values'].bfloat16().reshape(-1,*own['pixel_values'].shape[-3:]));patch,vision_positions=independent_vision_positions(own['pixel_attention_mask']);assert torch.equal(tensors['actual_patch_attention_mask'],patch) and torch.equal(tensors['actual_vision_positions'],vision_positions)
 encoder=tensors['actual_encoder'];features=tensors['actual_connector'];merged=tensors['actual_prefill_text_inputs_embeds'];injected=tensors['actual_injected_visual']
 assert encoder.shape==(B*5,patch.shape[1]*patch.shape[2],1152) and features.shape==(B*5,64,4096) and merged.shape==(B,L,4096) and torch.equal(features.reshape(B*320,4096),injected) and features.abs().sum()>0
 ids=own['input_ids'];assert torch.equal(merged[ids==32001],injected) and torch.equal(merged[ids!=32001],embedding[ids[ids!=32001]])
 full=torch.tensor(g['full_sequences']);generated=torch.tensor(g['generation_token_ids']);assert full.shape==(B,L+steps) and generated.shape==(B,steps) and torch.equal(full[:,:L],ids) and torch.equal(full[:,L:],generated)
 unfinished=torch.ones(B,dtype=torch.bool)
 for step,s in enumerate(g['actual_steps']):
  input_slice=full[:,:L] if step==0 else full[:,L+step-1:L+step];mask=torch.cat([own['attention_mask'],torch.ones((B,step),dtype=torch.int64)],dim=-1);past=0 if step==0 else L+step-1
  positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0);positions=positions if step==0 else positions[:,-1:]
  assert s['step']==step and s['input_ids']==input_slice.tolist() and s['input_ids_sha256']==fp(input_slice) and s['attention_mask']==mask.tolist() and s['position_ids']==positions.tolist() and s['past_length_before']==past and s['cache_length_after']==L+step
  q=positions.shape[1];key_length=L+step;actual_mask=tensors['causal_mask_%04d'%step];assert actual_mask.shape==(B,1,q,key_length)
  qpos=torch.arange(0 if step==0 else key_length-1,key_length);kpos=torch.arange(key_length);allowed=(kpos[None,:]<=qpos[:,None])&(kpos[None,:]>qpos[:,None]-4096);allowed=allowed[None,None]&mask[:,None,None,:].bool();expected_mask=torch.where(allowed,torch.tensor(0.,dtype=actual_mask.dtype),torch.tensor(torch.finfo(actual_mask.dtype).min,dtype=actual_mask.dtype));assert torch.equal(actual_mask,expected_mask)
  c,si=rope(torch.zeros((B,q,4096),dtype=torch.bfloat16),positions);ac=tensors['rope_cos_%04d'%step];asi=tensors['rope_sin_%04d'%step];diff=max(float((c-ac).abs().max()),float((si-asi).abs().max()));assert diff<=torch.finfo(torch.bfloat16).eps;rope_differences.append(diff)
  for layer in s['cache_layers']:assert layer['keys']==layer['values']==[B,8,key_length,128] and layer['device']==device and layer['dtype']=='torch.bfloat16'
  assert len(s['cache_layers'])==32
  # Native generate casts the original full BF16 vector to FP32, applies only
  # its original two image-token bad-word constraints, then first-index argmax.
  independent=logits[step].float().clone();independent[:,32000:32002]=-float('inf');next_ids=independent.argmax(-1);next_ids=torch.where(unfinished,next_ids,torch.tensor(0));assert torch.equal(next_ids,generated[:,step]),(group_number,step,next_ids,generated[:,step])
  unfinished &= (next_ids!=2)&(next_ids!=32002)
  if step<steps-1:assert unfinished.any()
 assert steps==2048 or not unfinished.any()
 assert g['actual_vision_calls']==g['actual_connector_calls']==1 and acts[group_number-1]['module_call_counts']==g['native_activation_call_counts'] and acts[group_number-1]['batch_size']==B
 for n,v in g['native_activation_call_counts'].items():assert n in expected_act and v==(steps if '.text_model.' in n else 1)
 assert set(g['native_activation_call_counts'])==set(expected_act)
 frows=[f for f in finite if f['group']==group_number];assert len(frows)==steps and [f['step'] for f in frows]==list(range(steps)) and all(f['all_finite'] and f['device']==device and f['shape']==[B,32003] for f in frows)
 complete=[x for x in progress if x['group']==group_number and x.get('batch_complete')];assert len(complete)==1 and complete[0]['decoder_steps_completed']==steps and complete[0]['indices']==indices
 for j,index in enumerate(indices):
  toks=generated[j].tolist();e=next(e for e in selected if e['index']==index);text=rust.decode(toks,skip_special_tokens=True).strip();ends=[i for i,t in enumerate(toks) if t in [2,32002]];active=ends[0]+1 if ends else len(toks)
  assert e['actual_group_row']==j and e['raw_output']==text and e['generation_token_ids']==toks and e['full_decoded_output_including_special_tokens']==rust.decode(toks,skip_special_tokens=False) and e['first_EOS_index']==(ends[0] if ends else None) and e['active_generation_count']==active and e['truncated']==(active>=2048) and e['error'] is None and e['actual_visual_injection_verified']
  all_samples[index]=e
 total_steps+=steps
correct=0;strict_correct=0;empty=[];invalid=[];truncated=[];sample_metrics=[];metric_fallback_credits=[]
for i,(a,b,r,e) in enumerate(zip(inputs,outputs,details,trace)):
 assert e==all_samples[i]==r['extra']['execution'] and r['prompt']==a['prompt']==original_clean_prompt(data[i]['prompt']) and r['reference']==a['reference']==b['reference']==data[i]['reference'] and r['raw_output']==b['raw_output']==e['raw_output'] and b['error'] is None
 choice=independent_choice(e['raw_output']);prediction=choice if choice is not None else e['raw_output'];assert r['prediction']==b['prediction']==prediction
 strict_metric=int(prediction==r['reference']);strict_correct+=strict_metric;metric=original_accuracy_sample(prediction,r['reference']);reason=safe_credit_reason(e['raw_output'],choice,r['reference'],metric);correct+=metric;sample_metrics.append(metric)
 if metric!=strict_metric:metric_fallback_credits.append({'index':i,'raw_output':e['raw_output'],'algorithm_prediction':prediction,'reference':r['reference'],'original_Accuracy_credit':metric,'strict_v2_label_equality_credit':strict_metric,'credit_reason':reason})
 if not e['raw_output']:empty.append(i)
 if choice is None:invalid.append(i)
 if e['truncated']:truncated.append(i)
assert empty==[i for i,e in enumerate(trace) if e['empty']] and invalid==[i for i,e in enumerate(trace) if e['invalid']]
score=correct/count;assert report['metrics']['Accuracy']==score and report['summary']['primary_metric']=='Accuracy' and report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and report['statistics']['batch_size']==10
prof=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hardware={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernels=[x for x in events if x.get('pid') in hardware and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
assert len(external)==len(groups)*3+7
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score,'metrics':{'Accuracy':score},'sample_Accuracy':sample_metrics,'historical_record_reuse':False,'original_protocol':'BF16 original greedy generation max2048, true insertion-ordered size groups inside algorithm batches10, own5 images/320tokens/untied32003head/left padding/native cacheTrue','original_source_cast_NPU_tensor_SHA_count':763,'actual_activation_module_count':63,'actual_generation_group_count':len(groups),'actual_generation_group_sizes':dict(collections.Counter(map(len,groups))),'actual_full_vocab_next_token_rows_independently_replayed':sum(len(groups[k])*json.loads(pathlib.Path(next(e for e in trace if e['actual_group']==k+1)['group_evidence']).read_text())['decoder_steps'] for k in range(len(groups))),'actual_group_decoder_steps':total_steps,'actual_CPU_original_inputs_replayed':count,'actual_visual_injection_verified':count,'actual_positions_masks_native_cache_replayed':total_steps,'native_RoPE_CPU_max_difference':max(rope_differences),'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'report_sha256':{f.name:sha(f) for f in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_numerical_review_compatibility_only_not_approved':True}

if 'native_numerical_review' in kw:
 assert sha(kw['native_numerical_review'])==kw['native_numerical_review_sha256']
 nr=json.loads(pathlib.Path(kw['native_numerical_review']).read_text());assert nr==line['native_numerical_review'] and nr['model_id']==p['model_id'] and nr['passed_diagnostic_integrity'] and nr['CPU_and_NPU8_FP32_diagnostic_only'] and nr['official_inference_remains_original_BF16_eager_cacheTrue'] and nr['no_activation_replacement'] and nr['all_control_CPU_NPU_parameter_SHA_equal'] and nr['same_original_BF16_rounded_weights_pixels_and_original_BF16_vision_position_ids'] and nr['control_parameter_count']==472 and nr['actual_5_images_per_sample']
 assert len(nr['SearchSorted_already_BF16_rounded_coordinates_CPU_equivalence'])==5 and all(x['bitwise_equal'] and x['original_CPU_BF16_output_sha256']==x['widened_original_BF16_coords_CPU_output_sha256'] for x in nr['SearchSorted_already_BF16_rounded_coordinates_CPU_equivalence'])
 diagnostic=dict(nr['evidence_sha256']);diagnostic[kw['native_numerical_review']]=kw['native_numerical_review_sha256']
 for comparison in nr['comparisons']:diagnostic[comparison['official_real3_external_NPU_evidence']]=comparison['official_real3_external_NPU_evidence_sha256']
 for path,digest in diagnostic.items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
 for name,digest in nr['original_real3_file_sha256'].items():assert sha(pathlib.Path(nr['original_real3_report'])/name)==digest
 assert line['native_BF16_computation_unchanged'] and line['prior_real3_CPU_diagnostic_reused_not_recomputed_per_report']
 proof.update(native_numerical_review=nr,native_numerical_review_sha256=kw['native_numerical_review_sha256'],native_BF16_numerical_difference_retained_and_disclosed=True,idefics2_grouped_numerical_diagnostic_evidence_sha256=diagnostic,prior_real3_CPU_diagnostic_reused_not_official_inference=True,pre_numerical_review_compatibility_only_not_approved=False)


metric_source=pathlib.Path('/root/projects/comperror/universal_eval/metric/accuracy.py')
proof.update(original_Accuracy_source_sha256=sha(metric_source),original_Accuracy_independent_normalization_and_ordered_extraction_replayed=True,original_Accuracy_fallback_credits=metric_fallback_credits,strict_v2_label_equality_Accuracy=strict_correct/count,strict_v2_extractor_invalid_indices_preserved=invalid,no_ordinary_article_credit_accepted=True,inference_reused_for_original_Accuracy_metric_verifier_repair=('--reuse-original-record-for-Accuracy-verifier-repair' in sys.argv[3:]),native_model_inference_not_repeated=('--reuse-original-record-for-Accuracy-verifier-repair' in sys.argv[3:]),historical_record_reuse=('--reuse-original-record-for-Accuracy-verifier-repair' in sys.argv[3:]))

dest=weights_path.parent/('independent_report_verification_accuracy_v3.json' if count==3 and '--reuse-original-record-for-Accuracy-verifier-repair' in sys.argv[3:] else 'independent_report_verification.json');assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))

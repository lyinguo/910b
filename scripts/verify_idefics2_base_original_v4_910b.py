"""Independent full-source, original pixel/token/position/visual/full-vocab and Accuracy audit."""
import pathlib,sys,os,json,re,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from idefics2_base_independent_cpu_v3_910b import *
from transformers import AutoConfig,Idefics2ForConditionalGeneration
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
torch.set_num_threads(4)
def check(value,rec):assert list(value.shape)==rec['shape'] and str(value.dtype)==rec['dtype'] and fp(value)==rec['sha256']
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256'];p=json.loads(pathlib.Path(kw['protocol_file']).read_text());wp=pathlib.Path(cfg['model']['model_path'])
assert cfg['model']['name']==p['model_id']=='HuggingFaceM4/idefics2-8b-base' and str(wp)==p['weight_path'] and folder.parent.name==cfg['task_type']=='image-text-to-text' and folder.parent.parent.name==p['model_id'].replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['batch_size']==8 and cfg['algorithm']['kwargs']=={'extract_answer':False} and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False and cfg['metrics']['override']==['Accuracy'] and cfg['metrics']['include_failed_predictions'] is True and report['summary']['primary_metric']=='Accuracy'
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];assert line['protocol']=='idefics2_base_original_first_letter_BF16_log_softmax' and line['protocol_sha256']==kw['protocol_sha256'] and line['all_expected_parameters_loaded'] and line['independent_lm_head_preserved'] and line['source_dtype']=='F32' and line['runtime_dtype']=='BF16' and line['native_attention_implementation']=='eager' and line['original_cache'] is True and line['original_algorithm_batch_size']==8 and line['actual_inference_batch_size']==1 and line['image_splitting'] is False and line['model_vocab_size']==32002 and line['image_seq_len']==64 and line['HF32_enabled']=={'conv':False,'matmul':False}
device='npu:'+str(kw['npu_devices'][0]);assert line['devices']==kw['npu_devices'] and set(line['module_placements'].values())=={device}
for key in ['reference_report','reference_log','dataset_path','original_dataset','historical_transformers_before_run']:assert sha(p[key])==p[key+'_sha256']
for source,digest in dict(p['runtime_sha256'],**p['historical_sources']).items():assert sha(source)==digest
for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
old=json.loads(pathlib.Path(p['reference_report']).read_text());assert len(old['detailed_results'])==200 and old['metrics']['Accuracy']==.605 and old['pipeline_config']['algorithm']['name']=='mc_score' and old['pipeline_config']['algorithm']['batch_size']==8 and old['pipeline_config']['model']['kwargs']['torch_dtype']=='bfloat16'
assert 'batch_size=8' in pathlib.Path(p['reference_log']).read_text() and '成功 200 条, 失败 0 条' in pathlib.Path(p['reference_log']).read_text()
data=[json.loads(x) for x in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert len(data)==4329
original_correct=0
for i,r in enumerate(old['detailed_results']):
 assert r['index']==i and r['prompt']==original_clean_prompt(data[i]['prompt']) and r['reference']==data[i]['reference'];s=r['extra']['scores'];best=max(s,key=s.get);assert best==r['prediction']==r['raw_output'];original_correct+=best==r['reference']
assert original_correct==121
mc=AutoConfig.from_pretrained(wp,local_files_only=True);assert mc.tie_word_embeddings is False and mc.text_config.vocab_size==32002 and mc.text_config.use_cache is True
with torch.device('meta'):meta=Idefics2ForConditionalGeneration(mc)
expected=dict(meta.named_parameters());assert len(expected)==763
expected_activations={n:type(m).__name__ for n,m in meta.named_modules() if type(m).__name__ in ['GELUTanh','SiLUActivation']};assert len(expected_activations)==63 and sum(x=='GELUTanh' for x in expected_activations.values())==27 and sum(x=='SiLUActivation' for x in expected_activations.values())==36 and line['activation_modules']==expected_activations
weights_path=pathlib.Path(line['weight_loading_evidence']);loading=json.loads(weights_path.read_text());mapping={r['source_key']:r for r in p['source_tensor_records']};assert len(loading)==763==line['weight_tensor_count'] and set(mapping)==set(expected)=={r['source_key'] for r in loading}
actual_rows={r['source_key']:r for r in loading};embedding=None;head=None
for shard,header in p['weight_headers'].items():
 assert sha(wp/shard)==header['file_sha256']
 with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
  for key in f.keys():
   value=f.get_tensor(key);cast=value.bfloat16();r=actual_rows[key];m=mapping[key]
   assert value.dtype==torch.float32 and r['source_dtype']=='torch.float32' and r['runtime_dtype']=='torch.bfloat16' and r['device']==device and key==r['target_key'] and list(value.shape)==r['shape']==list(expected[key].shape)
   assert fp(value)==r['source_sha256']==m['source_sha256'] and fp(cast)==r['actual_NPU_sha256']==r['original_BF16_cast_sha256']==m['original_BF16_cast_sha256'] and torch.isfinite(cast).all()
   if key=='model.text_model.embed_tokens.weight':embedding=cast.clone()
   if key=='lm_head.weight':head=cast.clone()
assert embedding is not None and head is not None and not torch.equal(embedding,head)
assert sum(r['nonzero_underflows'] for r in loading)==p['source_underflow_total']==line['source_underflow_total'] and sum(r['source_roundtrip_changed_elements'] for r in loading)==p['source_BF16_roundtrip_changed_elements']==line['source_BF16_roundtrip_changed_elements']
rope=MistralRotaryEmbedding(mc.text_config,device='cpu');check(rope.inv_freq,line['text_inv_freq'])
inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];trace_path=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(x) for x in trace_path.read_text().splitlines()];apath=pathlib.Path(line['activation_execution_trace']);activations=[json.loads(x) for x in apath.read_text().splitlines()]
assert len(inputs)==len(outputs)==len(details)==len(trace)==len(activations)==count
for rows in [inputs,outputs,details]:assert [r['index'] for r in rows]==list(range(count))
assert [r['attempt'] for r in trace]==[r['attempt'] for r in activations]==list(range(1,count+1))
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'] and probe['device']==device;torch.set_num_threads(probe['CPU_threads'])
with np.load(probe['path'],allow_pickle=False) as z:
 pv=torch.from_numpy(z['input_BF16_bytes'].copy()).view(torch.bfloat16)
 for approximate in ['none','tanh']:
  cpu=torch.nn.functional.gelu(pv,approximate=approximate);native=torch.from_numpy(z['NPU_'+approximate+'_BF16_bytes'].copy()).view(torch.bfloat16);assert torch.equal(cpu,torch.from_numpy(z['CPU_'+approximate+'_BF16_bytes'].copy()).view(torch.bfloat16)) and torch.isfinite(native).all() and float((native-cpu).abs().max())==probe[approximate]['max_abs_CPU_same_declared_GELU']
torch.set_num_threads(4)
external={str(trace_path):sha(trace_path),str(apath):sha(apath),str(weights_path):sha(weights_path),line['profiler_trace']:sha(line['profiler_trace']),probe['path']:sha(probe['path'])}
types={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64,'torch.bool':torch.bool}
correct=0;sample_metrics=[];softmax_diagnostics=[];rope_diagnostics=[]
for i,(a,b,r,e,act) in enumerate(zip(inputs,outputs,details,trace,activations)):
 assert r['extra']['execution']==e and r['prompt']==a['prompt']==original_clean_prompt(data[i]['prompt']) and r['reference']==a['reference']==b['reference']==data[i]['reference'] and e['prompt']==r['prompt'] and e['error'] is None and b['error'] is None and not e['truncated'] and e['no_generation'] and e['generation_count']==0 and e['generation_token_ids']==[]
 media=(pathlib.Path(p['dataset_path']).parent/data[i]['media_path']).resolve();assert sha(media)==p['reference_first200_image_sha256'][str(media)]==e['source_image_sha256'] and sha(old['detailed_results'][i]['media_path'])==sha(media)
 own=independent_inputs(wp,r['prompt'],media);assert own['input_ids'].tolist()==e['input_ids'] and own['attention_mask'].tolist()==e['attention_mask'] and own['prefix']==e['prefix_text']
 for key in ['input_ids','attention_mask','pixel_values','pixel_attention_mask']:check(own[key],e['CPU_inputs'][key])
 archive=pathlib.Path(e['full_vocab_and_visual_archive']);assert sha(archive)==e['full_vocab_and_visual_archive_sha256'];external[str(archive)]=sha(archive)
 with np.load(archive,allow_pickle=False) as z:
  assert set(z.files)==set(e['tensor_records']);tensors={k:torch.from_numpy(z[k].copy()).view(types[rec['dtype']]).reshape(rec['shape']) for k,rec in e['tensor_records'].items()}
 for key,t in tensors.items():check(t,e['tensor_records'][key]);assert torch.isfinite(t).all()
 assert torch.equal(tensors['actual_vision_pixels'],own['pixel_values'].bfloat16().reshape(-1,*own['pixel_values'].shape[-3:]))
 patch,positions=independent_vision_positions(own['pixel_attention_mask']);assert torch.equal(tensors['actual_patch_attention_mask'],patch) and torch.equal(tensors['actual_vision_positions'],positions)
 assert tensors['actual_connector'].shape==(1,64,4096) and tensors['actual_encoder'].shape==(1,patch.shape[1]*patch.shape[2],1152) and torch.equal(tensors['actual_connector'].reshape(64,4096),tensors['actual_injected_visual']) and tensors['actual_connector'].abs().sum()>0
 ids=own['input_ids'];merged=tensors['actual_text_inputs_embeds'];assert merged.shape==(1,ids.shape[1],4096) and torch.equal(merged[ids==32001],tensors['actual_injected_visual']) and torch.equal(merged[ids!=32001],embedding[ids[ids!=32001]])
 length=ids.shape[1];assert tensors['actual_text_positions'].tolist()==[list(range(length))] and e['original_full_forward_logit_shape']==[1,length,32002]
 mask=tensors['actual_text_causal_mask'];assert mask.shape==(1,1,length,length)
 allowed=torch.ones((length,length),dtype=torch.bool).tril()
 if mc.text_config.sliding_window is not None:allowed &= torch.arange(length)[:,None]-torch.arange(length)[None,:]<mc.text_config.sliding_window
 expected_mask=torch.where(allowed,torch.tensor(0.,dtype=mask.dtype),torch.tensor(torch.finfo(mask.dtype).min,dtype=mask.dtype));assert torch.equal(mask[0,0],expected_mask)
 cos,sin=rope(torch.zeros((1,length,4096),dtype=torch.bfloat16),torch.arange(length)[None]);cosdiff=float((cos-tensors['actual_rope_cos']).abs().max());sindiff=float((sin-tensors['actual_rope_sin']).abs().max());assert max(cosdiff,sindiff)<=torch.finfo(torch.bfloat16).eps;rope_diagnostics.append({'index':i,'CPU_vs_native_BF16_cos_max':cosdiff,'CPU_vs_native_BF16_sin_max':sindiff})
 logits=tensors['actual_last_logits'];lp=tensors['actual_BF16_log_softmax'];assert logits.shape==lp.shape==(32002,) and logits.dtype==lp.dtype==torch.bfloat16 and e['BF16_log_softmax'] and e['full_vocab_size']==32002
 cpu=torch.log_softmax(logits,dim=-1);explicit=logits.double()-torch.logsumexp(logits.double(),dim=-1);rounded=explicit.bfloat16()
 diag={'index':i,'native_vs_CPU_BF16_max':float((lp-cpu).abs().max()),'native_vs_CPU_BF16_different_elements':int((lp!=cpu).sum()),'native_vs_correctly_rounded_FP64_max':float((lp-rounded).abs().max()),'actual_candidate_scores':e['candidate_scores'],'CPU_BF16_candidate_scores':{k:float(cpu[t].item()) for k,t in own['candidate_first_token_ids'].items()}}
 softmax_diagnostics.append(diag)
 maximum=float(logits.max());normalizer=maximum+math.log(math.fsum(math.exp(float(x)-maximum) for x in logits.float().tolist()));independent=torch.tensor([float(x)-normalizer for x in logits.float().tolist()],dtype=torch.float64).bfloat16()
 assert torch.equal(lp,independent),dict(diag,independent_FP64_fsum_max=float((lp-independent).abs().max()))
 diag['actual_native_equals_independent_FP64_fsum_then_BF16_bitwise']=True
 diag['independent_FP64_fsum_then_BF16_candidate_scores']={k:float(independent[t].item()) for k,t in own['candidate_first_token_ids'].items()}
 scores={k:float(lp[t].item()) for k,t in own['candidate_first_token_ids'].items()};assert scores==e['candidate_scores']==r['extra']['scores'] and own['candidate_first_token_ids']==e['candidate_first_token_ids'] and options(r['prompt'])==list(scores)
 best=max(scores,key=scores.get);assert best==e['raw_output']==r['prediction']==r['raw_output']==b['prediction']==b['raw_output'];correct+=best==r['reference'];sample_metrics.append(int(best==r['reference']))
 assert e['actual_vision_calls']==e['actual_connector_calls']==e['actual_text_model_calls']==1 and e['actual_visual_injection_verified'] and act['module_call_counts']==e['native_activation_call_counts'] and set(e['native_activation_call_counts'])==set(line['activation_modules']) and set(e['native_activation_call_counts'].values())=={1} and act['device']==device and act['error'] is None
 assert e['final_cache_length']==length and len(e['cache_device_and_shapes'])==32
 for j,slot in enumerate(e['cache_device_and_shapes']):assert slot['slot']==j and slot['device']==device and slot['dtype']=='torch.bfloat16' and slot['key_shape']==slot['value_shape']==[1,8,length,128]
assert report['statistics']['total_samples']==count and report['statistics']['error_count']==0 and line['actual_inference_batch_size']==1 and report['statistics']['batch_size']==8 and report['statistics']['algorithm']=='mc_score' and cfg['algorithm']['kwargs']['extract_answer'] is False
score=correct/count;assert report['metrics']['Accuracy']==score
prof=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'Accuracy':score,'metrics':{'Accuracy':score},'sample_Accuracy':sample_metrics,'historical_record_reuse':False,'original_protocol':'mc_score single forward raw BF16 log_softmax of first candidate letter token; batch metadata8, actual serial1','source_dtype':'F32','runtime_dtype':'BF16','full_original_source_cast_NPU_tensor_SHAs_verified':763,'independent_lm_head_preserved':True,'original200_recomputed_Accuracy':.605,'independent_CPU_pixels_tokens_masks_replayed':count,'actual_visual_connector_injection_verified':count,'complete_NPU_vocab_and_BF16_log_softmax_CPU_replays':count,'full_vocab_size':32002,'native_GELU_probe':probe,'native_activation_execution_replayed':count,'actual_activation_module_count':63,'actual_vision_GELUTanh_modules':27,'actual_connector_and_language_SiLU_modules':36,'original_position_mask_cache_replays':count,'original_source_underflow_total':line['source_underflow_total'],'original_source_BF16_roundtrip_changed_elements':line['source_BF16_roundtrip_changed_elements'],'native_log_softmax_diagnostics':softmax_diagnostics,'native_rope_diagnostics':rope_diagnostics,'error_indices':[],'empty_indices':[],'invalid_indices':[],'truncated_indices':[],'index_duplicates':False,'NPU_profiler_kernel_events':len(kernels),'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure']}
proof['full_vocab_reference']='independent FP64 max-shift exp/math.fsum/log/subtract then original BF16 rounding; exact bitwise comparison to actual NPU BF16. CPU torch BF16 kernel retained as a separate numerical diagnostic.'
proof['inference_reused_for_independent_reference_repair']='--reuse-original-real3' in sys.argv[3:]
proof['inference_reused_for_verifier_schema_repair']='--reuse-original-real3-for-schema-repair' in sys.argv[3:]
proof['historical_record_reuse']=proof['inference_reused_for_independent_reference_repair'] or proof['inference_reused_for_verifier_schema_repair']
proof['existing_report_statistics_schema_verified']=True
proof['actual_batch_and_extraction_verified_in_config_lineage_and_all_sample_tensors']=True
if 'native_numerical_review' in kw:
 assert sha(kw['native_numerical_review'])==kw['native_numerical_review_sha256'] and sha(kw['log_softmax_reference_review'])==kw['log_softmax_reference_review_sha256']
 nr=json.loads(pathlib.Path(kw['native_numerical_review']).read_text());lr=json.loads(pathlib.Path(kw['log_softmax_reference_review']).read_text())
 assert nr==line['native_numerical_review'] and lr==line['log_softmax_reference_review'] and nr['passed_diagnostic_integrity'] and nr['CPU_and_NPU6_FP32_diagnostic_only'] and nr['official_inference_remains_original_BF16_eager_cacheTrue'] and nr['no_activation_replacement'] and nr['all_control_CPU_NPU_parameter_SHA_equal'] and nr['same_original_BF16_rounded_weights_pixels_and_original_BF16_vision_position_ids'] and nr['control_parameter_count']==472
 assert len(nr['SearchSorted_already_BF16_rounded_coordinates_CPU_equivalence'])==5 and all(x['bitwise_equal'] and x['original_CPU_BF16_output_sha256']==x['widened_original_BF16_coords_CPU_output_sha256'] for x in nr['SearchSorted_already_BF16_rounded_coordinates_CPU_equivalence'])
 diagnostics=dict(nr['evidence_sha256']);diagnostics[kw['native_numerical_review']]=kw['native_numerical_review_sha256'];diagnostics[kw['log_softmax_reference_review']]=kw['log_softmax_reference_review_sha256']
 for path,digest in diagnostics.items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
 for name,digest in nr['original_real3_file_sha256'].items():assert sha(pathlib.Path(nr['original_real3_report'])/name)==digest==lr['original_real3_file_sha256'][name]
 assert lr['passed'] and lr['original_extra_real3_NPU_inference_unchanged'] and all(x['actual_native_BF16_vs_independent_FP64_fsum_then_BF16_bitwise_equal'] for x in lr['comparisons']) and line['native_BF16_computation_unchanged'] and line['prior_real3_CPU_diagnostic_reused_not_recomputed_per_report']
 assert nr['original_vision_activation']=='gelu_pytorch_tanh'
 proof.update(native_numerical_review=nr,native_numerical_review_sha256=kw['native_numerical_review_sha256'],log_softmax_reference_review=lr,native_BF16_numerical_difference_retained_and_disclosed=True,idefics2_base_numerical_diagnostic_evidence_sha256=diagnostics,prior_real3_CPU_diagnostic_reused_not_official_inference=True)
else:proof['pre_disclosure_real3_compatibility_only_not_approved']=True

dest=weights_path.parent/'independent_report_verification.json';assert not dest.exists();dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2));print(json.dumps(proof,ensure_ascii=False,indent=2))

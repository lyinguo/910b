"""Independent own-input, full-weight, native-vision and full-vocab Gemma3 score audit."""
import pathlib,sys,os,json,re,math,hashlib,collections
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from gemma3_answer_score_independent_cpu_910b import sha,fp,tr,independent_inputs
from transformers import AutoConfig,Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
torch.set_num_threads(4)
def check(value,record):assert tr(value)==record
def clean(s):
 s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',str(s));s=re.sub(r'[ \t]+',' ',s);return re.sub(r'<image\d+>\s*','',re.sub(r'\n{3,}','\n\n',s).strip())
def options(s):return list(dict.fromkeys(m.upper() for m in re.findall(r'(?:^|\n)\s*([A-Ea-e])\.\s+[^\n]*',s))) or list('ABCD')
def mapped(k):
 for old,new in [('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.'),('vision_tower.vision_model.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.')]:
  if k.startswith(old):return new+k[len(old):]
 raise AssertionError(k)
def arrays(path,records):
 out={}
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(records)
  for name,r in records.items():
   b=np.ascontiguousarray(z[name]);assert b.dtype==np.uint8 and hashlib.sha256(b.tobytes()).hexdigest()==r['sha256'];x=torch.from_numpy(b.copy()).view({'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64,'torch.bool':torch.bool}[r['dtype']]).reshape(r['shape']);check(x,r);out[name]=x
 return out
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {q.name for q in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256'];p=json.loads(pathlib.Path(kw['protocol_file']).read_text());wp=pathlib.Path(cfg['model']['model_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution']
assert cfg['model']['name']==p['model_id'] in ['qingy2024/GRMR-V3-G4B','OddTheGreat/Meteor_4B_V.1'] and str(wp)==p['weight_path'] and folder.parent.parent.name==p['model_id'].replace('/','_') and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['algorithm']['name']=='gemma3_original_answer_score' and cfg['algorithm']['adapter']=='universal_eval/algorithm/gemma3_original_score_evidence_910b.py:Gemma3OriginalScoreEvidence910B' and cfg['algorithm']['batch_size']==1 and cfg['algorithm']['kwargs']=={'extract_answer':False} and cfg['metrics']['override']==['Accuracy'] and cfg['metrics']['include_failed_predictions'] and cfg['data']['filter_empty'] is False and cfg['data']['preprocess'] is True and cfg['data']['apply_template'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert line['protocol']=='gemma3_original_three_variant_first_token_FP32_log_softmax' and line['protocol_sha256']==kw['protocol_sha256'] and line['runtime_dtype']=='BF16' and line['original_tied_output_head_preserved'] and line['all_expected_parameters_loaded'] and line['native_attention_implementation']=='sdpa' and line['original_cache'] and line['original_algorithm_batch_size']==line['actual_inference_batch_size']==1 and line['image_seq_len']==256 and line['model_vocab_size']==262208 and line['HF32_enabled']=={'conv':False,'matmul':False}
assert p['original_score_target']==line['original_score_target']=='letter' and p['original_score_tail']==line['original_score_tail']=='\n\nAnswer:' and p['original_use_chat_template']==line['original_use_chat_template'] and not p['effective_original_chat_template_used'] and not line['effective_original_chat_template_used']
device='npu:'+str(kw['npu_devices'][0]);assert line['devices']==kw['npu_devices'] and set(line['module_placements'].values())=={device}
for source,digest in dict(p['runtime_sha256'],**p['historical_sources'],**p['evidence_sha256']).items():assert sha(source)==digest
for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(wp/name)==digest
assert sha(p['dataset_path'])==p['dataset_sha256']==sha(p['original_dataset'])==p['original_dataset_sha256']
old=json.loads(pathlib.Path(p['reference_report']).read_text());assert sha(p['reference_report'])==p['reference_report_sha256'] and sha(p['reference_log'])==p['reference_log_sha256'] and '成功 200 条, 失败 0 条' in pathlib.Path(p['reference_log']).read_text();original=old['detailed_results'];data=[json.loads(x) for x in pathlib.Path(p['dataset_path']).read_text().splitlines()];assert len(data)==4329 and len(original)==len(p['all200_original_cpu_inputs'])==200
original_correct=0
for i,r in enumerate(original):
 assert r['index']==i and r['prompt']==clean(data[i]['prompt']) and r['reference']==data[i]['reference'];scores=r['extra']['scores'];assert list(scores)==options(r['prompt']) and max(scores,key=scores.get)==r['prediction'];original_correct+=r['prediction']==r['reference']
assert original_correct/200==p['original_report_Accuracy']==old['metrics']['Accuracy']
mc=AutoConfig.from_pretrained(wp,local_files_only=True);assert mc.tie_word_embeddings and mc.text_config.use_cache and mc.text_config.vocab_size==262208 and mc.mm_tokens_per_image==256
with torch.device('meta'):meta=Gemma3ForConditionalGeneration(mc)
expected=dict(meta.named_parameters(remove_duplicate=False));assert len(dict(meta.named_parameters()))==883
expected_act={n:type(m).__name__ for n,m in meta.named_modules() if type(m).__name__=='GELUTanh'};assert len(expected_act)==61 and expected_act==line['activation_modules']
loading_path=pathlib.Path(line['weight_loading_evidence']);loading=json.loads(loading_path.read_text());loaded={r['source_key']:r for r in loading};source_records={r['source_key']:r for r in p['source_tensor_records']};assert len(loading)==len(source_records)==line['weight_tensor_count']==883
cast_hashes={};underflow=0;changed=0
for shard,header in p['weight_headers'].items():
 assert (wp/shard).stat().st_size==header['size'] and sha(wp/shard)==header['file_sha256']
 with safe_open(str(wp/shard),framework='pt',device='cpu') as f:
  for key in f.keys():
   x=f.get_tensor(key);cast=x.bfloat16();r=loaded[key];original_r=source_records[key];name=mapped(key);assert name==r['target_key']==original_r['target_key'] and name in expected and list(x.shape)==r['shape']==list(expected[name].shape) and str(x.dtype)==r['source_dtype']==original_r['source_dtype']
   assert fp(x)==r['source_sha256']==original_r['source_sha256'] and fp(cast)==r['actual_NPU_sha256']==r['original_BF16_cast_sha256']==original_r['original_BF16_cast_sha256'] and r['device']==device and r['runtime_dtype']=='torch.bfloat16' and torch.isfinite(cast).all();cast_hashes[name]=fp(cast)
   actual_underflow=int(((x!=0)&(cast==0)).sum());actual_changed=int((cast.to(x.dtype)!=x).sum());assert actual_underflow==r['nonzero_underflow'] and actual_changed==r['original_BF16_roundtrip_changed_elements'];underflow+=actual_underflow;changed+=actual_changed
assert underflow==p['source_to_BF16_underflow_total']==line['source_underflow_total'] and changed==p['source_BF16_roundtrip_changed_elements']==line['source_BF16_roundtrip_changed_elements']
for group in p['native_alias_groups']:assert len({cast_hashes[name] for name in group if name in cast_hashes})==1 and any(name in cast_hashes for name in group)
rope=Gemma3RotaryEmbedding(mc.text_config)
for name,value in rope.named_buffers():check(value,line['text_rotary_buffers'][name]);assert value.dtype==torch.float32
check(torch.tensor(math.sqrt(mc.text_config.hidden_size),dtype=torch.bfloat16),line['original_embedding_scale'])
ins=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()];outs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];trace_path=pathlib.Path(line['per_sample_evidence']);trace=[json.loads(x) for x in trace_path.read_text().splitlines()];act_path=pathlib.Path(line['activation_execution_trace']);acts=[json.loads(x) for x in act_path.read_text().splitlines()]
assert len(ins)==len(outs)==len(details)==len(trace)==len(acts)==count
for rows in [ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
assert [r['attempt'] for r in trace]==[r['attempt'] for r in acts]==list(range(1,count+1))
external={str(q):sha(q) for q in [trace_path,act_path,loading_path,pathlib.Path(line['profiler_trace']),pathlib.Path(line['native_GELU_probe']['path'])]}
correct=0;sample_scores=[];softmax_diagnostics=[];rope_diagnostics=[];ties=[]
for i,(a,b,r,e,act) in enumerate(zip(ins,outs,details,trace,acts)):
 assert a['prompt']==r['prompt']==e['prompt']==clean(data[i]['prompt']) and a['reference']==b['reference']==r['reference']==data[i]['reference'] and e['error'] is None and b['error'] is None and r['extra']['execution']==e and not e['truncated'] and not e['empty'] and not e['invalid'] and e['no_generation'] and e['generation_count']==0 and e['generation_token_ids']==[]
 media=(pathlib.Path(p['dataset_path']).parent/data[i]['media_path']).resolve();assert sha(media)==p['reference_first200_image_sha256'][str(media)]==e['source_image_sha256'] and sha(original[i]['media_path'])==sha(media)
 own,rendered,rust=independent_inputs(wp,r['prompt'],media);assert e['rendered_text']==rendered and own['input_ids'].tolist()==e['input_ids'] and own['attention_mask'].tolist()==e['attention_mask'] and own['token_type_ids'].tolist()==e['token_type_ids']
 for name,value in own.items():check(value,e['CPU_inputs'][name])
 archive=pathlib.Path(e['full_vocab_and_visual_archive']);assert sha(archive)==e['full_vocab_and_visual_archive_sha256'];external[str(archive)]=sha(archive);t=arrays(archive,e['tensor_records']);ids=own['input_ids'];length=ids.shape[1];assert e['original_full_forward_logit_shape']==[1,length,262208] and e['actual_head_forward_shapes']=={'input':[1,length,mc.text_config.hidden_size],'output':[1,length,262208]}
 assert t['actual_encoder'].shape==(1,4096,1152) and t['actual_connector'].shape==(1,256,mc.text_config.hidden_size) and t['actual_text_inputs_embeds'].shape==(1,length,mc.text_config.hidden_size) and torch.equal(t['actual_vision_pixels'],own['pixel_values'].bfloat16()) and torch.equal(t['actual_vision_positions'],torch.arange(4096)[None])
 visual_mask=ids==mc.image_token_id;assert visual_mask.sum()==256 and torch.equal(t['actual_injected_visual'],t['actual_connector'].reshape(-1,mc.text_config.hidden_size)) and torch.equal(t['actual_text_inputs_embeds'][visual_mask],t['actual_injected_visual'])
 assert torch.equal(t['actual_text_positions'],torch.arange(length)[None]);q=torch.arange(length)[:,None];k=torch.arange(length)[None,:];block=visual_mask[0,:,None]&visual_mask[0,None,:];base=(q>=k)|block
 for layer_type in ['full_attention','sliding_attention']:
  allowed=base if layer_type=='full_attention' else base & (q-k<mc.text_config.sliding_window);mask=t['actual_'+layer_type+'_mask'];assert mask.shape==(1,1,length,length)
  if mask.dtype==torch.bool:assert torch.equal(mask[0,0],allowed)
  else:assert torch.equal(mask[0,0],torch.where(allowed,torch.tensor(0.,dtype=mask.dtype),torch.tensor(torch.finfo(mask.dtype).min,dtype=mask.dtype)))
  cos,sin=rope(torch.zeros((1,length,mc.text_config.hidden_size),dtype=torch.bfloat16),torch.arange(length)[None],layer_type);dc=float((cos-t['actual_'+layer_type+'_cos']).abs().max());ds=float((sin-t['actual_'+layer_type+'_sin']).abs().max());assert max(dc,ds)<=torch.finfo(torch.bfloat16).eps;rope_diagnostics.append({'index':i,'layer_type':layer_type,'CPU_native_BF16_cos_max_abs':dc,'CPU_native_BF16_sin_max_abs':ds})
 assert len(e['actual_attention_layers'])==len(e['cache_device_and_shapes'])==34 and e['final_cache_length']==length
 for j,(attn,slot) in enumerate(zip(e['actual_attention_layers'],e['cache_device_and_shapes'])):
  layer_type=mc.text_config.layer_types[j];assert attn['layer']==slot['slot']==j and attn['layer_type']==layer_type and attn['position_ids']==[list(range(length))] and attn['mask']==e['tensor_records']['actual_'+layer_type+'_mask'] and attn['scaling']==mc.text_config.query_pre_attn_scalar**-.5 and attn['sliding_window']==(mc.text_config.sliding_window if layer_type=='sliding_attention' else None)
  n=min(length,mc.text_config.sliding_window-1) if layer_type=='sliding_attention' else length;assert slot['key_shape']==slot['value_shape']==[1,mc.text_config.num_key_value_heads,n,mc.text_config.head_dim] and slot['device']==device and slot['dtype']=='torch.bfloat16' and slot['cumulative_length']==length
 logits=t['actual_last_logits'];lp=t['actual_FP32_log_softmax'];assert logits.shape==lp.shape==(262208,) and logits.dtype==torch.bfloat16 and lp.dtype==torch.float32 and torch.isfinite(logits).all() and torch.isfinite(lp).all() and e['original_FP32_log_softmax'] and e['full_vocab_size']==262208
 # Reference arithmetic is independent FP64 max-shift/exp/fsum/log/subtract over the complete vocabulary.
 values=logits.float().numpy().astype(np.float64);maximum=float(values.max());normalizer=maximum+math.log(math.fsum(np.exp(values-maximum)));reference=torch.from_numpy(values-normalizer);rounded=reference.float();difference=(lp.double()-reference).abs();max_error=float(difference.max());bound=4*torch.finfo(torch.float32).eps*(float(abs(normalizer))+float(logits.float().abs().max())+1)
 assert max_error<=bound,('FP32 log_softmax exceeds forward-error bound',i,max_error,bound);assert int(lp.argmax())==int(reference.argmax())
 candidates={};variant_tokens={};independent_scores={};raw_scores={}
 for letter in options(r['prompt']):
  variants={v:rust.encode(v,add_special_tokens=False).ids for v in [letter,' '+letter,'\n'+letter]};assert all(variants.values());variant_tokens[letter]=variants;candidates[letter]=[x[0] for x in variants.values()];independent_scores[letter]=max(float(reference[j]) for j in candidates[letter]);raw_scores[letter]=max(float(logits[j]) for j in candidates[letter])
 actual_scores={letter:max(float(lp[j]) for j in token_ids) for letter,token_ids in candidates.items()};assert actual_scores==e['candidate_scores']==r['extra']['scores'] and candidates==e['candidate_first_token_ids'] and variant_tokens==e['full_candidate_variant_token_ids']
 selected=max(actual_scores,key=actual_scores.get);assert selected==max(independent_scores,key=independent_scores.get)==max(raw_scores,key=raw_scores.get)==e['raw_output']==r['prediction']==r['raw_output']==b['prediction']==b['raw_output']
 if list(actual_scores.values()).count(max(actual_scores.values()))>1:ties.append(i)
 correct+=selected==r['reference'];sample_scores.append(int(selected==r['reference']));softmax_diagnostics.append({'index':i,'actual_native_FP32_vs_independent_FP64_max_abs':max_error,'analytic_FP32_forward_error_bound':bound,'actual_native_FP32_vs_correctly_rounded_FP64_different_elements':int((lp!=rounded).sum()),'candidate_actual_FP32':actual_scores,'candidate_independent_FP64':independent_scores,'stable_argmax_same':True})
 assert e['actual_vision_calls']==e['actual_connector_calls']==e['actual_text_model_calls']==e['actual_head_calls']==1 and e['actual_visual_injection_verified'] and e['native_activation_call_counts']==act['module_call_counts'] and set(e['native_activation_call_counts'])==set(expected_act) and set(e['native_activation_call_counts'].values())=={1} and act['error'] is None and act['device']==device
probe=line['native_GELU_probe'];assert sha(probe['path'])==probe['sha256'] and probe['device']==device;torch.set_num_threads(probe['CPU_threads'])
with np.load(probe['path'],allow_pickle=False) as z:
 x=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(z['input_BF16_bytes'],x.view(torch.uint8).numpy())
 for mode in ['none','tanh']:
  cpu=torch.nn.functional.gelu(x,approximate=mode);native=torch.from_numpy(z['NPU_'+mode+'_BF16_bytes'].copy()).view(torch.bfloat16);assert torch.isfinite(native).all() and np.array_equal(z['CPU_'+mode+'_BF16_bytes'],cpu.view(torch.uint8).numpy()) and float((native-cpu).abs().max())==probe[mode]['max_abs_CPU_same_declared_GELU']
score=correct/count;assert report['metrics']['Accuracy']==report['summary']['current_score']==score and report['summary']['primary_metric']=='Accuracy' and report['statistics']['error_count']==0 and report['statistics']['total_samples']==count and report['statistics']['batch_size']==1 and report['statistics']['algorithm']=='mc_score' and len(external)==count+5
profile=json.loads(pathlib.Path(line['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert len(kernels)>100
proof={'passed':True,'report_dir':str(folder),'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Accuracy':score},'sample_Accuracy':sample_scores,'historical_record_reuse':False,'original_protocol':'Gemma3AnswerScore letter: full-position full-vocab BF16 forward; actual FP32 full-vocab log_softmax; maximum of first token from letter/space-letter/newline-letter then original insertion-order argmax. No generation.','original200_Accuracy_independently_recomputed':original_correct/200,'source_dtype_counts':p['source_dtype_counts'],'runtime_dtype':'BF16','full_original_source_cast_actual_NPU_SHA_tensors':883,'source_underflow_total':underflow,'source_BF16_roundtrip_changed_elements':changed,'original_tied_embedding_head_preserved':True,'independent_CPU_pixels_tokens_masks_replayed':count,'actual_encoder_connector_injection_verified':count,'native_full_position_head_GEMM_verified':count,'complete_vocabulary_FP32_log_softmax_independent_FP64_replays':count,'native_log_softmax_diagnostics':softmax_diagnostics,'native_rope_diagnostics':rope_diagnostics,'original_three_variant_max_and_stable_tie_verified':True,'tied_best_indices':ties,'full_vocab_size':262208,'actual_activation_module_count':61,'all61_activation_calls_each_sample_verified':True,'all34_original_positions_full_and_sliding_masks_native_cache_verified':count,'NPU_profiler_kernel_events':len(kernels),'error_indices':[],'empty_indices':[],'invalid_indices':[],'truncated_indices':[],'index_duplicates':False,'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'protocol_file':kw['protocol_file'],'protocol_sha256':kw['protocol_sha256'],'precision_disclosure':line['precision_disclosure'],'pre_disclosure_real3_only_not_yet_approved':True}
if 'native_numerical_review' in kw:
 path=pathlib.Path(kw['native_numerical_review']);assert sha(path)==kw['native_numerical_review_sha256'];nr=json.loads(path.read_text());assert nr==line['native_numerical_review'] and nr['official_inference_remains_original_BF16_SDPA_full_forward_FP32_log_softmax'] and nr['CPU_FP32_controls_diagnostic_only'] and nr['all_control_CPU_NPU_parameter_SHA_equal'] and nr['native_operator_formulas_unchanged']
 evidence=dict(nr['evidence_sha256']);evidence[str(path)]=sha(path)
 for q,digest in evidence.items():assert pathlib.Path(q).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(q)==digest
 proof.update(pre_disclosure_real3_only_not_yet_approved=False,native_BF16_numerical_difference_retained_and_disclosed=True,native_numerical_review=nr,gemma3_answer_score_numerical_diagnostic_evidence_sha256=evidence,static_CPU_diagnostics_from_prior_real3_not_recomputed_per_report=True)
dest=loading_path.parent/'independent_report_verification.json';assert not dest.exists() or json.loads(dest.read_text())==proof
if not dest.exists():dest.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))

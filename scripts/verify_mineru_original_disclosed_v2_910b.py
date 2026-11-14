"""Independent full-denominator MinerU pixels, tokens, weights, true kernels and actual-logit audit."""
import os,pathlib,sys,json,hashlib,re,collections,datetime
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from mineru_original_independent_cpu_v2_910b import sha,fp,tr,clean,INSTRUCTION,manual_inputs,manual_mrope
from idefics2_original_accuracy_reference_910b import original_accuracy_sample
torch.set_num_threads(4)
def independent_choice(raw):
 s=str(raw).strip()
 for rule in [r'<([A-E])>',r'\[([A-E])\]',r'\(([A-E])\)',r'\*\*([A-E])\*\*',r'([A-E])']:
  m=re.fullmatch(rule,s,re.I)
  if m:return m[1].upper()
 m=re.fullmatch(r'<([A-E])>\s*([A-E])\s*</([A-E])>',s,re.I)
 if m and len({c.upper() for c in m.groups()})==1:return m[1].upper()
 for rule in [r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}']:
  a=re.findall(rule,s,re.I)
  if a:return a[-1].upper()
 prefixes=[r'(?i:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*)',r'(?i:(?:correct\s+)?option\s+(?:is\s+)?\s*)',r'(?i:(?:choose|select|pick)\s+)']
 for i,prefix in enumerate(prefixes):
  a=re.findall(prefix+r'[\'"(\[]\s*([A-Ea-e])\s*\.?\s*[\'"\])]',s)
  if a:return a[-1].upper()
  a=re.findall(prefix+(r'([A-Ea-e])' if i==1 else r'([A-E])')+r'(?=[\s.,:;!?]|$)',s)
  if a:return a[-1].upper()
 m=re.match(r'^([A-Ea-e])\s*[.)](?=\s|$)',s)
 if m:return m[1].upper()
 a=re.findall(r'(?m)^\s*([A-Ea-e])\s*$',s);return a[-1].upper() if a else None
def load_arrays(path,records):
 out={}
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(records)
  for name,r in records.items():
   b=np.ascontiguousarray(z[name]);assert b.dtype==np.uint8 and hashlib.sha256(b.tobytes()).hexdigest()==r['sha256']
   dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64,'torch.bool':torch.bool,'torch.float16':torch.float16}[r['dtype']];x=torch.from_numpy(b.copy()).view(dtype).reshape(r['shape']);out[name]=x
 return out
folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
required={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {p.name for p in folder.iterdir()}==required
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];lineage=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];params=cfg['model']['kwargs'];pfile=pathlib.Path(params['protocol_file']);assert sha(pfile)==params['protocol_sha256']==lineage['protocol_sha256'];p=json.loads(pfile.read_text());root=pathlib.Path(cfg['model']['model_path']).resolve();assert str(root)==p['weight_path']
assert cfg['model']['name']==p['model_id']=='opendatalab/MinerU2.5-Pro-2604-1.2B' and cfg['task_type']=='image-text-to-text' and cfg['model']['download'] is False
assert folder.parent.parent.name==p['model_id'].replace('/','_') and folder.parent.name==cfg['task_type'] and re.fullmatch(r'\d{8}_\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['data']['preprocess'] is True
assert cfg['algorithm']['adapter']=='universal_eval/algorithm/native_explicit_mc_v2.py:NativeExplicitMCV2' and cfg['algorithm']['batch_size']==1 and cfg['metrics']['override']==['Strict_MC_Accuracy'] and cfg['metrics']['include_failed_predictions'] is True
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
for path,h in dict(p['runtime_sha256'],**p['historical_sources'],**p['evidence_sha256']).items():assert sha(path)==h,path
for rel,h in p['resource_sha256'].items():assert sha(root/rel)==h
assert sha(root/'model.safetensors')==p['weight_file_sha256']
assert lineage['devices']==params['npu_devices'] and len(lineage['devices'])==1;device='npu:'+str(lineage['devices'][0]);assert set(lineage['module_placements'].values())=={device} and not any(lineage['HF32_enabled'].values())
assert lineage['all_expected_parameters_loaded'] and lineage['weight_tensor_count']==681 and lineage['original_tied_output_head_preserved'] and lineage['original_actual_batch_size']==1 and lineage['original_max_new_tokens']==512 and lineage['original_use_cache'] and lineage['original_resolved_attention']=={'root':'sdpa','vision':'sdpa','text':'sdpa'}
assert lineage['resolved_original_generation_config']==p['resolved_original_generation_config'] and lineage['own_generation_json_not_used_by_original_constructor']
assert p['resolved_original_generation_config']['eos_token_id']==151645 and p['resolved_original_generation_config']['pad_token_id']==151643 and p['resolved_original_generation_config']['repetition_penalty']==1 and p['resolved_original_generation_config']['forced_eos_token_id'] is None
vocab=Tokenizer.from_file(str(root/'tokenizer.json'));unmapped=[i for i in range(151936) if vocab.id_to_token(i) is None];assert unmapped==lineage['original_model_IDs_without_token_string'] and lineage['unmapped_padded_IDs_not_removed_resized_masked_or_filled'] and lineage['model_vocab_size']==151936
records=json.loads(pathlib.Path(lineage['weight_loading_evidence']).read_text());expected={r['source_key']:r for r in p['source_tensor_records']};assert len(records)==len(expected)==681 and len({r['target_key'] for r in records})==681
with safe_open(str(root/'model.safetensors'),framework='pt',device='cpu') as h:
 assert set(h.keys())==set(expected)
 for r in records:
  key=r['source_key'];assert {k:r[k] for k in expected[key]}==expected[key];x=h.get_tensor(key);assert tr(x)=={'shape':r['shape'],'dtype':'torch.bfloat16','sha256':r['source_sha256']} and r['actual_NPU_sha256']==r['original_BF16_sha256']==r['source_sha256'] and r['device']==device
 embedding=h.get_tensor('model.embed_tokens.weight')
for field,dim,theta in [('text_inv_freq',64,1000000.0),('vision_inv_freq',40,10000.0)]:
 inv=1/(theta**(torch.arange(0,dim,2,dtype=torch.int64).float()/dim));assert lineage[field]==tr(inv)
raw_inputs=[json.loads(l) for l in (folder/'sample_inputs.jsonl').read_text().splitlines()];raw_outputs=[json.loads(l) for l in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results'];trace=[json.loads(l) for l in pathlib.Path(lineage['per_sample_evidence']).read_text().splitlines()];activation=[json.loads(l) for l in pathlib.Path(lineage['activation_execution_trace']).read_text().splitlines()];finite=[json.loads(l) for l in pathlib.Path(lineage['next_token_logits_finiteness']).read_text().splitlines()];progress=[json.loads(l) for l in pathlib.Path(lineage['generation_progress']).read_text().splitlines()]
for rows in [raw_inputs,raw_outputs,details,trace,activation]:assert len(rows)==count and [r['index'] for r in rows]==list(range(count))
source=json.loads(pathlib.Path(p['source_preflight']).read_text());assert len(source['samples'])==200
external={};correct=legacy_correct=0;empty=[];invalid=[];truncated=[];rope_differences=[];totalsteps=0;finite_offset=0
for i,(a,b,d,t,act) in enumerate(zip(raw_inputs,raw_outputs,details,trace,activation)):
 old=source['samples'][i];assert not t['error'] and a['prompt']==old['prompt'] and a['reference']==old['reference'] and d['reference']==b['reference']==a['reference']
 media=pathlib.Path(a['media_path']).resolve();assert str(media)==old['path'] and sha(media)==old['original_image_sha256']==t['source_image_sha256'];inputs,chat,vision_positions=manual_inputs(root,old['prepared_prompt'],media)
 sample_path=pathlib.Path(t['sample_evidence']);assert sha(sample_path)==t['sample_evidence_sha256'];sample=json.loads(sample_path.read_text());assert sample['index']==i and not sample['error'] and sample['chat']==chat==old['chat']
 for k,v in inputs.items():assert sample['CPU_inputs'][k]==tr(v)
 for field in ['visual_archive','full_logits_archive']:
  path=pathlib.Path(sample[field]);assert str(path)==t[field] and sha(path)==sample[field+'_sha256']==t[field+'_sha256'];external[str(path)]=sha(path)
 external[str(sample_path)]=sha(sample_path);arrays=load_arrays(sample['visual_archive'],sample['visual_tensor_records']);logits=load_arrays(sample['full_logits_archive'],sample['full_logits_tensor_records'])['actual_next_token_full_logits'];tokens=t['generation_token_ids'];steps=len(tokens);L=inputs['input_ids'].shape[1];assert sample['decoder_steps']==len(sample['actual_steps'])==steps and logits.shape==(steps,1,151936) and logits.dtype==torch.bfloat16 and bool(torch.isfinite(logits).all()) and logits.float().argmax(-1)[:,0].tolist()==tokens
 assert sample['generation_token_ids']==tokens and 0<steps<=512 and (151645 not in tokens[:-1]) and (steps==512 or tokens[-1]==151645)
 assert torch.equal(arrays['actual_vision_pixels'],inputs['pixel_values'].bfloat16()) and torch.equal(arrays['actual_grid'],inputs['image_grid_thw']) and torch.equal(arrays['actual_vision_position_ids'],vision_positions)
 n=int(inputs['image_grid_thw'].prod())//4;features=arrays['actual_visual_features'];assert features.shape==(n,896) and arrays['actual_last_vision_block'].shape==(n*4,1280) and features.dtype==torch.bfloat16 and bool(torch.isfinite(features).all())
 x=arrays['actual_prefill_inputs_embeds'];image_mask=inputs['input_ids']==151655;assert torch.equal(x[image_mask],features) and torch.equal(arrays['actual_injected_visual'],features) and torch.equal(x[~image_mask],embedding[inputs['input_ids'][~image_mask]])
 pos,delta=manual_mrope(inputs['input_ids'],inputs['image_grid_thw'][0]);last=None
 for j,step in enumerate(sample['actual_steps']):
  q=L if j==0 else 1;past=0 if j==0 else L+j-1;assert step['step']==j and step['past_length_before']==past and step['cache_length_after']==past+q and step['use_cache'] is True and step['attention_mask']==[[1]*(past+q)] and step['input_ids']==(inputs['input_ids'].tolist() if j==0 else [[tokens[j-1]]])
  expected_pos=pos if j==0 else torch.full((3,1,1),L+j-1+delta,dtype=torch.int64);actual_pos=torch.tensor(step['mrope_position_ids']);assert torch.equal(actual_pos,expected_pos),(i,j)
  language_pos=torch.tensor(step['language_position_ids']);assert language_pos.shape[0] in [3,4]
  assert torch.equal(language_pos[-3:],expected_pos)
  if language_pos.shape[0]==4:assert torch.equal(language_pos[0],torch.arange(past,past+q).view(1,-1))
  mask=step['actual_attention_mask'];assert step['native_is_causal_attribute']
  if mask is None:assert step['native_sdpa_implicit_causal']==(q>1)
  else:
   actual=arrays['causal_mask_%04d'%j];assert tr(actual)==mask;allow=torch.arange(past+q)[None,:]<=torch.arange(past,past+q)[:,None];assert torch.equal(actual.reshape(q,past+q)==0,allow)
  for k,layer in enumerate(step['cache_layers']):assert layer=={'layer':k,'keys_shape':[1,2,past+q,64],'values_shape':[1,2,past+q,64],'device':device,'dtype':'torch.bfloat16'}
  assert len(step['cache_layers'])==24
  freq=1/(1000000.0**(torch.arange(0,64,2).float()/64));angle=(expected_pos[:,:,None,:].float()*freq[None,None,:,None]).transpose(-1,-2);theta=torch.cat([angle,angle],-1)
  for name,expected_trig in [('cos',theta.cos().bfloat16()),('sin',theta.sin().bfloat16())]:
   actual=arrays['rope_'+name+'_%04d'%j];assert actual.shape==expected_trig.shape and actual.dtype==torch.bfloat16 and bool(torch.isfinite(actual).all());rope_differences.append(float((actual.float()-expected_trig.float()).abs().max()))
  assert finite[finite_offset]=={'index':i,'step':j,'shape':[1,151936],'device':device,'dtype':'torch.bfloat16','all_finite':True};finite_offset+=1
 raw=vocab.decode(tokens,skip_special_tokens=True).strip();assert raw==t['raw_output']==sample['raw_output']==d['raw_output']==b['raw_output'];choice=independent_choice(raw);prediction=choice if choice is not None else raw;assert d['prediction']==b['prediction']==prediction
 if not raw:empty.append(i)
 if choice is None:invalid.append(i)
 if steps>=512:truncated.append(i)
 assert t['truncated']==t['budget_reached']==(steps>=512) and t['empty']==(not bool(raw)) and t['invalid']==(choice is None) and t['actual_visual_injection_verified'] and t['actual_generation_batch_size']==1
 correct+=int(choice is not None and choice==str(a['reference']).strip().upper());legacy_correct+=original_accuracy_sample(prediction,a['reference']);totalsteps+=steps
 counts=sample['native_activation_call_counts'];assert set(counts)==set(lineage['native_activation_modules']) and act['module_call_counts']==counts and act['decoder_steps']==steps and act['device']==device and act['dtype']=='torch.bfloat16'
 for name,calls in counts.items():assert calls==(steps if '.language_model.' in name else 1)
 assert sample['actual_vision_calls']==1
assert finite_offset==len(finite)==totalsteps and [r['index'] for r in progress if r.get('sample_complete')]==list(range(count))
score=correct/count;assert report['metrics']['Strict_MC_Accuracy']==score and report['summary']['current_score']==score and report['summary']['primary_metric']=='Strict_MC_Accuracy' and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
for name in ['per_sample_evidence','activation_execution_trace','generation_progress','next_token_logits_finiteness','weight_loading_evidence','profiler_trace']:
 path=pathlib.Path(lineage[name]);external[str(path)]=sha(path)
probe=pathlib.Path(lineage['native_GELU_probe']['path']);assert sha(probe)==lineage['native_GELU_probe']['sha256'];external[str(probe)]=sha(probe)
assert len(external)==3*count+7
profile=json.loads(pathlib.Path(lineage['profiler_trace']).read_text());events=profile if isinstance(profile,list) else profile['traceEvents'];hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'};kernels=[e for e in events if e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL')];assert kernels

numeric_path=pathlib.Path(params['numerical_review_file']);assert sha(numeric_path)==params['numerical_review_sha256']==lineage['numerical_review_sha256'];nr=json.loads(numeric_path.read_text());assert nr==lineage['native_numerical_review'] and nr['official_inference_remains_original_BF16_SDPA_cacheTrue_serial512'] and nr['CPU_and_NPU6_FP32_controls_diagnostic_only'] and nr['all_control_CPU_NPU_parameter_SHA_equal'] and nr['same_original_BF16_rounded_weights_pixels'] and nr['native_operator_formulas_unchanged']
for path,digest in nr['evidence_sha256'].items():assert sha(path)==digest
assert nr['source_BF16_visual_parameter_count']==391 and nr['same_input_diagnostic_block_count']==32 and nr['same_input_QuickGELU_modules_bitwise_equal']==32
with np.load(lineage['native_GELU_probe']['path'],allow_pickle=False) as z:
 expected=torch.linspace(-8,8,20001).bfloat16();assert np.array_equal(z['input_BF16_bytes'],expected.view(torch.uint8).numpy());probe_difference={}
 for mode in ['none','tanh']:
  actual=torch.from_numpy(z['NPU_'+mode+'_BF16_bytes'].copy()).view(torch.bfloat16);assert bool(torch.isfinite(actual).all()) and np.array_equal(z['CPU_'+mode+'_BF16_bytes'],torch.nn.functional.gelu(expected,approximate=mode).view(torch.uint8).numpy());probe_difference[mode]={'max_abs_NPU_vs_CPU_declared_mode':float((actual.float()-torch.nn.functional.gelu(expected,approximate=mode).float()).abs().max())}

proof={'report_dir':str(folder),'passed':True,'sample_count':count,'correct':correct,'Accuracy':score,'metrics':{'Strict_MC_Accuracy':score},'original_legacy_Accuracy_diagnostic_only':legacy_correct/count,'ordinary_articles_never_credited':True,'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,'index_duplicates':False,'CPU_image_preprocessing_replayed':count,'tokenizer_replayed':count,'decoded_generation_replayed':count,'actual_original_serial_generation_batch_size':1,'source_cast_actual_NPU_weight_tensors':681,'all24_native_KV_layers_all_steps_verified':totalsteps,'all_full_vocabulary_native_logits_argmax_replayed':totalsteps,'full_vocab_size':151936,'native_activation_module_count':57,'M_RoPE_positions_and_native_causal_SDPA_semantics_verified':True,'CPU_NPU_RoPE_trig_max_abs_difference_retained':max(rope_differences),'all_expected_visual_features_actually_injected':True,'NPU_profiler_kernel_events':len(kernels),'original_generation_config_source':'Constructed native model config; own generation_config.json was not loaded by original source','original_model_padded_IDs_without_token_string':unmapped,'no_token_added_removed_or_logits_masked':True,'external_evidence_sha256':external,'report_sha256':{q.name:sha(q) for q in folder.iterdir()},'historical_record_reuse':False,'protocol':'Original frozen serial1 BF16/native SDPA/cacheTrue/greedy512/own tied head and full input image chain. Only explicit MCv2 labels count under the previously authorized strict extraction, with every empty/invalid/truncated sample in denominator. Original legacy Accuracy is diagnostic only; no ordinary-article fallback credit. Full native151936 logits are independently replayed, without a choice grammar. Real CANN BF16 numerical differences retained; CPU output is never official model output.'}
proof.update(native_BF16_numerical_difference_retained_and_disclosed=True,native_numerical_review=nr,mineru_numerical_diagnostic_evidence_sha256=nr['evidence_sha256'],native_GELU_probe_CPU_replayed=True,native_GELU_probe_differences=probe_difference)
destination=pathlib.Path(lineage['weight_loading_evidence']).parent/'independent_report_verification.json';assert not destination.exists() or json.loads(destination.read_text())==proof
if not destination.exists():destination.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))

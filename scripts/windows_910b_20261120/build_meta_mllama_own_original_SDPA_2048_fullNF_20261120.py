"""Keep full numeric architecture checks; bind only own Meta original protocol."""
import ast,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
p=json.loads((V/'R_owned_MetaMllama_existing_full_observer_sources_20261120.json').read_bytes())['source_files']
old=p['scripts/verify_unsloth_mllama11b_original_fallback_disclosed_v2_910b.py']['complete_text']
imports=old[:old.index('torch.set_num_threads(4)')]
header='''torch.set_num_threads(4)
assert len(sys.argv)==3;folder=pathlib.Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in (3,200)
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_text());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];assert sha(kw['protocol_file'])==kw['protocol_sha256']
p=json.loads(pathlib.Path(kw['protocol_file']).read_text());w=pathlib.Path(p['weight_path']);line=json.loads((folder/'operator_lineage.json').read_text())['actual_execution'];vd='npu:'+str(kw['npu_devices'][0]);ld='npu:'+str(kw['npu_devices'][1])
assert cfg['model']['name']==p['model_id']=='meta-llama/Llama-3.2-11B-Vision-Instruct' and cfg['model']['model_path']==str(w) and folder.parent.parent.name=='meta-llama_Llama-3.2-11B-Vision-Instruct' and folder.parent.name==cfg['task_type']=='image-text-to-text' and re.fullmatch(r'\\d{8}_\\d{6}',folder.name)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1)
assert cfg['algorithm']['batch_size']==10 and cfg['algorithm']['kwargs']=={'extract_answer':True}
assert cfg['metrics']['override']==['Accuracy'] and cfg['metrics']['include_failed_predictions']
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert not cfg['data']['download'] and not cfg['model']['download'] and cfg['data']['preprocess'] and not cfg['data']['filter_empty'] and not cfg['data']['apply_template']
assert line['protocol']=='MetaMllama_own_original_BF16_defaultSDPA_outer10_inner1_2048' and line['weight_tensor_count']==906 and line['all_expected_parameters_loaded'] and line['runtime_dtype']=='BF16' and line['original_max_new_tokens']==2048 and line['original_outer_batch_size']==10 and line['original_untied_head128256_embedding128264_tokenizer128257_preserved'] and line['original_two_BOS128000_preserved'] and line['original_default_SDPA_all_text_and_vision'] and line['HF32_enabled']=={'conv':False,'matmul':False}
assert set(line['module_placements'].values())=={vd,ld}
for name,dev in line['module_placements'].items():assert dev==(vd if name=='model.vision_model' or name.startswith('model.vision_model.') or name=='model.multi_modal_projector' else ld)
assert sha(p['CPU200_protocol'])==p['CPU200_protocol_sha256'] and p['original_outer10_inner_serial1'] and p['group_count']==200
for n,h in p['runtime_and_original_source_sha256'].items():assert sha(n)==h
'''
sourceblock=old[old.index("for n,h in p['own_resource_sha256']"):old.index("old=json.loads(pathlib.Path(p['reference_report'])")]
sourceblock=sourceblock.replace("mc._attn_implementation='eager';mc.text_config._attn_implementation='eager';mc.vision_config._attn_implementation='eager'","assert mc._attn_implementation is mc.text_config._attn_implementation is mc.vision_config._attn_implementation is None")
sourceblock=sourceblock.replace('max_new_tokens=4096','max_new_tokens=2048')
body=old[old.index('ins=[json.loads'):old.index('kernels={}')]
a=body.index('for start in range(0,count,10):');b=body.index('assert len(acttrace)',a);body=body[:a]+'groups=[[i] for i in range(count)]\n'+body[b:]
body=body.replace("len(step_rows)<=4096","len(step_rows)<=2048").replace("stop>=4096","stop>=2048").replace("report['summary']['primary_metric']=='Strict_MC_Accuracy'","report['summary']['primary_metric']=='Accuracy'").replace("report['metrics']=={'Strict_MC_Accuracy':score}","report['metrics']=={'Accuracy':score}")
body=body.replace("if s['self_mask'] is None:assert torch.all(causal==0)","if s['self_mask'] is None:assert p['original_default_attention']=='sdpa' and torch.all(inp['attention_mask']==1)")
body=body.replace("prefix=inp['input_ids'].clone();pos0=", "reconstructed_KV={};prefix=inp['input_ids'].clone();pos0=")
needle="        assert s['pixels_present']==(n==1)"
insert='''        for i,rec in enumerate(s['cache40']):
            cross_layer=i in [3,8,13,18,23,28,33,38]
            for kind in ['key','value']:
                name='%04d_cache_%02d_%s'%(n,i,kind)
                if n==1:reconstructed_KV[(i,kind)]=t[name]
                elif not cross_layer:reconstructed_KV[(i,kind)]=torch.cat([reconstructed_KV[(i,kind)],t[name]],dim=-2)
                assert tr(reconstructed_KV[(i,kind)])==rec['complete_'+kind]
        calls=s['actual_native_SDPA_calls'];assert len(calls)==(80 if n==1 else 40)
        assert len([r for r in calls if r['section']=='text'])==40
        for r in calls:
            assert r['dropout']==0. and r['dtype']=='torch.bfloat16' and r['device']==(vd if r['section']=='vision' else ld)
            assert r['query_shape'][1]==(16 if r['section']=='vision' else 32) and r['query_shape'][-1]==(80 if r['section']=='vision' else 128)
            assert r['scale']==1./math.sqrt(r['query_shape'][-1])
            if r['mask'] is None:
                assert r['kind']=='self' and r['is_causal']==(r['query_shape'][-2]>1) and torch.all(inp['attention_mask']==1)
            else:
                actualmask=t[r['mask_array']];assert tr(actualmask)==r['mask'] and not r['is_causal']
                expected_mask=t['actual_vision_attention_mask'] if r['section']=='vision' else (causal if r['kind']=='self' else cross)
                assert actualmask.dtype==torch.bool and torch.equal(actualmask,torch.logical_not(expected_mask.bool()))
'''
assert body.count(needle)==1;body=body.replace(needle,insert+needle)
tail='''kernels={}
for kind,path in line['profiler_traces'].items():
    prof=json.loads(pathlib.Path(path).read_text());events=prof if isinstance(prof,list) else prof['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
    ks=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and (str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') or str(e.get('args',{}).get('Task Type','')) in {'AI_CORE','AI_CPU','AICPU','AIV','AIC'})]
    assert len(ks)>100;kernels[kind]=len(ks)
from meta_mllama_own_visual_value_reference_20261120 import verify_own_visual_values
visual=verify_own_visual_values(p,line,by,count)
external.update(visual['external_evidence_sha256'])
for path,digest in p['runtime_and_original_source_sha256'].items():external[path]=digest
for name,head in p['all_source_file_sha256'].items():external[str(w/name)]=head['sha256']
for name,digest in p['own_resource_sha256'].items():external[str(w/name)]=digest
external[p['CPU200_protocol']]=p['CPU200_protocol_sha256'];external[kw['protocol_file']]=kw['protocol_sha256']
proof=dict(passed=True,report_dir=str(folder),sample_count=count,Accuracy=score,metrics={'Accuracy':score},sample_Accuracy=per,correct=correct,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_native_forwards=total_steps,all906_own_source_cast_actual_NPU_SHA_verified=True,all511_own_visual_source_SHA_verified=True,all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward=True,all_native_SDPA_call_masks_causal_scale_head_arguments_independently_verified=True,original_outer10_inner_serial1_preserved=True,original_max_new_tokens=2048,original_two_BOS_and_untied_128256_head_128264_embedding_preserved=True,all_full128256_native_greedy_argmax_replayed=total_steps,independent_original_F32_angles_BF16_max_abs=max_rope,NPU_profiler_kernel_events=sum(kernels.values()),first_actual_vision_and_language_prefill_kernels_only=kernels,external_evidence_sha256=external,report_sha256={q.name:sha(q) for q in folder.iterdir()},protocol_file=kw['protocol_file'],protocol_sha256=kw['protocol_sha256'],precision_disclosure=line['precision_disclosure'],own_original_visual_numerical_controls=visual,historical_A100_engine_or_tensor_capture_claimed=False,all_original_input_output_raw_ref_token_error_denominators_preserved=True)
q=loadpath.parent/'independent_report_verification.json';proof=json.loads(json.dumps(proof));assert not q.exists() or json.loads(q.read_text())==proof
if not q.exists():q.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
'''
s=imports+header+sourceblock+body+tail
ast.parse(s);name='verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v1_20261120.py';(V/name).write_text(s,encoding='utf8',newline='\n')
print(json.dumps(dict(passed=True,path=name,SHA256=hashlib.sha256(s.encode()).hexdigest(),own_source_input_stopping_original2048_binding=True,old_other_model_reference_scores_AA_MLU_fallback_and_control_files_removed=True,full_numerical_body_preserved_and_extended_KV_and_actual_SDPA_arguments=True)))

"""No forecast scores: require completed genuine200 and all final gates first."""
from pathlib import Path
from decimal import Decimal
import json,hashlib
P=Path(__file__).resolve().parent;x=json.loads((P/'R_Huihui_completed200_exact_publication_inputs_20261119.json').read_bytes())
assert x['ready_for_publication_review'] and not x['FLOCK'] and not x['own_controller_still_alive']
pf=x['formal_proof'];assert pf['passed'] and pf['sample_count']==200 and pf['actual_NPU_inference'] and not pf['historical_record_reuse'] and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['source_tensor_count']==704 and pf['all_native_KV_values_losslessly_reconstructed'] and pf['all_full_inputs_tokens_raw_references_errors_preserved']
assert pf['formal_alignment_within_5pct'] and pf['original_protocol_final_qualification_complete'] and not pf['error_indices']
assert set(pf['metrics'])=={'Accuracy'} and all(Decimal(v)<=Decimal('.05') for v in pf['relative_error_to_current_A100_D'].values())
assert sum(n['full_head_rows'] for n in pf['numeric_sample_checks'])==pf['actual_forward_count'] and len(pf['numeric_sample_checks'])==200
score=pf['metrics']['Accuracy'];MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated';row=x['current_row'];assert row['row']==355 and row['task']=='image-text-to-text' and row['score'] is None and Decimal(row['current_D'])==Decimal('.945')
H=P/'HuihuiGLM46Flash_local_result_sync_verified181_20261119';q=json.loads((H/'preview_plan.json').read_bytes());book=Path(q['target']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(book)==sha(q['source_backup'])==q['expected_before_sha256']=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
q.pop('preview_only_no_predicted200_score');q['changes'][0].update(new_value=score,relative_error=str(abs(Decimal(row['current_D'])-Decimal(str(score)))/abs(Decimal(row['current_D']))),new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir=x['completion']['accepted'][-1]['report_dir'],formal_proof_sha256=x['formal_proof_SHA256'])
st=book.stat();q.update(real_measured_count=181,numeric_within5pct=166,numeric_outside5pct=15,complete_weights_pending_formal=209,confirmed_missing=310,original_protocol_qualified=67,model_blank_count=519,complete_numeric_verifier_passed=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
q['models']={MID:dict(job=x['job'],controller=x['controller'],job_SHA256=x['job_SHA256'],true3_score=1.0,true3_proof_SHA256=x['true3_proof_SHA256'],expected_forward_count=pf['actual_forward_count'],expected_kernels=pf['NPU_profiler_kernel_events'],expected_empty=pf['empty_indices'],expected_invalid=pf['invalid_indices'],expected_truncated=pf['truncated_indices'])}
q.update(test_template='scripts/tests/verified_Gemma4E2B_manifest_task_writer_11_cases_template_20261116.py',test_template_SHA256='8e595053cedf0ff3fc46b1f1a76fde787ed268ac190bc037e7521882e048e318')
with (H/'plan.json').open('x',encoding='utf8') as f:json.dump(q,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(plan=str(H/'plan.json'),score=score,three_D_errors=pf['relative_error_to_current_A100_D'],forward_count=pf['actual_forward_count'],production_not_written=True),ensure_ascii=False))

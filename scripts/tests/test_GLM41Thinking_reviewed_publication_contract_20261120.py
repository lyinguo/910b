"""Metadata rejection tests against own completed3 evidence; no learned execution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import ast,copy,json
R=Path('/root/projects/comperror')
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
P=R/'scripts/publish_GLM41Thinking_verified184_same_reserved_original_lock_dual_transaction_v1_20261120.py'
Q=R/'scripts/reserve_GLM41_completed200_reviewed_dual_publication_original_lock_v2_actual_runner_directory_20261120.py'
functions=[]
for path,names in [(P,['glm46_capacity_review_passed','full_numeric_head_verified']),(Q,['qualified'])]:
 tree=ast.parse(path.read_bytes())
 functions.extend(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names)
assert len(functions)==3
g={'Decimal':Decimal};exec(compile(ast.Module(body=functions,type_ignores=[]),'<own-publication-contract>','exec'),g)
home=B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
cap=json.loads((home/'own_passed3_complete_shape_capacity_review.json').read_bytes())
assert g['glm46_capacity_review_passed'](cap)
passed=['actual_own_complete3009_capacity_accepted']
for k,v in [('model_id','zai-org/AutoGLM-Phone-9B'),('capacity_passed',False),('original_factor',3),('empirical_estimate_bytes',1),('original_true3_proof_SHA256','0'*64),('all200_inputs_are_this_models_own_original_frontend',False)]:
 bad=copy.deepcopy(cap);bad[k]=v;assert not g['glm46_capacity_review_passed'](bad);passed.append(k+'_capacity_rejected')
proof=json.loads((B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120/fresh3/verification_stdout.json').read_bytes())
assert g['full_numeric_head_verified'](proof)
passed.append('actual_all_full_head_rows_and40KV_checks_accepted')
for kind in ['missing_sample','wrong_head_denominator','missing_head_reference','KV_not_reconstructed','mask_positions_unchecked']:
 bad=copy.deepcopy(proof)
 if kind=='missing_sample':bad['numeric_sample_checks'].pop()
 elif kind=='wrong_head_denominator':bad['actual_forward_count']+=1
 elif kind=='missing_head_reference':bad['numeric_sample_checks'][0]['head_reference_checks'].pop()
 elif kind=='KV_not_reconstructed':bad['numeric_sample_checks'][0]['all40_native_cache_values_reconstructed']=False
 else:bad['numeric_sample_checks'][0]['all_masks_3D_positions_and_own_visual_injection_checked']=False
 assert not g['full_numeric_head_verified'](bad);passed.append(kind+'_rejected')
positive=dict(passed=True,sample_count=200,actual_NPU_inference=True,historical_record_reuse=False,formal_alignment_within_5pct=True,original_protocol_final_qualification_complete=True,relative_error_to_current_A100_D=dict(server='.05',local='.05',preserved_original='.05'))
assert g['qualified'](positive);passed.append('all_three_exact5pct_complete200_accepted')
for k in ['server','local','preserved_original']:
 bad=copy.deepcopy(positive);bad['relative_error_to_current_A100_D'][k]='.050000001';assert not g['qualified'](bad);passed.append(k+'_outside5pct_rejected')
assert not g['qualified'](proof);passed.append('passed_small3_not_formal200_qualification')
print(json.dumps(dict(passed=True,test_count=len(passed),checks=passed,actual_NEW_model_forwards=0,old_completed3_proof_read_only=True,no_production_book_or_lock_write=True)))

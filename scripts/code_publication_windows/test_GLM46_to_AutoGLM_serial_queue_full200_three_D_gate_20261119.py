"""Exercise queue routing guards without creating jobs, running NN or writing G."""
from pathlib import Path
from decimal import Decimal
import ast,copy,json
P=Path(__file__).resolve().parent
source=P/'watch_GLM46_actual_completed200_then_qualified_dual_commit_or_preserved_gap_start_AutoGLMPhone_serial_20261119.py'
node=next(x for x in ast.parse(source.read_bytes()).body if isinstance(x,ast.FunctionDef) and x.name=='actual_full200_qualification')
g={'Decimal':Decimal};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),g);gate=g[node.name]
base=dict(passed=True,sample_count=200,actual_NPU_inference=True,all_own_source_actual_NPU_SHA256_verified=True,all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,all_original_active_and_unused_source_bytes_verified=True,source_tensor_count=704,actual_forward_count=500,relative_error_to_current_A100_D=dict(server='0.05',local='0',preserved_original='0.05'),formal_alignment_within_5pct=True,original_protocol_final_qualification_complete=True)
assert gate(base) is True
bad=copy.deepcopy(base);bad['relative_error_to_current_A100_D']['local']='0.0500000000000000000000001';bad['formal_alignment_within_5pct']=bad['original_protocol_final_qualification_complete']=False;assert gate(bad) is False
rejected=0
for change in [dict(sample_count=199),dict(actual_NPU_inference=False),dict(source_tensor_count=703),dict(all_full_inputs_tokens_raw_references_errors_preserved=False),dict(original_protocol_final_qualification_complete=False),dict(relative_error_to_current_A100_D=dict(server='0',local='0')),dict(relative_error_to_current_A100_D=dict(server='0',local='0.2',preserved_original='0'))]:
 v=copy.deepcopy(base);v.update(change)
 try:gate(v)
 except AssertionError:rejected+=1
 else:raise AssertionError(change)
assert rejected==7
result=dict(passed=True,checks=9,qualification_requires_all_three_exact_Decimal_errors=True,full200_source_full_values_and_real_NPU_required=True,numerical_baselinegap_is_preserved_without_G_write=True,actual_NEW_NN_or_G_writes=0)
(P/'GLM46_AutoGLM_serial_queue_gate_tests_20261119.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

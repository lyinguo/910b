"""Correct only the independent scale expression to original power operation."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent
old=V/'verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v2_exact_instruction_metadata_20261120.py';raw=old.read_bytes();assert hashlib.sha256(raw).hexdigest()=='828f833dc5372c5e1daebc28339cbaa44ebb78d34d62cc4574ed1730af38a8eb'
a="assert r['scale']==1./math.sqrt(r['query_shape'][-1])";b="assert r['scale']==r['query_shape'][-1]**-0.5";s=raw.decode();assert s.count(a)==1;s=s.replace(a,b);assert s.replace(b,a)==raw.decode();ast.parse(s)
nf='verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py';(V/nf).write_text(s,encoding='utf8',newline='\n');nfsha=hashlib.sha256(s.encode()).hexdigest()
control=json.loads((V/'R_MetaMllama_original_SDPA_exact_scale_control_20261120.json').read_bytes());assert all(v['actual_equals_original_pow'] for v in control['captured_distinct_arguments'].values()) and not control['captured_distinct_arguments']['self']['actual_equals_reciprocal_sqrt']
watch=(V/'continue_meta_mllama_completed_genuine3_formal_NF2_exact_instruction_no_NN_replay_20261120.py').read_text();watch=watch.replace('NF2','NF3').replace('full_values_v2_exact_instruction_metadata','full_values_v3_exact_original_power_reference').replace('828f833dc5372c5e1daebc28339cbaa44ebb78d34d62cc4574ed1730af38a8eb',nfsha)
watch=watch.replace('metadata_repair_v2','exact_original_power_reference_v3').replace('only_exact_original_instruction_metadata_assert_changed=True','original_instruction_metadata_and_original_power_expression_reference_restored=True').replace('only_exact_original_instruction_metadata_assert_changed=True','original_instruction_metadata_and_original_power_expression_reference_restored=True')
needle=" done=js(OLD/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==1,done"
gate=""" prior=B/'MetaMllama_completed_genuine3_formal_NF2_exact_original_instruction_no_NN_replay_capacity_serial_20261120'
 assert js(prior/'actual_completion.json')['actual_returncode']==1 and \"r['scale']==1./math.sqrt(r['query_shape'][-1])\" in (prior/'formal_fullNF2_stderr.log').read_text()
"""
assert watch.count(needle)==1;watch=watch.replace(needle,gate+needle);ast.parse(watch)
wname='continue_meta_mllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_20261120.py';(V/wname).write_text(watch,encoding='utf8',newline='\n')
builder=(V/'build_stage_meta_mllama_formal_NF2_same_genuine3_no_NN_replay_20261120.py').read_text().replace('NF2','NF3').replace('full_values_v2_exact_instruction_metadata','full_values_v3_exact_original_power_reference');ast.parse(builder);(V/'build_stage_meta_mllama_formal_NF3_same_genuine3_no_NN_replay_20261120.py').write_text(builder,encoding='utf8',newline='\n')
review=dict(passed=True,new_verifier=nf,SHA256=nfsha,old_NF2_SHA256=hashlib.sha256(raw).hexdigest(),exact_one_original_pow_reference_assert_corrected=True,all_other_source_bytes_identical=True,actual_native_SDPA_scale_dtypes_masks_cache_budget_and_NN_untouched=True,no_threshold_relaxation=True,original_source_expression_and_actual_capture_proof=control,no_model_NN_replay=True)
(V/'MetaMllama_NF3_exact_original_power_derivation_review_20261120.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review,ensure_ascii=False))

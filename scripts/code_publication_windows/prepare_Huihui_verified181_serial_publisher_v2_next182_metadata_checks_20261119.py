"""Review actual full-head schema and correct only future-count/test-file metadata."""
from pathlib import Path
import ast,copy,json,hashlib
P=Path(__file__).resolve().parent
original=(P/'publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py').read_text(encoding='utf8')
source=original.replace('next_new181_count_branch_review_mandatory','next_new182_count_branch_review_mandatory').replace('_verified181_original_writer_isolated_20261117.py','_verified181_original_writer_isolated_20261119.py')
assert source!=original and 'next_new181_count_branch_review_mandatory' not in source
tree=ast.parse(source);helpers=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['full_numeric_head_verified','huihui_capacity_review_passed']],type_ignores=[]);ns={};exec(compile(helpers,'reviewed_static_helpers','exec'),ns)
d=json.loads((P/'R_user_Huihui_serial_continue_actual_current_20261119.json').read_bytes());proof=d['stages']['fresh3']['verification_stdout.json'];fn=ns['full_numeric_head_verified'];assert fn(proof)
bad=copy.deepcopy(proof);bad['numeric_sample_checks'][0]['full_head_rows']+=1;assert not fn(bad)
bad=copy.deepcopy(proof);bad['numeric_sample_checks'][0]['all40_native_cache_values_reconstructed']=False;assert not fn(bad)
bad=copy.deepcopy(proof);bad['numeric_sample_checks']=bad['numeric_sample_checks'][:-1];assert not fn(bad)
home=next(v for k,v in d['files'].items() if 'resumer' in k);cap=home['initial_original_storage_capacity.json'];assert ns['huihui_capacity_review_passed'](cap)
bad=copy.deepcopy(cap);bad['original_factor4_plus6GiB_and_all_sample_guards_unchanged']=False;assert not ns['huihui_capacity_review_passed'](bad)
calls=lambda s:[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_verified_A100_G'];assert calls(source)==calls(original)
name='publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py';raw=source.encode();(P/name).write_bytes(raw)
review=dict(passed=True,source=name,SHA256=hashlib.sha256(raw).hexdigest(),checks=6,current_full_head_positive_and_incomplete_corrupt_cache_refusals=True,real_lossless_capacity_positive_and_reduced_guard_refusal=True,original_G_writer_calls_AST_identical=True,v1_not_executed_both_sources_preserved=True,next_after_verified181_is182=True,private_isolated_test_source_date_20261119=True,actual_production_not_written=True)
(P/'Huihui_verified181_publisher_v2_scope_checks_20261119.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review,ensure_ascii=False))

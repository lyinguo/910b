"""Archive current facts and exact new source bytes, with one retention-only CAS."""
from pathlib import Path
import ast,base64,hashlib,json,sys
P=Path(__file__).resolve().parent;read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
prior=read('Huihui_serial200_dual181_GLM_next_complete_late_plan_v40_20261119.json');closed=read('R_close_Huihui_serial200_GLM_next_all_sources_v40_20261119.json');official=read('official910b_Huihui_fullNF_gap105_GLM_serial_readback_20261119.json');monitor=read(sys.argv[1]);assert official['passed']
R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D+'/code_publication_source_layout_20261114'
main=[
 ('audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119.py','scripts/maintenance/audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119.py'),
 ('retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py','scripts/maintenance/retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py'),
 ('test_Huihui_completed200_baselinegap_selective_retention_wrong_controls_20261119.py','scripts/tests/test_Huihui_completed200_baselinegap_selective_retention_wrong_controls_20261119.py'),
 ('continue_GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.py','scripts/continue_GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.py'),
 ('close_alignment_handoff_verified181_publication_and_retention_v7_20261119.py','scripts/maintenance/close_alignment_handoff_verified181_publication_and_retention_v7_20261119.py'),
 ('close_alignment_handoff_verified_Huihui_measurement_only_retention_v8_20261119.py','scripts/maintenance/close_alignment_handoff_verified_Huihui_measurement_only_retention_v8_20261119.py')]
controls=[
 'read_Huihui_full200_real_five_and_active_independent_NF_progress_20261119_remote.py',
 'read_Huihui_full200_metric_schema_and_original_saved_frontend_source_no_NN_20261119_remote.py',
 'read_Huihui_publication_role_and_original_cell_schema_20261119.py',
 'read_Huihui_fresh3_schema_and_current200_retention_candidates_no_delete_20261119_remote.py',
 'read_Huihui_completed200_audit_actual_metadata_schema_no_mutation_20261119_remote.py',
 'build_deploy_Huihui_full_measurement_audit_selective_retention_no_model_change_20261119.py',
 'check_Huihui_measurement_audit_scope_tests_and_real_completion_20261119_remote.py',
 'run_Huihui_completed200_full_numerical_gap_target_audit_20261119_remote.py',
 'read_GLM46_serial_next_own_protocol_source_contract_no_NN_20261119_remote.py',
 'build_Huihui_reviewed_measurement_retention_plan_20261119.py',
 'run_Huihui_reviewed_measurement_retention_plan_20261119_remote.py',
 'build_Huihui_exact_reviewed_correct_nonexample_retention_apply_20261119.py',
 'apply_Huihui_exact_reviewed_correct_nonexample_retention_20261119_remote.py',
 'build_launch_GLM46_after_verified_Huihui_gap_cleanup_serial_v2_20261119.py',
 'read_GLM46_after_Huihui_verified105_serial_v2_real_process_attempt_20261119_remote.py',
 'read_GLM46_serial_v2_loading_actual_FD_io_card_no_signal_20261119_remote.py',
 'build_official910b_Huihui_fullNF_gap105_GLM_serial_actual_20261119.py',
 'verify_official910b_Huihui_fullNF_gap105_GLM_serial_readback_20261119.py',Path(__file__).name]
sources=[]
for n,relative in main+[(n,'scripts/code_publication_windows/'+n) for n in controls]:
 raw=(P/n).read_bytes();ast.parse(raw);sources.append(dict(relative=relative,SHA256=sha(raw),base64=base64.b64encode(raw).decode(),role='Actual complete numerical baseline-gap acceptance, correct-only retention with all incorrect controls preserved, serial own-model first200, Windows consumers and exact official handoff; Day20 review all useful paths and dependencies'))
names=[n for n,_ in main]+controls+[
 'Huihui_completed200_baselinegap_local_current_snapshot_20261119.json',
 'R_Huihui_completed200_exact_publication_inputs4_20261119.json',
 'R_Huihui_full200_metric_schema_original_frontend_no_NN_20261119.json',
 'R_retention_policy_full_new_numerical_measurement_scope_20261119.json',
 'R_Huihui_fresh3_schema_current200_retention_candidates_no_delete_20261119.json',
 'R_Huihui_completed200_audit_actual_metadata_schema_20261119.json',
 'R_deploy_Huihui_full_measurement_audit_selective_retention_20261119.json',
 'R_Huihui_measurement_scope_tests_actual_completion_20261119.json',
 'R_Huihui_completed200_full_numerical_gap_target_audit_20261119.json',
 'R_GLM46_serial_next_own_protocol_source_contract_20261119.json',
 'R_Huihui_reviewed_measurement_retention_plan_20261119.json',
 'R_Huihui_exact_reviewed_correct_nonexample_retention_actual_20261119.json',
 'R_Huihui_serial_followup_after_actual_full_NF_completed_20261119.json',
 'R_GLM46_after_Huihui_gap_verified105_cleanup_serial_v2_actual_launch_20261119.json',
 'R_GLM46_serial_v2_loading_actual_FD_io_card_20261119.json',
 'read_GLM46_serial_v2_original_prefix_mismatch_reader_20261119.py.txt',
 'GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.json',
 sys.argv[1],sys.argv[2],
 'official910b_Huihui_fullNF_gap105_GLM_serial_arguments_20261119.json',
 'official910b_Huihui_fullNF_gap105_GLM_serial_tool_result_20261119.json',
 'official910b_Huihui_fullNF_gap105_GLM_serial_readback_20261119.json',
 'automation910b_before_Huihui_fullNF_gap105_GLM_serial_20261119.toml',
 'automation910b_after_Huihui_fullNF_gap105_GLM_serial_20261119.toml',
 'handoff_Huihui_fullNF_gap105_GLM_serial_actual_20261119.txt']
packets=[]
for n in dict.fromkeys(names):
 raw=(P/n).read_bytes();assert len(raw)<8*1024**2;packets.append(dict(name=n,bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
HH=B+'/HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119';M=B+'/Huihui_completed200_full_numerical_baselinegap_target_audit_and_correct_nonexample_retention_20261119';G=B+'/GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119';H=B+'/GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119'
stage={
 'Huihui/completion.json':HH+'/completion.json','Huihui/fullNF200.json':HH+'/formal200/verification_stdout.json',
 'full_current_handoff_text':closed['handoff_directory']+'/full_actual_Huihui_baselinegap105_GLM_serial_20261119.txt',
 'measurement/target_audit.json':M+'/measurement_target_audit.json','measurement/retention_plan.json':M+'/retention_plan.json','measurement/retention_actual.json':M+'/retention_actual_completion.json',
 'GLM46/controller_state.json':H+'/state.json','GLM46/controller_actual.json':H+'/actual_completion.json','GLM46/approval.json':H+'/first200_measurement_approval.json','GLM46/capacity.json':H+'/fresh_complete_component_capacity_review.json',
 'GLM46/job.json':G+'/job.json','GLM46/state.json':G+'/state.json','GLM46/completion.json':G+'/completion.json','GLM46/fresh3_fullNF.json':G+'/fresh3/verification_stdout.json','GLM46/formal200_fullNF.json':G+'/formal200/verification_stdout.json',
 'previous_serial/actual_capacity_refusal.json':B+'/after_Huihui_real_dual181_then_GLM46_own_completed3_component_capacity_serial_v1_20261119/actual_completion.json'}
plan=dict(handoff_directory=closed['handoff_directory'],expected_pointer_SHA256=closed['pointer_SHA256'],canonical_SHA256=monitor['canonical_SHA256'],note_name='Huihui_fullNF200_baselinegap_all61_wrong_controls_kept_verified105_cleanup_GLM46_serial_actual_all_sources_20261119',local_original_packets=packets,official_readback=official,additional_existing_primary_sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],process_keyword='GLM46',current_stage_files=stage,new_pending=C+'/pending_after_daily_20261119_verified180_Huihui_fullNF_gap_verified105_retention_GLM46_serial_all_primary_v41.json')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'Huihui_fullNF_gap105_GLM_serial_complete_late_plan_v41_20261119.json').write_bytes(raw)
data=dict(sources=sources,plan_base64=base64.b64encode(raw).decode(),plan_SHA256=sha(raw))
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for row in DATA['sources']:
 p=R/row['relative'];raw=base64.b64decode(row['base64'],validate=True);assert p.resolve().is_relative_to(R) and hashlib.sha256(raw).hexdigest()==row['SHA256'];ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw,('Existing source differs; never overwrite live/frozen source',str(p))
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw)
J=Path(plan['current_stage_files']['GLM46/job.json']);job=json.loads(J.read_bytes());runtime=dict(job['spec']['source_sha256'])
runtime['configs/retention/Huihui_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.json']=hashlib.sha256((R/'configs/retention/Huihui_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.json').read_bytes()).hexdigest()
for n,h in runtime.items():
 p=R/n;assert not Path(n).is_absolute() and p.resolve().is_relative_to(R) and hashlib.sha256(p.read_bytes()).hexdigest()==h
 plan['additional_existing_primary_sources'].append(dict(relative=n,SHA256=h,role='Actual immutable NEWfresh3/fixed200 source and real YAML/protocol/smoke or exact user retention config consumer; full dependencies reviewed Day20'))
path=Path(plan['handoff_directory'])/'Huihui_fullNF_gap105_GLM_serial_complete_late_plan_v41_20261119.json';assert not path.exists();body=(json.dumps(plan,ensure_ascii=False,indent=2)+'\\n').encode();path.write_bytes(body)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.close_alignment_handoff_verified_Huihui_measurement_only_retention_v8_20261119');print(json.dumps(m.execute(path,hashlib.sha256(body).hexdigest()),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_close_Huihui_fullNF_gap105_GLM_serial_all_sources_v41_20261119_remote.py').write_text(remote,encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(raw),plan_bytes=len(raw),new_primary_sources=len(sources),original_packets=len(packets),new_pending=plan['new_pending']),ensure_ascii=False))

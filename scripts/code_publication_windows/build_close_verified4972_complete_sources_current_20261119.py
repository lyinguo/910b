"""Preserve the complete actual cleanup/restart evidence and source/config consumers."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
old=read('R_user_reaudit_accepted_cleanup_current_20261119.json');ptr=old['pointer'];official=read('official910b_verified4972_lossless_serial_v4_readback_20261119.json');assert official['passed']
monitor=read('R_user_verified4972_live_Huihui_frozen_monitor_checkpoint_20261119.json');canonical=monitor['canonical_SHA256'];assert {n for n,h in canonical.items() if ptr['canonical_SHA256'][n]!=h}=={'/root/projects/comperror/workspace/diagnostics/batch_910b/A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'}
R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D+'/code_publication_source_layout_20261114'
models=[
 ('retain_verified180_current_accepted_nonexample_user_serial_20261119.py','scripts/maintenance/retain_verified180_current_accepted_nonexample_user_serial_20261119.py'),
 ('verified180_accepted_nonexample_user_serial_continue_20261119.json','configs/retention/verified180_accepted_nonexample_user_serial_continue_20261119.json'),
 ('retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119.py','scripts/maintenance/retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119.py'),
 ('verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json','configs/retention/verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json'),
 ('diagnostic_huihui_completed_true3_complete_standard_LZMA_capacity_after_verified_cleanup_20261119.py','scripts/diagnostic_huihui_completed_true3_complete_standard_LZMA_capacity_after_verified_cleanup_20261119.py'),
 ('cleanup_regenerable_bytecode_and_inactive_tmp_installers_exact_20261119.py','scripts/maintenance/cleanup_regenerable_bytecode_and_inactive_tmp_installers_exact_20261119.py'),
 ('prune_supported_offline_unused_package_cache_final_serial_gap_20261119.py','scripts/maintenance/prune_supported_offline_unused_package_cache_final_serial_gap_20261119.py'),
 ('huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_v3_20261119.py','universal_eval/model/adapters/huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_v3_20261119.py'),
 ('continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py','scripts/continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v3_20261119.py'),
 ('continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v4_approval_schema_20261119.py','scripts/continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v4_approval_schema_20261119.py'),
 ('close_alignment_handoff_official_readback_authorized_retention_current_owned_v6_20261119.py','scripts/maintenance/close_alignment_handoff_official_readback_authorized_retention_current_owned_v6_20261119.py'),
]
controls=[
 'discover_verified180_current_accepted_nonexample_user_serial_20261119_remote.py',
 'read_verified180_unmapped_accepted_trace_formats_20261119_remote.py',
 'read_verified180_previous_completed_accepted200_only_metadata_20261119_remote.py',
 'read_verified180_retention_actual_progress_current_20261119_remote.py',
 'read_verified180_foreign_reference_exact_reasons_20261119_remote.py',
 'build_verified180_user_accepted_nonexample_cleanup_plan_20261119.py',
 'build_verified180_owner_reference_v2_accepted_cleanup_plan_20261119.py',
 'apply_reviewed_verified180_owner_reference_v2_exact4972_20261119_remote.py',
 'build_Huihui_complete_standard_lossless_storage_capacity_after_cleanup_20261119.py',
 'read_verified4972_cleanup_Huihui_lossless_capacity_current_20261119_remote.py',
 'read_regenerable_bytecode_metadata_cache_for_final_capacity_gap_20261119_remote.py',
 'apply_final_gap_bytecode_tmp_wheel_exact_reviewed_20261119_remote.py',
 'read_installed_uv_cache_prune_supported_help_20261119_remote.py',
 'read_existing_uv_binary_absolute_candidates_only_20261119_remote.py',
 'find_existing_uv_cache_management_binary_no_install_20261119_remote.py',
 'read_bound_uv_bootstrap_supported_cache_prune_help_20261119_remote.py',
 'build_Huihui_verified4972_lossless_storage_serial_resume_v3_20261119.py',
 'build_Huihui_verified4972_lossless_storage_serial_v4_approval_schema_20261119.py',
 'read_Huihui_verified4972_lossless_serial_v3_actual_job_proc_samples_20261119_remote.py',
 'read_Huihui_v3_runner_early_exit_complete_logs_20261119_remote.py',
 'read_Huihui_verified4972_lossless_serial_v4_actual_job_proc_samples_20261119_remote.py',
 'read_Huihui_lossless_serial_v4_current_v5_partial_JSON_safe_20261119_remote.py',
 'build_close_alignment_handoff_retention_transition_current_owned_v6_20261119.py',
 'build_official910b_verified4972_cleanup_lossless_serial_v4_current_20261119.py',
 'verify_official910b_verified4972_lossless_serial_v4_readback_20261119.py',
 Path(__file__).name,
]
sources=[]
for n,relative in models+[(n,'scripts/code_publication_windows/'+n) for n in controls]:
 b=(P/n).read_bytes();ast.parse(b) if n.endswith('.py') else json.loads(b)
 sources.append(dict(relative=relative,SHA256=sha(b),base64=base64.b64encode(b).decode(),role='Actual user-authorized precise current200 nonexample cleanup, verified same-math lossless NPZ/native, serial schema recovery, offline package-cache maintenance, real consumers and Windows control/official handoff source; review Day20'))
# Actual runtime-generated model YAML/protocol sources are collected from the real job reader.
live=read('R_user_Huihui_lossless_serial_v4_independent_partial_safe_current_20261119.json')
runtime_paths=set()
for files in live['files'].values():
 if not files:continue
 job=files.get('job.json')
 if job:
  spec=job['spec']
  runtime_paths.update(n for n in spec['source_sha256'] if n.startswith('configs/') and ('lossless_LZMA' in n or 'lossless-LZMA' in n))
# v3 attempt remains an immutable source/config failure artifact.
runtime_paths.update([
'configs/Glm4vForConditionalGeneration/huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-lossless-LZMA-serial-approved-first200-v3-20261119.yaml',
'configs/protocols/huihui_glm46flash_own_original_BF16_2048_lossless_LZMA_serial_first200_v3_20261119.json',
'configs/Glm4vForConditionalGeneration/huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-lossless-LZMA-serial-approved-first200-v4-approval-schema-20261119-smoke.yaml'])
packets=[]
names=controls+[n for n,_ in models]+[
'build_Huihui_verified4972_lossless_storage_serial_resume_v3_original_literal_failure_20261119.py',
'verified180_dependency_owner_classifier_seven_scope_checks_20261119.json',
'Huihui_lossless_storage_same_model_math_AST_review_20261119.json',
'Huihui_v4_approval_schema_change_actual_review_20261119.json',
'handoff_user_reaudit4972_lossless_serial_v4_current_20261119.txt',
'official910b_verified4972_lossless_serial_v4_arguments_20261119.json',
'official910b_verified4972_lossless_serial_v4_tool_result_20261119.json',
'official910b_verified4972_lossless_serial_v4_readback_20261119.json',
'automation910b_before_verified4972_lossless_serial_v4_20261119.toml',
'automation910b_before_verified4972_lossless_serial_v4_prompt_20261119.txt',
'automation910b_after_verified4972_lossless_serial_v4_20261119.toml',
'automation910b_after_verified4972_lossless_serial_v4_prompt_20261119.txt',
'R_user_reaudit_accepted_cleanup_monitor_20261119.json',
'R_user_reaudit_accepted_cleanup_current_20261119.json',
'R_user_verified180_accepted_nonexample_fresh_discovery_20261119.json',
'R_user_verified180_unmapped_trace_format_metadata_20261119.json',
'R_user_verified180_previous_completed_accepted200_metadata_20261119.json',
'R_user_verified180_reaudit_exact_removal_protection_plan_20261119.json',
'R_user_verified180_foreign_reference_exact_reasons_20261119.json',
'R_user_verified180_owner_reference_v2_exact_removal_plan_20261119.json',
'R_user_verified180_owner_reference_v2_exact4972_actual_20261119.json',
'R_user_Huihui_complete_lossless_storage_capacity_after_cleanup_20261119.json',
'R_user_verified4972_Huihui_lossless_capacity_current_20261119.json',
'R_user_final_gap_regenerable_bytecode_cache_inventory_20261119.json',
'R_user_final_gap_bytecode_tmp_wheel_cleanup_plan_20261119.json',
'R_user_final_gap_bytecode_tmp_wheel_cleanup_actual_20261119.json',
'R_user_installed_uv_supported_cache_prune_help_20261119.json',
'R_user_installed_uv_supported_cache_prune_help_20261119.json.stderr',
'R_user_existing_uv_absolute_cache_prune_help_20261119.json',
'R_user_existing_uv_cache_manager_binary_discovery_20261119.json',
'R_user_bound_uv_cache_prune_supported_scope_20261119.json',
'R_user_supported_uv_offline_prune_final_serial_gap_actual_20261119.json',
'R_user_Huihui_verified4972_lossless_storage_serial_v3_launch_20261119.json',
'R_user_Huihui_v3_runner_early_exit_full_logs_20261119.json',
'R_user_Huihui_verified4972_lossless_serial_v4_approval_schema_launch_20261119.json',
'R_user_Huihui_verified4972_lossless_serial_v4_genuine_first_samples_20261119.json',
'R_user_verified4972_live_Huihui_frozen_monitor_checkpoint_20261119.json',
'R_user_Huihui_verified4972_lossless_serial_v4_independent_current_20261119.json.stderr',
'R_user_Huihui_lossless_serial_v4_independent_partial_safe_current_20261119.json',
]
for n in dict.fromkeys(names):
 b=(P/n).read_bytes();assert len(b)<8*1024**2,(n,len(b));packets.append(dict(name=n,bytes=len(b),SHA256=sha(b),base64=base64.b64encode(b).decode()))
H4=B+'/HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119';J4=B+'/HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119';RET=B+'/authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119'
stage_files={
 'cleanup/actual_completion.json':RET+'/actual_completion.json',
 'bytecode_tmp_installers/actual_completion.json':B+'/derived_bytecode_and_inactive_tmp_installers_final_serial_capacity_gap_20261119/actual_completion.json',
 'uv_cache/actual_completion.json':B+'/supported_uv_offline_unused_package_cache_prune_final_serial_gap_20261119/actual_completion.json',
 'current_v4/state.json':H4+'/state.json',
 'current_v4/actual_completion.json':H4+'/actual_completion.json',
 'current_v4/approval.json':H4+'/first200_measurement_approval.json',
 'current_v4/storage_capacity.json':H4+'/full_shape_complete_numeric_storage_capacity_review.json',
 'current_v4/job.json':J4+'/job.json',
 'current_v4/runner_state.json':J4+'/state.json',
 'current_v4/completion.json':J4+'/completion.json',
 'current_v4/fresh3_accepted.json':J4+'/fresh3/accepted.json',
 'current_v4/fresh3_verification_stdout.json':J4+'/fresh3/verification_stdout.json',
 'current_v4/fresh3_verification_stderr.log':J4+'/fresh3/verification_stderr.log',
 'current_v4/formal200_accepted.json':J4+'/formal200/accepted.json',
 'current_v4/formal200_verification_stdout.json':J4+'/formal200/verification_stdout.json',
 'current_v4/formal200_console.log':J4+'/formal200/console.log',
}
transition=dict(receipt=RET+'/actual_completion.json',receipt_SHA256='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3',manifest_SHA256='f955add85f6012e7b1f8f4ad5fc0b78af043b80d2d0797315c21d59b35478382',old_ledger_SHA256='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5',new_ledger_SHA256='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32')
plan=dict(handoff_directory=ptr['handoff_directory'],expected_pointer_SHA256=old['pointer_SHA256'],canonical_SHA256=canonical,authorized_retention_ledger_transition=transition,note_name='user_reaudit4972_exact_cleanup_complete_values_LZMA_real_serial_Huihui_v4_20261119',local_original_packets=packets,official_readback=official,additional_existing_primary_sources=[{k:v for k,v in row.items() if k!='base64'} for row in sources],process_keyword='Huihui',explicit_additional_PIDs=[1565741,1565755,1565767],current_stage_files=stage_files,new_pending=C+'/pending_after_daily_20261119_verified180_verified4972_lossless_Huihui_serial_v4_all_primary_v38.json',reviewed_monitor_inspection_context='Exact frozen-monitor owner-filter exception is runner1565755 cwd/root with real kernel EX, current joba1bf90557a326523990a6004396e2dbe336e85d72139b216fe00f5337340c0fd and matching PPid1565741/startticks/argv; no lost lock/second NN. Two ERROR text matches are the literal SyntaxWarning regex at installed torch_npu/utils/_error_code.py103, full source/log context preserved; no actual model exception there. Unknown new errors never exempted. Fresh3 real3/27 completed and all five formed; current fullNF follows before200. A readonly reader JSONDecodeError was an empty live verifier stdout, fixed only reader partial-file representation. Original model/NF/runner sources untouched. Old waiters self-failed on legitimate ledger transition; v3 approval-schema KeyError occurred before model/lock; v4 backed approval flags and immutable new private paths resolve it, not old-job status fabrication.')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'verified4972_complete_late_handoff_plan_20261119.json').write_bytes(raw)
data=dict(primary_sources=sources,runtime_primary_paths=sorted(runtime_paths),plan_SHA256=sha(raw),plan_base64=base64.b64encode(raw).decode())
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for row in DATA['primary_sources']:
 p=R/row['relative'];b=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R)
 ast.parse(b) if p.suffix=='.py' else json.loads(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw)
for n in DATA['runtime_primary_paths']:
 p=R/n
 assert p.is_file() and p.resolve().is_relative_to(R/'configs');b=p.read_bytes()
 plan['additional_existing_primary_sources'].append(dict(relative=n,SHA256=hashlib.sha256(b).hexdigest(),role='Actual directly consumed new private model YAML/protocol/smoke configuration, original math/source and full numeric evidence preserved'))
H=Path(plan['handoff_directory']);p=H/'verified4972_complete_late_handoff_plan_with_actual_runtime_sources_20261119.json';assert not p.exists();final=(json.dumps(plan,ensure_ascii=False,indent=2)+'\\n').encode();p.write_bytes(final);sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_authorized_retention_current_owned_v6_20261119');print(json.dumps(m.execute(p,hashlib.sha256(final).hexdigest()),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_close_verified4972_complete_sources_current_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),plan_bytes=len(raw),primary_source_files=len(sources),runtime_configuration_paths=sorted(runtime_paths),original_packets=len(packets),new_pending=plan['new_pending']),ensure_ascii=False))

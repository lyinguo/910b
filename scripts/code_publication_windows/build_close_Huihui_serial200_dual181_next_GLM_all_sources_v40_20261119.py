"""Preserve complete live serial stage, next-stage source/queue and all actual failures."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
closed=read('R_close_verified4972_fresh3_accepted_formal200_actual_20261119.json');prior=read('verified4972_fresh3_accepted_formal200_actual_late_plan_20261119.json');official=read('official910b_Huihui_serial161_next_queue_readback_20261119.json');monitor=read('R_Huihui_serial161_next_GLM_queue_full_monitor_20261119.json')
assert official['passed'];R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D+'/code_publication_source_layout_20261114'
models=[
 ('complete_bitplane_shared_npz_20261119.py','universal_eval/utils/complete_bitplane_shared_npz_20261119.py'),
 ('test_complete_bitplane_shared_npz_20261119.py','scripts/tests/test_complete_bitplane_shared_npz_20261119.py'),
 ('diagnostic_glm46_completed3_full_bitplane_exact_sharing_next_serial_20261119.py','scripts/diagnostic_glm46_completed3_full_bitplane_exact_sharing_next_serial_20261119.py'),
 ('run_GLM46_complete_bitplane_storage_diagnostic_actual_20261119.py','scripts/run_GLM46_complete_bitplane_storage_diagnostic_actual_20261119.py'),
 ('publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py','scripts/publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py'),
 ('publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py','scripts/publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py'),
 ('glm46_complete_shape_component_capacity_20261119.py','scripts/glm46_complete_shape_component_capacity_20261119.py'),
 ('glm46_complete_shape_component_capacity_v2_actual_operands_20261119.py','scripts/glm46_complete_shape_component_capacity_v2_actual_operands_20261119.py'),
 ('glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py','scripts/glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py'),
 ('test_GLM46_complete_shape_component_capacity_20261119.py','scripts/tests/test_GLM46_complete_shape_component_capacity_20261119.py'),
 ('test_GLM46_complete_component_capacity_v2_actual_operands_20261119.py','scripts/tests/test_GLM46_complete_component_capacity_v2_actual_operands_20261119.py'),
 ('test_GLM46_component_capacity_v3_actual_one_based_20261119.py','scripts/tests/test_GLM46_component_capacity_v3_actual_one_based_20261119.py'),
 ('review_glm46_completed3_LZMA_complete200_component_shapes_no_NN_20261119.py','scripts/review_glm46_completed3_LZMA_complete200_component_shapes_no_NN_20261119.py'),
 ('review_glm46_completed3_LZMA_complete200_component_shapes_v2_actual_operands_no_NN_20261119.py','scripts/review_glm46_completed3_LZMA_complete200_component_shapes_v2_actual_operands_no_NN_20261119.py'),
 ('review_glm46_completed3_LZMA_complete200_component_shapes_v3_actual_one_based_no_NN_20261119.py','scripts/review_glm46_completed3_LZMA_complete200_component_shapes_v3_actual_one_based_no_NN_20261119.py'),
 ('continue_after_Huihui_real_dual181_then_GLM46_component_capacity_serial_v1_20261119.py','scripts/continue_after_Huihui_real_dual181_then_GLM46_component_capacity_serial_v1_20261119.py'),
 ('glm46flash_own_original_BF16_thinking4096_standard_LZMA_serial_v2_20261119.py','universal_eval/model/adapters/glm46flash_own_original_BF16_thinking4096_standard_LZMA_serial_v2_20261119.py'),
 ('retain_verified181_Huihui_accepted_nonexamples_serial_next_20261119.py','scripts/maintenance/retain_verified181_Huihui_accepted_nonexamples_serial_next_20261119.py'),
 ('run_Huihui_verified181_reviewed_dual_publication_actual_20261119.py','scripts/run_Huihui_verified181_reviewed_dual_publication_actual_20261119.py'),
]
controls=[
 'read_serial_next_own_weight_pending_metadata_20261119_remote.py',
 'read_Pali10_completed_true3_serial_next_full_source_capacity_metadata_20261119_remote.py',
 'read_Pali10_completed_true3_serial_next_v2_actual_lineage_metadata_20261119_remote.py',
 'prepare_Huihui_verified181_exact_original_writer_serial_publication_source_20261119.py',
 'prepare_Huihui_verified181_serial_publisher_v2_next182_metadata_checks_20261119.py',
 'build_deploy_GLM46_complete_bitplane_storage_diagnostic_next_20261119.py',
 'read_Huihui_serial200_and_next_GLM_storage_review_current_20261119_remote.py',
 'prepare_Huihui_verified181_local_baseline_preview_no_predicted_score_20261119.py',
 'huihui_glm46_workbook_cell_verified181_20261119.mjs',
 'read_GLM46_original_complete_capacity_component_metadata_only_20261119_remote.py',
 'build_deploy_GLM46_complete_component_capacity_review_20261119.py',
 'build_GLM46_complete_component_capacity_v2_actual_decode_operands_20261119.py',
 'build_GLM46_component_capacity_v3_actual_one_based_capture_ordinals_20261119.py',
 'read_Huihui_completed200_exact_publication_inputs_no_live_assumptions_20261119_remote.py',
 'prepare_Huihui_verified181_completed200_actual_score_local_plan_20261119.py',
 'apply_Huihui_verified181_exact_server_bytes_local_CAS_20261119.py',
 'read_Huihui_verified181_server_book_for_local_CAS_20261119_remote.py',
 'build_Huihui_verified181_exact_local_receipt_upload_20261119.py',
 'build_stage_and_launch_Huihui_dual181_then_GLM46_serial_queue_20261119.py',
 'build_launch_Huihui_verified181_actual_reviewed_publication_20261119.py',
 'render_Huihui_verified181_actual_published_20261119.mjs',
 'read_Huihui_serial_followup_true_process_and_next_gate_20261119_remote.py',
 'build_official910b_Huihui_serial200_dual181_next_GLM_actual_queue_20261119.py',
 'verify_official910b_Huihui_serial161_next_queue_readback_20261119.py',
 Path(__file__).name,
]
sources=[]
for n,relative in models+[(n,'scripts/code_publication_windows/'+n) for n in controls]:
 b=(P/n).read_bytes()
 if n.endswith('.py'):ast.parse(b)
 sources.append(dict(relative=relative,SHA256=sha(b),base64=base64.b64encode(b).decode(),role='Actual serial continuation, full tensor-shape storage guard, guarded accepted-only retention, staged exact one-G publication and Windows/ArtifactTool consumers; no model math changes; all original failures retained; Day20 review all dependencies'))
packet_names=[n for n,_ in models]+controls+[
 'test_complete_bitplane_shared_npz_original_tempfile_permissions_failure_20261119.py.txt',
 'handoff_Huihui_serial161_dual181_then_GLM_actual_queue_20261119.txt',
 'official910b_Huihui_serial161_next_queue_arguments_20261119.json',
 'official910b_Huihui_serial161_next_queue_tool_result_20261119.json',
 'official910b_Huihui_serial161_next_queue_readback_20261119.json',
 'automation910b_before_Huihui_serial161_next_queue_20261119.toml',
 'automation910b_after_Huihui_serial161_next_queue_20261119.toml',
 'R_user_Huihui_serial_continue_accepted180_monitor_current_20261119.json',
 'R_user_Huihui_serial_continue_actual_current_20261119.json',
 'R_serial_next_own_weight_pending_metadata_20261119.json',
 'R_Pali10_completed_true3_serial_next_full_source_capacity_metadata_20261119.json.stderr',
 'R_Pali10_completed_true3_serial_next_v2_actual_lineage_metadata_20261119.json',
 'Huihui_verified181_publisher_scope_review_20261119.json',
 'Huihui_verified181_publisher_v2_scope_checks_20261119.json',
 'R_GLM46_complete_bitplane_storage_diagnostic_next_actual_launch_20261119.json',
 'R_Huihui_serial200_next_GLM_storage_current4_20261119.json',
 'R_GLM46_original_complete_capacity_component_metadata_only_20261119.json',
 'R_GLM46_complete_component_capacity_review_20261119.json',
 'R_GLM46_complete_component_capacity_review_v2_actual_operands_20261119.json',
 'R_GLM46_complete_component_capacity_review_v3_actual_one_based_20261119.json',
 'Huihui_dual181_then_GLM46_serial_queue_source_scope_review_20261119.json',
 'R_Huihui_dual181_then_GLM46_serial_queue_actual_launch_20261119.json',
 'R_Huihui_serial_followup_true_process_and_next_gate_20261119.json',
 'R_Huihui_serial200_next_GLM_storage_current7_20261119.json',
 'R_Huihui_serial161_next_GLM_queue_full_monitor_20261119.json',
]
packets=[]
for n in dict.fromkeys(packet_names):
 b=(P/n).read_bytes();assert len(b)<8*1024**2;packets.append(dict(name=n,bytes=len(b),SHA256=sha(b),base64=base64.b64encode(b).decode()))
# The local original workbook is already backed up; add only its real authoring preview evidence.
for n in ['preview_plan.json','before_355.png','original_target_view.ndjson']:
 b=(P/'HuihuiGLM46Flash_local_result_sync_verified181_20261119'/n).read_bytes();packets.append(dict(name='Huihui_original_local_preview/'+n,bytes=len(b),SHA256=sha(b),base64=base64.b64encode(b).decode()))
J=B+'/HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119';NEXT=B+'/after_Huihui_real_dual181_then_GLM46_own_completed3_component_capacity_serial_v1_20261119';PUB=B+'/HuihuiGLM46Flash_original_BF16_2048_lossless_verified181_blank_G_publication_20261119'
stage=dict(prior['current_stage_files']);stage.update({
 'serial_next/state.json':NEXT+'/state.json',
 'serial_next/actual_completion.json':NEXT+'/actual_completion.json',
 'serial_next/NOTIFY_error.json':NEXT+'/NOTIFY_error.json',
 'serial_next/first200_measurement_approval.json':NEXT+'/first200_measurement_approval.json',
 'serial_next/retention_actual.json':NEXT+'/current_Huihui_retention_actual.json',
 'component_capacity/review.json':B+'/GLM46Flash_completed_true3_existing_full_LZMA_all200_component_shape_capacity_review_v3_actual_one_based_20261119/review.json',
 'bitplane_diagnostic/review.json':B+'/GLM46Flash_completed3_complete_bitplane_exact_sharing_storage_for_immediate_serial_next_20261119/review.json',
 'publication/dual_completed.json':PUB+'/dual_publication_completed.json',
 'publication/full_target_audit.json':PUB+'/full_target_dual_publication_audit.json',
})
plan=dict(handoff_directory=closed['handoff_directory'],expected_pointer_SHA256=closed['pointer_SHA256'],canonical_SHA256=monitor['canonical_SHA256'],note_name='Huihui_live_serial200_exact_dual181_publication_and_GLM46_next_real_waiter_all_sources_20261119',local_original_packets=packets,official_readback=official,additional_existing_primary_sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],process_keyword='Huihui',current_stage_files=stage,new_pending=C+'/pending_after_daily_20261119_verified180_Huihui_serial200_dual181_GLM46_component_next_all_primary_v40.json')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'Huihui_serial200_dual181_GLM_next_complete_late_plan_v40_20261119.json').write_bytes(raw)
data=dict(sources=sources,plan_base64=base64.b64encode(raw).decode(),plan_SHA256=sha(raw))
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for row in DATA['sources']:
 p=R/row['relative'];b=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R)
 if p.suffix=='.py':ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw)
n='configs/queues/Huihui_real_dual181_then_own_GLM46_complete_component_capacity_serial_20261119.json';p=R/n;b=p.read_bytes();plan['additional_existing_primary_sources'].append(dict(relative=n,SHA256=hashlib.sha256(b).hexdigest(),role='Real running serial queue consumer configuration; max1, exact parent/next proof/source and current accepted-only retention authorization'))
H=Path(plan['handoff_directory']);path=H/'Huihui_serial200_dual181_GLM_next_complete_late_plan_v40_20261119.json';assert not path.exists();body=(json.dumps(plan,ensure_ascii=False,indent=2)+'\\n').encode();path.write_bytes(body)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118');print(json.dumps(m.execute(path,hashlib.sha256(body).hexdigest()),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_close_Huihui_serial200_GLM_next_all_sources_v40_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),plan_bytes=len(raw),new_primary_sources=len(sources),original_packets=len(packets),new_pending=plan['new_pending']),ensure_ascii=False))

from pathlib import Path
import ast, base64, hashlib, json
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
prior=read('verified182_AutoSingle_recovery_current_late_plan_v47_20261120.json')
closed=read('R_close_verified182_AutoSingle_current_sources_v47_20261120.json')
current=read('R_AutoGLMPhone9BSingle_verified154_retention_fresh3_fixed200_v3_actual3_20261120.json')
sources={v['relative']:dict(relative=v['relative'],SHA256=v['SHA256'],role='Actual own Single bound runtime source or generated standalone config/protocol/smoke') for v in read('R_AutoGLMPhone9BSingle_verified154_bound_runtime_sources_latest_20261120.json')['actual_bound_runtime_sources']}
install=[]
working={
 'scripts/maintenance/retain_verified182_GLM_accepted_correct_nonexamples_continue_Single_serial_20261120.py':'retain_verified182_GLM_accepted_correct_nonexamples_continue_Single_serial_20261120.py',
 'scripts/continue_AutoGLMPhone9B_single_completed_true3_verified182_GLM_retention_first200_serial_v3_20261120.py':'continue_AutoGLMPhone9B_single_completed_true3_verified182_GLM_retention_first200_serial_v3_20261120.py',
 'configs/retention/verified182_GLM_accepted_correct_nonexamples_continue_Single_serial_20261120.json':'verified182_GLM_retention_actual_consumer_20261120.json',
 'configs/queues/AutoGLMPhone9BSingle_completed_true3_verified182_GLM_retention_first200_serial_v3_20261120.json':'AutoGLMPhone9BSingle_completed_true3_verified182_GLM_retention_first200_serial_v3_20261120.json',
}
for rel,n in working.items():sources[rel]=dict(relative=rel,SHA256=sha((P/n).read_bytes()),role='Actually executed exact accepted correct-only retention or own completed3 serial continuation and real consumed config')
extras=['prepare_daily20_qualified182_source_snapshot_v2_and_full_diff_20261120.py','prepare_daily20_actual_plan_and_full_review_manifest_20261120.py','build_verify_daily20_R_source_CAS_before_publish_20261120.py','invoke_daily_code_publisher_20261120.py','build_daily20_verified_push_receipt_and_R_policy_CAS_20261120.py','read_verified182_new_accepted_GLM_AutoMulti_retention_inventory_only_20261120_remote.py','read_verified182_complete_publication_retention_source_schema_20261120_remote.py','read_exact_verified182_GLM_correct_nonexample_retention_plan_20261120_remote.py','apply_exact_reviewed_verified182_GLM_correct_nonexample_retention_20261120_remote.py','read_AutoGLMPhone9BSingle_verified154_retention_fresh3_fixed200_serial_v3_actual_20261120_remote.py',Path(__file__).name]
closer='close_alignment_handoff_exact_verified182_Day20_policy_GLM154_ledger_current_v13_20261120.py'
for n in extras+[closer]:
 raw=(P/n).read_bytes();ast.parse(raw)
 rel=('scripts/maintenance/' if n==closer else 'scripts/code_publication_windows/')+n
 v=dict(relative=rel,SHA256=sha(raw),role='Exact bounded source collector/reviewer/test/CAS or full metadata reader and receipt-bound handoff closer; not a model change')
 sources[rel]=v;install.append(dict(v,base64=base64.b64encode(raw).decode()))
full='full_actual_verified182_Day20_pushed_GLM154_exact_retention_Single_true3_pass_first200_serial_20261120.txt'
originals=extras+[closer,full,'R_AutoGLMPhone9BSingle_verified154_bound_runtime_sources_latest_20261120.json','R_AutoGLMPhone9BSingle_current_v1_v2_actual2_20261120.json','R_AutoGLMPhone9BSingle_current_v1_v2_actual3_20261120.json','R_AutoGLMPhone9BSingle_verified154_retention_fresh3_fixed200_v3_actual1_20261120.json','R_AutoGLMPhone9BSingle_verified154_retention_fresh3_fixed200_v3_actual2_20261120.json','R_AutoGLMPhone9BSingle_verified154_retention_fresh3_fixed200_v3_actual3_20261120.json','R_verified182_Single_serial_v3_frozen_monitor_complete_20261120.json','R_verified182_GLM_retention_full_reviewed_plan_20261120.json','R_verified182_GLM_retention_review_plan_actual_20261120.json','R_verified182_GLM_retention_exact_apply_actual_20261120.json','verified182_GLM_retention_local_exact_review_20261120.json','verified182_GLM_retention_selection_tests_20261120.json','R_launch_AutoGLMPhone9BSingle_completed_true3_verified182_GLM_retention_serial_v3_20261120.json','R_daily20_current_declared_primary_sources_v2_20261120.json','daily_20261120_source_content_review_v2.json','daily_20261120_full_source_diff_v2.patch','daily_20261120_plan_review.json','daily_20261120_publish.stdout.json','daily_20261120_publish.stderr.log','daily_20261120_receipt.json','R_daily20_R_source_CAS_before_publish_actual_20261120.json','R_daily20_actual_code_push_and_reviewed_R_publication_policy_20261120.json']
packets=[]
for n in dict.fromkeys(originals):
 raw=(P/n).read_bytes()
 if len(raw)<8*1024**2:parts=[(n,raw)]
 else:
  parts=[(n+f'.raw_part_{i//(4*1024**2):03d}',raw[i:i+4*1024**2]) for i in range(0,len(raw),4*1024**2)]
  manifest=(json.dumps(dict(original_name=n,original_bytes=len(raw),original_SHA256=sha(raw),all_original_bytes_preserved=True,parts=[dict(name=a,bytes=len(b),SHA256=sha(b)) for a,b in parts]),indent=2)+'\n').encode();parts.append((n+'.raw_parts_manifest.json',manifest))
 for name,raw in parts:packets.append(dict(name=name,bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=closed['handoff_directory']
plan={k:v for k,v in prior.items() if k not in ['local_original_packets','additional_existing_primary_sources']}
plan.update(canonical_SHA256={**prior['canonical_SHA256'],**current['canonical_SHA256']},expected_pointer_SHA256=closed['pointer_SHA256'],note_name='verified182_Day20_pushed_GLM154_exact_retention_Single_passed3_approved_fresh3_fixed200_serial_v48_20261120',local_original_packets=packets,additional_existing_primary_sources=list(sources.values()),new_pending=D+'/code_publication_source_layout_20261114/pending_after_daily_20261120_verified182_GLM154_retention_Single_completed3_first200_serial_all_primary_v48.json',new_full_current_handoff=H+'/'+full,reviewed_Day20_publication_transition=dict(receipt=D+'/code_publication_source_layout_20261114/daily_20261120_actual_reviewed_commit_remote_receipts/daily_20261120_receipt.json',receipt_SHA256='49bf894593e1c8bc82040382f9168dbcc53dd3d33c7245a35135e6092f127337'),reviewed_exact_GLM154_cleanup_receipt=B+'/verified182_GLM_accepted_correct_nonexamples_user_serial_Single_capacity_recovery_20261120/actual_completion.json')
j=B+'/AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120';h=B+'/AutoGLMPhone9BSingle_completed_true3_verified182_GLM_retention_first200_serial_v3_20261120';ret=B+'/verified182_GLM_accepted_correct_nonexamples_user_serial_Single_capacity_recovery_20261120'
plan['current_stage_files'].update({'full_current182_Day20_retention_serial_handoff':H+'/'+full,'Single/current_J200_job':j+'/job.json','Single/current_J200_state':j+'/state.json','Single/current_J200_completion':j+'/completion.json','Single/current_first200_controller_state':h+'/state.json','Single/current_first200_controller_actual':h+'/actual_completion.json','Single/current_first200_capacity':h+'/own_passed3_complete_shape_capacity_review.json','Single/current_first200_approval':h+'/own_first200_measurement_approval.json','GLM154_exact_retention_actual':ret+'/actual_completion.json','Day20_actual_receipt':plan['reviewed_Day20_publication_transition']['receipt']})
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();name='verified182_Day20_GLM154_Single_serial_late_plan_v48_20261120.json';(P/name).write_bytes(raw)
data=dict(install=install,plan_name=name,plan_base64=base64.b64encode(raw).decode(),plan_SHA256=sha(raw),full_name=full,full_base64=base64.b64encode((P/full).read_bytes()).decode(),full_SHA256=sha((P/full).read_bytes()))
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for v in DATA['install']:
 p=R/v['relative'];raw=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['SHA256'] and p.resolve().is_relative_to(R)
 if p.suffix=='.py':ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw);H=Path(plan['handoff_directory']);full=base64.b64decode(DATA['full_base64'],validate=True);assert hashlib.sha256(full).hexdigest()==DATA['full_SHA256'];p=H/DATA['full_name'];assert not p.exists();p.write_bytes(full);p=H/DATA['plan_name'];assert not p.exists();p.write_bytes(raw)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.close_alignment_handoff_exact_verified182_Day20_policy_GLM154_ledger_current_v13_20261120');print(json.dumps(m.execute(p,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_close_verified182_Day20_GLM154_Single_serial_v48_20261120_remote.py').write_text(remote,encoding='utf-8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),full_SHA256=sha((P/full).read_bytes()),plan_bytes=len(raw),additional_primary_sources=len(sources),original_packets=len(packets))))

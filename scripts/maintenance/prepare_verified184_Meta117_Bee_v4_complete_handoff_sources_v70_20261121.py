"""Preserve useful exact source/config bytes and a current complete operational handoff."""
from pathlib import Path,PurePosixPath
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
old=V/'record_serial_completed_NPU200_fullNF_publication_handoff_20261121.py'
s=old.read_text('utf8')
a="assert all(receipt['canonical_before_sha256'].get(p)==old and audit['canonical_SHA256'].get(p)==new for p,(old,new) in changed.items())"
b="""for p,(old,new) in changed.items():
   if p==str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'):
    ret=B/'verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121'
    done=js(ret/'actual_completion.json')
    assert old=='6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c' and new=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd'
    assert sha(ret/'actual_completion.json')=='c2ed3e30c9f71b4695ea3dfa35630834f4b9752fc58e13f84c1d64a900c63d3c' and sha(ret/'ledger_before.original')==old
    assert done['actual_returncode']==0 and done['removed_files']==117 and done['all_protected_SHA256_unchanged'] and not done['protected_errors']
    assert sha(done['manifest'])==done['manifest_SHA256']=='363e58f7b5e5463c9761df7b11c6780aa9ff0774f89a0f650e440c53c4facc59'
   else:assert receipt['canonical_before_sha256'].get(p)==old and audit['canonical_SHA256'].get(p)==new"""
assert a in s;s=s.replace(a,b)
s=s.replace("'Bee_serial_v2':stage(B/", "'Bee_capacity_v3':stage(B/'BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121'),\n 'Bee_capacity_wait_v4':stage(B/'BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121'),\n 'Meta117_retention':stage(B/'verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121'),\n 'Bee_serial_v2':stage(B/")
s=s.replace("'publication_receipt.json']", "'publication_receipt.json','capacity_review.json','completed_storage_proof_reuse.json']")
s=s.replace("no_signal_hot_edit_NN_replay_or_cleanup=True", "no_signal_hot_edit_NN_replay_or_deletion_in_this_recorder=True")
s=s.replace("current_actual_stage='Ovis_fullNF2_completed_or_actual_serial_fullNF_with_conditional_dual_publication_and_Bee_v2'", "current_actual_stage='verified184_Ovis_delivered_Meta117_retention_complete_Bee_v4_actual_capacity_wait_no_formal_NN_or_NF'")
s=s.replace("current_formal_model_stage='actual serial fullNF and exact conditional publication; see current snapshot'", "current_formal_model_stage='Bee full storage NF already passed; complete200 capacity blocked since 20261121 20:35:24 BJT; v4 capacity-only wait; see actual snapshot'")
s=s.replace("current_verified_count=counts['done'])", "current_verified_count=counts['done'],numeric_within5pct=169,numeric_outside5pct=15,complete_weights_pending_formal=206,confirmed_missing=310,complete_original_protocol_qualified=70,current_stop_earliest_proven_Beijing='2026-11-21T20:35:24.837093+08:00',current_Bee_wait_home=str(B/'BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121'),current_retention_ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'))")
compile(s,'record_verified184_Meta117_Bee_v4_handoff_v2_20261121.py','exec')
rec='scripts/maintenance/record_verified184_Meta117_Bee_v4_handoff_v2_20261121.py'
(V/PurePosixPath(rec).name).write_text(s,encoding='utf8',newline='\n')
existing=[
 'scripts/continue_BeeSFT_completed_full_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.py',
 'configs/queues/BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.json',
 'scripts/continue_BeeSFT_completed_storage_NF_wait_actual_capacity_v4_serial_20261121.py',
 'configs/queues/BeeSFT_completed_storage_NF_capacity_v4_wait_actual_capacity_serial_20261121.json',
 'scripts/maintenance/discover_verified184_current_full_accepted_and_Meta_fullNF4_nonexamples_Bee_capacity_20261121.py']
for version in ['', '_v2_current_ledger', '_v3_actual_groups']:
 existing+=['scripts/maintenance/retain_verified184_Meta_fullNF4_correct_nonexample117'+version+'_Bee_capacity_20261121.py']
cfg=['configs/retention/verified184_Meta_fullNF4_correct_nonexample117_user_Bee_capacity_20261121.json','configs/retention/verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_20261121.json','configs/retention/verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_20261121.json']
existing+=cfg
names=[
 'prepare_BeeSFT_capacity_resume_v3_actual_saved_patches_shape_20261121.py',
 'prepare_BeeSFT_completed_storage_capacity_v4_immediate_serial_20261121.py',
 'prepare_Meta_fullNF4_correct_nonexample117_retention_20261121.py','prepare_Meta117_retention_v2_exact_current_ledger_20261121.py','prepare_Meta117_retention_v3_actual_group_layout_20261121.py',
 'prepare_verified184_current_full_accepted_cleanup_discovery_20261121.py','prepare_Bee_v4_capacity_reader_v2_20261121.py',
 'read_Bee_completed_storage_NF_actual_pixel_metadata_capacity_recovery_20261121_remote.py','read_Bee_v3_actual_capacity_or_runner_20261121_remote.py','read_BeeSFT_v2_actual_storage_fullNF_capacity_process_20261121_remote.py',
 'read_Bee_v4_wait_actual_capacity_and_existing_dedup_sources_20261121_remote.py','read_Bee_v4_wait_actual_capacity_and_existing_dedup_sources_v2_20261121_remote.py',
 'read_Meta117_v3_plan_actual_process_status_20261121_remote.py','read_Meta_retention_plan_prewrite_canonical_mismatch_20261121_remote.py','apply_Meta117_v3_exact_reviewed_plan_20261121_remote.py',
 'read_verified184_post_Meta117_frozen_monitor_storage_and_process_20261121_remote.py','read_original_resource_gate_and_project_storage_usage_20261121_remote.py','read_LlavaMed_and_LlavaMore_own_original_resources_metadata_20261121_remote.py','read_verified184_export_and_report_stat_exact_20261121_remote.py','read_Ovis_fullNF2_conditional_dual_and_Bee_actual_20261121_remote.py']
records=[]
for n in names+[PurePosixPath(rec).name,Path(__file__).name]:
 p=V/n;data=p.read_bytes();ast.parse(data);records.append(dict(path='scripts/maintenance/'+n,SHA256=sha(data),base64=base64.b64encode(data).decode()))
primary=existing+[x['path'] for x in records]
(V/'verified184_Meta117_Bee_v4_all_new_primary_v70_20261121.json').write_text(json.dumps(primary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
deploy='DATA='+repr(records)+'\nEXISTING='+repr(existing)+r'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json,base64
R=Path('/root/projects/comperror');out=[]
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
 out.append({'path':x['path'],'SHA256':x['SHA256'],'bytes':len(b)})
for rel in EXISTING:
 p=R/rel;assert p.is_file();out.append({'path':rel,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
print(json.dumps(dict(passed=True,primary=out,source_only_no_model_or_canonical_change=True),ensure_ascii=False))
'''
(V/'stage_verified184_Meta117_Bee_v4_complete_handoff_sources_v70_20261121_remote.py').write_text(deploy,encoding='utf8',newline='\n')
print(json.dumps({'prepared':True,'new_primary':len(primary),'new_recorder_SHA256':sha(s.encode()),'new_source_bytes':sum(len(base64.b64decode(x['base64'])) for x in records)}))

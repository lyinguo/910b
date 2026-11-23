"""Append exact completed-stage, cleanup and official receipts without rewriting accepted evidence."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, os, subprocess
R = Path('/root/projects/comperror'); D = R/'workspace/diagnostics/batch_910b'
C = D/'code_publication_source_layout_20261114'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def js(p): return json.loads(Path(p).read_bytes())
def put(p, x):
    p = Path(p); t = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf8'); os.replace(t,p)

def main():
    p = Path(sys.argv[1]); assert sha(p)==sys.argv[2]; q=js(p)
    pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'; pb=pointer.read_bytes()
    assert hashlib.sha256(pb).hexdigest()=='97a980f6eda2d1f6dbbbf1101f9c096e99e38e35359d4227d266559c24e27c9b'
    ptr=json.loads(pb); closing=Path(ptr['handoff_directory'])/'closing_addendum.json'; cb=closing.read_bytes()
    assert hashlib.sha256(cb).hexdigest()=='aa3069e5ac5fe3c218b63debbb4a7f5d25277e86426fdeba19498a3cbff56007'
    parent=Path(ptr['new_primary_pending']); assert sha(parent)=='bc5f4275df7abc884b3f021f096cb0a5934ce061fb6d7843f310dab587059ef8'
    rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows)==2964
    policy=R/'configs/publication/code_only.json';policy_row=dict(rows[str(policy)])
    syncpath=C/'daily_20261123_actual_reviewed_commit_remote_receipts/R_policy_exact_CAS_actual.json'
    sync=js(syncpath);assert sync['passed'] and sync['commit']==sync['remote_commit']=='5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9'
    assert sync['reviewed_plan_SHA256']=='66dad86f3b449fc5f3b66479c8157a837678c6657dc80af101572e92d818b992'
    assert sync['policy_before_SHA256']==policy_row['SHA256']=='39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5'
    assert sync['policy_after_SHA256']==sha(policy)=='1585e1dce7edea8a7b3c99bf27ae4b2840b715ac3b054542d55141bbc44906e9'
    assert all(sha(n)==r['SHA256'] for n,r in rows.items() if n!=str(policy))

    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
    note=Path(ptr['handoff_directory'])/'late_actual_notes'/'verified185_Day23_actual_commit_resource_blocked_full_actual_v90_20261123'
    note.mkdir(exist_ok=False); (note/'previous_pointer.json').write_bytes(pb); (note/'previous_closing_addendum.json').write_bytes(cb)
    local=[]
    for z in q['local_original_records']:
        b=base64.b64decode(z['base64'],validate=True); assert hashlib.sha256(b).hexdigest()==z['SHA256']
        dest=note/'local_original_records'/z['name']; assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(b)
        local.append(dict(path=str(dest),SHA256=z['SHA256'],bytes=len(b)))
    pdest=note/'primary_source_snapshots/configs/publication/code_only.json';pdest.parent.mkdir(parents=True,exist_ok=True);pdest.write_bytes(policy.read_bytes())
    rows[str(policy)]=dict(policy_row,SHA256=sha(policy),bytes=policy.stat().st_size,snapshot=str(pdest),previous_exact_source_row=policy_row,authorized_policy_only_transition_receipt=str(syncpath),authorized_policy_only_transition_receipt_SHA256=sha(syncpath))
    for n,h in q['new_primary_expected_SHA256'].items():
        p=R/n; assert n.startswith(('scripts/','configs/','universal_eval/')) and p.resolve().is_relative_to(R) and sha(p)==h,n
        if str(p) in rows: assert rows[str(p)]['SHA256']==h
        else:
            dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='Actual completed Turkish fullNF/capacity/cleanup metadata read-only and official original heartbeat maintenance controls, true consumers and full source preservation')
    reader=R/'scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py'
    r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode==0,r.stderr;actual=json.loads(r.stdout);x=actual['all_original_and_current_actual']['same_GLM41Base_only_actual_serial_model']
    assert x['cards_returncode']==0 and not x['frozen_monitor_summary']['source_errors']
    snapshot=note/'completed_Turkish_fullNF_capacity_current_actual.json';put(snapshot,actual)
    records=actual['Turkish_serial_v2_and_Base_CPU_diagnostic_complete_records']
    td=[z['complete_value'] for k,z in records.items() if k.endswith('TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122/completion.json')]
    assert len(td)==1 and td[0]['actual_returncode']==0
    accepted,=td[0]['accepted'];assert accepted['sample_count']==3 and accepted['proof_sha256']=='cf89fde1ab3067eba798cde73d0cd37f2798bc21d283b86c0bcabd4e6086a4a1'
    assert sha(accepted['proof'])==accepted['proof_sha256']
    proof=js(accepted['proof']);assert proof['passed'] and proof['actual_forward_count']==6
    assert not proof['original_protocol_final_qualification_complete']
    capname='TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122/own_passed3_complete_all200_shape_capacity.json'
    caps=[(p,z) for p,z in records.items() if p.endswith(capname)];assert len(caps)==1
    capfile,caprec=caps[0];cap=caprec['complete_value']
    assert caprec['SHA256']=='a029a47b242df3e0b411e501a1f79a5cef4f604d77c6633dcce9230f15d22605'
    assert cap['empirical_estimate_bytes']==4228496264073 and not cap['capacity_passed']
    discovery=Path(q['cleanup_discovery']);assert sha(discovery)==q['cleanup_discovery_SHA256'];clean=js(discovery)
    assert clean['candidate_count']==0 and clean['candidate_bytes']==0 and clean['no_deletion'] and not clean['temporary_suffix_files_preserved_pending_actual_scope_verification']
    for z in [Path(accepted['proof']),Path(capfile),discovery]:
        dst=note/'complete_current_original_records'/z.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read_bytes())
    states={p:z['complete_value'] for p,z in records.items() if p.endswith(('completion.json','actual_completion.json','state.json'))}
    procs=x['actual_proc_argv_cwd_PPid_startticks_FD_io']
    models=[p for p in procs if 'run.py' in p['argv']]
    nf=[p for p in procs if any(Path(a).name.startswith('verify_') for a in p['argv'])]
    compact=dict(UTC=actual['UTC'],counts=dict(delivered=185,numeric_within=170,numeric_outside=15,complete_weights_pending=205,confirmed_missing=310,full_original_qualification=71),
        Turkish_completed_genuine3=dict(report_dir=proof['report_dir'],sample_count=3,native_forwards=6,fullNF=0,proof=str(accepted['proof']),proof_SHA256=accepted['proof_sha256'],runner_completion=td[0]),
        Turkish200_capacity=dict(path=capfile,SHA256=caprec['SHA256'],required_bytes=cap['empirical_estimate_bytes'],actual_free_at_review=cap['actual_free_bytes'],capacity_passed=False,no_approval_fresh3_200_G=True),
        cleanup=dict(discovery=str(discovery),SHA256=sha(discovery),candidate_count=0,candidate_bytes=0,temporary_count=0,actual_deletions=0),
        official_actual_readback=q['official_actual_readback'],full_primary_source_count=len(rows),parent_pending=str(parent),parent_pending_SHA256=sha(parent),
        current_model_processes=models,current_NF_processes=nf,current_original_EX=x['actual_original_lock_owners'],actual_free_bytes=x['actual_original_disk_free_bytes'],
        complete_monitor_summary=x['frozen_monitor_summary'],current_states=states,full_actual_snapshot=str(snapshot),full_actual_snapshot_SHA256=sha(snapshot),
        local_original_records=local,canonical_SHA256=ptr['canonical_SHA256'],no_new_complete200_G_or_qualification=True,no_deletion=True,no_second_Day23_commit=True)
    bee_cfg=R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json'
    assert sha(bee_cfg)==rows[str(bee_cfg)]['SHA256']
    bq=js(bee_cfg);wait=Path(bq['wait_home']);bs=js(wait/'state.json')
    assert bq['required_free_bytes']==192555984252 and bs['required_free_bytes']==192555984252
    compact['existing_BeeSFT_v5_current_actual']=dict(config=str(bee_cfg),config_SHA256=sha(bee_cfg),wait_home=str(wait),state=bs,new_NN_job_exists=Path(bq['new_job']).exists(),new_continuation_home_exists=Path(bq['home']).exists(),not_NN_or_formal_NF=True)
    compact['original_three_resources_fresh_readonly']=q['original_three_resources_fresh_readonly']
    compact['same_owned_ID_alternative_exact_class_locations_only']=q['original_three_owned_exact_locations']
    compact['legacy_schema_full_numeric_reaudits']=q['legacy_schema_reaudits']
    compact['auxiliary_v1_KeyError_preserved_and_v2_original_registry_path_actual0']=q['auxiliary_reader_failure']
    compact['numeric_physical_allocation_metadata_only']=q['numeric_allocation']
    compact['current_local_workbook_and_original_automation_readonly']=q['current_local_and_automation']
    compact['no_new_physical_storage_method_adopted_or_protected_matrix_changed']=True
    compact['no_official_automation_write_before_this_snapshot']=True
    compact['Day23_remote_verified_code_receipt']=q['Day23_remote_verified_code_receipt']
    compact['approved_R_publication_policy_only_transition']=sync
    compact['R_publication_policy_transition_receipt_SHA256']=sha(syncpath)
    compact['Day23_directory_allocation_metadata']=q['Day23_directory_allocation_metadata']
    compact['Day23_R_test_entry_v1_failed_before_tests_v2_correct_original_source_arg_passed39']=True
    compact['today_one_code_commit_not_new_model_NPU_acceptance']=True
    facts=note/'facts.json';put(facts,compact)
    text=q['full_handoff_template']
    full=note/'full_actual_verified185_Day23_commit_resource_blocked_v90_20261123.txt'
    full.write_text(text.replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_FACTS_SHA256',sha(facts)).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False)),encoding='utf8')
    pending=C/'pending_after_daily_20261123_verified185_Day23_commit_resource_blocked_all_primary_v90_20261123.json'
    assert not pending.exists();put(pending,dict(UTC=actual['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[n] for n in sorted(rows)],Code_Day_already_pushed='2026-11-23',Code_Day_commit='5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9',no_second_same_day_commit=True,next_daily_review='Day24 explicit useful main sources, true config consumers and dependencies, full diff/outgoing and tests plus real publisher plan_SHA, one ordinary push then verify remote'))
    upd=dict(UTC=actual['UTC'],new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_primary_source_count=len(rows),current_source_count=len(rows),latest_late_actual_note=str(full),latest_late_actual_note_SHA256=sha(full),latest_actual_snapshot=str(snapshot),latest_actual_snapshot_SHA256=sha(snapshot),current_completed_stage_facts=str(facts),current_completed_stage_facts_SHA256=sha(facts),current_model_process_active=bool(models),current_full_numerical_verifier_active=bool(nf),current_model_NN_or_NF_active=bool(models or nf),current_actual_model_NN_or_NF_active=bool(models or nf),current_NN_or_NF_active=bool(models or nf),current_formal_model_task_active=bool(models or nf),current_actual_model_process_or_NF_active=bool(models or nf),current_next_original_model_stage='Turkish same genuine3 fullNF0 completed; fixed200 capacity refused, do not replay. Actual original disk resource required; no active NN/NF unless later real process proves recovery',current_Turkish_fullNF0_proof_SHA256=accepted['proof_sha256'],current_Turkish_capacity=compact['Turkish200_capacity'],current_cleanup_readonly_actual=compact['cleanup'],current_official_actual_readback=q['official_actual_readback'],current_actual_model_processes=models,current_actual_NF_processes=nf,current_original_EX=x['actual_original_lock_owners'],current_resource_stop_earliest_UTC='2026-11-22T09:40:49.156504+00:00',current_full_actual_note=str(full),current_full_actual_note_SHA256=sha(full),current_full_facts=str(facts),current_full_facts_SHA256=sha(facts),current_BeeSFT_existing_waiter_actual=compact['existing_BeeSFT_v5_current_actual'],current_legacy_numeric_schema_readonly_reaudits=q['legacy_schema_reaudits'],current_original_three_resources_readonly=q['original_three_resources_fresh_readonly'],current_numeric_allocation_metadata=q['numeric_allocation'])
    assert pointer.read_bytes()==pb and closing.read_bytes()==cb
    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items()) and all(sha(n)==r['SHA256'] for n,r in rows.items())
    co=json.loads(cb);co.update(upd);put(closing,co);ptr.update(upd);ptr['closing_addendum_SHA256']=sha(closing);put(pointer,ptr)
    result=dict(passed=True,UTC=actual['UTC'],pointer_SHA256=sha(pointer),closing_SHA256=sha(closing),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),late_actual_note=str(full),late_actual_note_SHA256=sha(full),facts=str(facts),facts_SHA256=sha(facts),actual_snapshot_SHA256=sha(snapshot),official_actual_readback=q['official_actual_readback'],genuine3_fullNF0=True,capacity_passed=False,models=models,full_NF=nf,EX=x['actual_original_lock_owners'],actual_free_bytes=x['actual_original_disk_free_bytes'],monitor_summary=x['frozen_monitor_summary'],no_deletion=True,no_G_or_canonical_mutation=True,no_second_Day23_commit=True)
    put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()

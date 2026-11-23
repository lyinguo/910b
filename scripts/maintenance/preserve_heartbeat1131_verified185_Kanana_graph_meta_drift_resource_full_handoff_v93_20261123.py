"""Preserve exact current evidence and useful sources; never alter model evidence."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
C=D/'code_publication_source_layout_20261114'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
    plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan)
    pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';pb=pointer.read_bytes()
    assert hashlib.sha256(pb).hexdigest()=='17c7404eccf2784ef9c7647e3e791116170abb5c4b212b00ed9198ccbe6e20f1'
    ptr=json.loads(pb);closing=Path(ptr['handoff_directory'])/'closing_addendum.json';cb=closing.read_bytes()
    assert hashlib.sha256(cb).hexdigest()=='d3c900e5833326cded6a65c27f6031e7477bc44af1de07662863982a09075a79'
    parent=Path(ptr['new_primary_pending']);assert sha(parent)=='f3dc4620f1d8417c003003298e2cfd13464a20922175df4e1ddc1135d66f0b8d'
    rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows)==2987 and all(sha(p)==z['SHA256'] for p,z in rows.items())
    assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())
    assert q['official_actual_readback']['passed'] and q['official_actual_readback']['status']=='ACTIVE'
    assert q['official_actual_readback']['actual_TOML_SHA256']=='590801a94c1493e88f304f20f35bf5f31fd3a77e5548f346c735179ca0c52929'
    note=Path(ptr['handoff_directory'])/'late_actual_notes/verified185_heartbeat1131_Kanana_graph_meta_CPU_drift_resource_full_actual_v93_20261123'
    note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(pb);(note/'previous_closing_addendum.json').write_bytes(cb)
    originals=[]
    for z in q['local_original_records']:
        b=base64.b64decode(z['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==z['SHA256']
        dest=note/'local_original_records'/z['name'];assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        originals.append(dict(path=str(dest),SHA256=z['SHA256'],bytes=len(b)))
    # Reconstruct the byte-identical original AA stdout from its already stored
    # complete JSON control plan; require the original stdout SHA before writing.
    original_controls=D/'heartbeat_handoff_plans/Kanana_private_original_RegStage_complete_synthetic_control_and739_meta_v2_original_decorators_plan_20261123.json'
    assert sha(original_controls)=='02d4ae1b5dae2dd07db8286ff7b3d53d8945b92a35d5952b2391ff759d62e26e'
    raw=(json.dumps(js(original_controls)['complete_AA_synthetic_graph_controls'],ensure_ascii=False)+'\n').encode()
    assert hashlib.sha256(raw).hexdigest()=='4224939683d43e18b41f04fb25cd59a6145baf2f16c2abd69cca5ef0fb09d5c5'
    dest=note/'local_original_records/AA_Kanana_own_RegStage_weightfree_BF16_operator_graph_v5_full_control_values_20261123.json';dest.write_bytes(raw)
    originals.append(dict(path=str(dest),SHA256=sha(dest),bytes=len(raw),byte_identical_original_stdout_recovered_from_already_preserved_control_plan=True))
    for n,h in q['new_primary_expected_SHA256'].items():
        p=R/n;assert p.resolve().is_relative_to(R) and n.startswith(('scripts/','configs/','universal_eval/')) and sha(p)==h
        if str(p) in rows:assert rows[str(p)]['SHA256']==h
        else:
            dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),
                role='Actual Kanana private original graph interfaces, strict synthetic control failures, meta and filesystem diagnostics, readonly resource checks and full preservation controls; no learned model or canonical mutation')
    reader=R/'scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py'
    result=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    (note/'actual_reader_stderr.log').write_text(result.stderr,encoding='utf8')
    assert result.returncode==0,result.stderr
    actual=json.loads(result.stdout);x=actual['all_original_and_current_actual']['same_GLM41Base_only_actual_serial_model']
    assert x['cards_returncode']==0 and not x['frozen_monitor_summary']['source_errors']
    snapshot=note/'current_complete_actual_monitor_proc_FLOCK_cards_and_closed_stages.json';put(snapshot,actual)
    procs=x['actual_proc_argv_cwd_PPid_startticks_FD_io']
    models=[p for p in procs if any(Path(a).name=='run.py' for a in p['argv'])]
    nf=[p for p in procs if any(Path(a).name.startswith('verify_') for a in p['argv'])]
    audit=Path(q['Kanana_source_meta_diagnostic']['home'])/'review.json'
    assert sha(audit)==q['Kanana_source_meta_diagnostic']['review_SHA256']
    av=js(audit);assert av['passed'] and av['source_and_meta_interface_only'] and av['no_checkpoint_values_loaded'] and av['learned_CPU_GPU_NPU_forwards']==0
    assert av['not_numeric_acceptance'] and not av['passed_full_raw_bit_control'] and not av['formal_NEW_NPU3'] and not av['formal_fullNF_or_200_or_qualification']
    assert av['physical_own_BF16_source_tensor_headers']==739 and av['all739_actual_meta_shapes_and_dtypes_match']
    closed={}
    for home in ['Kanana_own_original_real_RegStage_private_full_synthetic_control_and739_meta_only_v1_20261123','Kanana_own_original_real_RegStage_private_full_synthetic_control_and739_meta_only_v2_graphtext_20261123','Kanana_own_original_real_RegStage_cross_version_CPU_drift_diagnostic_and739_meta_only_v3_20261123']:
        p=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'/home/'actual_completion.json'
        closed[str(p)]=dict(SHA256=sha(p),complete_value=js(p))
    assert sorted(z['complete_value']['actual_returncode'] for z in closed.values())==[0,1,1]
    oldfacts=Path(ptr['current_full_facts']);assert sha(oldfacts)==ptr['current_full_facts_SHA256']
    compact=js(oldfacts)
    bee_cfg=R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json'
    assert sha(bee_cfg)==rows[str(bee_cfg)]['SHA256'];bq=js(bee_cfg);bs=js(Path(bq['wait_home'])/'state.json')
    assert bq['required_free_bytes']==bs['required_free_bytes']==192555984252
    compact.update(UTC=actual['UTC'],parent_full_facts=str(oldfacts),parent_full_facts_SHA256=sha(oldfacts),
        full_primary_source_count=len(rows),parent_pending=str(parent),parent_pending_SHA256=sha(parent),
        official_actual_readback=q['official_actual_readback'],current_model_processes=models,current_NF_processes=nf,
        current_original_EX=x['actual_original_lock_owners'],actual_free_bytes=x['actual_original_disk_free_bytes'],
        complete_monitor_summary=x['frozen_monitor_summary'],full_actual_snapshot=str(snapshot),full_actual_snapshot_SHA256=sha(snapshot),
        current_Kanana_source_meta_diagnostic=dict(review=str(audit),review_SHA256=sha(audit),complete_value=av),
        existing_BeeSFT_v5_current_actual=dict(config=str(bee_cfg),config_SHA256=sha(bee_cfg),wait_home=bq['wait_home'],state=bs,
            new_NN_job_exists=Path(bq['new_job']).exists(),new_continuation_home_exists=Path(bq['home']).exists(),not_NN_or_formal_NF=True),
        local_original_records=originals,no_new_complete200_G_or_qualification=True,no_new_model_or_learned_CPU_forward=True,
        no_deletion=True,no_second_Day23_commit=True,official_prompt_only_update_completed=False,
        original_resource_stop_earliest_UTC='2026-11-22T09:40:49.156504+00:00')
    compact['current_round_original_automation_readonly']=q['current_round_local_and_automation'];compact['cleanup']=q['fresh_cleanup_discovery'];compact['original_AA_readonly_connection_available_this_round_not_A100_job_state_claim']=True;compact['no_new_model_or_learned_CPU_GPU_forward']=True;compact['current_original_three_resources_readonly']=q['current_original_three_resources_readonly'];compact['filesystem_tiny_synthetic_extent_dedupe_not_supported']=q['filesystem_capability'];compact['old_local_control_transport_actual_no_signal']=q['old_local_transport_actual']
    compact['all_current_Kanana_private_control_and_meta_actual_completions']=closed
    facts=note/'facts.json';put(facts,compact)
    full=note/'full_actual_verified185_Kanana_graph_meta_CPU_drift_resource_v93_20261123.txt'
    text=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_FACTS_SHA256',sha(facts)).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False))
    full.write_text(text,encoding='utf8')
    pending=C/'pending_after_daily_20261123_verified185_Kanana_graph_meta_CPU_drift_resource_all_primary_v93_20261123.json'
    assert not pending.exists();put(pending,dict(UTC=actual['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),
        source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[n] for n in sorted(rows)],
        Code_Day_already_pushed='2026-11-23',Code_Day_commit='5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9',
        no_second_same_day_commit=True,next_daily_review='Day24 exact useful primary sources, consumers and dependencies, full diff/outgoing and appropriate checks, actual publisher plan_SHA then one ordinary push and remote verification'))
    upd=dict(UTC=actual['UTC'],new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),
        current_primary_source_count=len(rows),current_source_count=len(rows),latest_late_actual_note=str(full),latest_late_actual_note_SHA256=sha(full),
        latest_actual_snapshot=str(snapshot),latest_actual_snapshot_SHA256=sha(snapshot),current_completed_stage_facts=str(facts),current_completed_stage_facts_SHA256=sha(facts),
        current_full_actual_note=str(full),current_full_actual_note_SHA256=sha(full),current_full_facts=str(facts),current_full_facts_SHA256=sha(facts),
        current_actual_model_processes=models,current_actual_NF_processes=nf,current_original_EX=x['actual_original_lock_owners'],
        current_official_actual_readback=q['official_actual_readback'],current_BeeSFT_existing_waiter_actual=compact['existing_BeeSFT_v5_current_actual'],
        current_Kanana_source_meta_diagnostic=compact['current_Kanana_source_meta_diagnostic'],current_resource_stop_earliest_UTC=compact['original_resource_stop_earliest_UTC'])
    for key in ['current_model_process_active','current_full_numerical_verifier_active','current_model_NN_or_NF_active',
                'current_actual_model_NN_or_NF_active','current_NN_or_NF_active','current_formal_model_task_active','current_actual_model_process_or_NF_active']:
        upd[key]=bool(models or nf)
    assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())
    assert pointer.read_bytes()==pb and closing.read_bytes()==cb
    ptr.update(upd);close=json.loads(cb);close.update(upd);put(closing,close);put(pointer,ptr)
    print(json.dumps(dict(passed=True,UTC=actual['UTC'],pointer_SHA256=sha(pointer),closing_SHA256=sha(closing),
        source_count=len(rows),pending=str(pending),pending_SHA256=sha(pending),late_actual_note=str(full),late_actual_note_SHA256=sha(full),
        facts=str(facts),facts_SHA256=sha(facts),actual_snapshot_SHA256=sha(snapshot),actual_free_bytes=x['actual_original_disk_free_bytes'],
        models=models,full_NF=nf,EX=x['actual_original_lock_owners'],monitor_summary=x['frozen_monitor_summary'],
        official_actual_readback=q['official_actual_readback'],Kanana_source_meta_diagnostic_SHA256=sha(audit),
        no_deletion=True,no_G_or_canonical_mutation=True,no_second_Day23_commit=True),ensure_ascii=False))
if __name__=='__main__':main()

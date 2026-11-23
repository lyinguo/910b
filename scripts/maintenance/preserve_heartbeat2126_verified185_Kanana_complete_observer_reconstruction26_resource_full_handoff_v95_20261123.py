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
    assert hashlib.sha256(pb).hexdigest()=='4a60aea2bd92f45d98c285eb6254655103a83b0bfa82d83fd344d5c6c198bcb9'
    ptr=json.loads(pb);closing=Path(ptr['handoff_directory'])/'closing_addendum.json';cb=closing.read_bytes()
    assert hashlib.sha256(cb).hexdigest()=='7e654fa39e0497d9ceb5d43c2cf0a9f2f6a1cadc4394570c0962600b9489474b'
    parent=Path(ptr['new_primary_pending']);assert sha(parent)=='c1eaa9fde8b98649ecf9d516b3911ff59cefff57f059ccf14db183d44f3cdae3'
    rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows)==3048 and all(sha(p)==z['SHA256'] for p,z in rows.items())
    assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())
    assert q['official_actual_readback']['passed'] and q['official_actual_readback']['status']=='ACTIVE'
    assert q['official_actual_readback']['actual_TOML_SHA256']=='590801a94c1493e88f304f20f35bf5f31fd3a77e5548f346c735179ca0c52929'
    note=Path(ptr['handoff_directory'])/'late_actual_notes/verified185_heartbeat2126_Kanana_complete_observer_reconstruction26_resource_full_actual_v95_20261123'
    note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(pb);(note/'previous_closing_addendum.json').write_bytes(cb)
    originals=[]
    for z in q['local_original_records']:
        b=base64.b64decode(z['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==z['SHA256']
        dest=note/'local_original_records'/z['name'];assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        originals.append(dict(path=str(dest),SHA256=z['SHA256'],bytes=len(b)))
    for n,h in q['new_primary_expected_SHA256'].items():
        p=R/n;assert p.resolve().is_relative_to(R) and n.startswith(('scripts/','configs/','universal_eval/')) and sha(p)==h
        if str(p) in rows:assert rows[str(p)]['SHA256']==h
        else:
            dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),
                role='Actual Kanana complete immutable native tensor/KV observation, separate full-value reconstruction, real synthetic/meta test consumers and current readonly resource/cleanup preservation; no learned model or canonical mutation')
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
    front=Path(q['kanana_frontend_acceptance']['home']);fp=front/'review.json'
    assert sha(fp)==q['kanana_frontend_acceptance']['review_SHA256'];fv=js(fp)
    assert fv['passed'] and fv['samples']==200 and fv['actual_NEW_model_forwards']==fv['actual_NEW_frontend_computations']==0
    assert not fv['actual_NPU_inference'] and not fv['formal_fullNF_or_200_or_qualification']
    assert fv['current_C_unique_row']==379 and fv['current_three_A100_D']==dict(server='0.8900',local='0.8900',preserved_original='0.89')
    assert all(not n.startswith('/') and sha(R/n)==h for n,h in fv['source_sha256'].items())
    op=Path(q['kanana_untrained_NPU_operator']['home']);opproof=op/'review.json'
    assert sha(opproof)==q['kanana_untrained_NPU_operator']['review_SHA256'];ov=js(opproof)
    assert ov['passed'] and ov['all6_original_graph_NPU_executions_completed'] and ov['all58_untrained_parameter_source_actual_NPU_SHA_verified']
    assert ov['checkpoint_values_loaded']==ov['learned_CPU_GPU_NPU_forwards']==ov['actual_NEW_model_forwards']==0
    assert not ov['formal_NEW_NPU3'] and not ov['formal_fullNF_or_200_or_qualification']
    assert len(ov['controls'])==6 and len(ov['synthetic_parameter_sources'])==58
    current_closed={}
    for home in ['Kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123','Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123','Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v3_20261123','Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v4_exact_pipeline_20261123','Kanana_original_parameterized_RegStage_actual_NPU_synthetic_interface_diagnostic_v4_20261123']:
        p=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'/home/'actual_completion.json'
        current_closed[str(p)]=dict(SHA256=sha(p),complete_value=js(p))
    assert [z['complete_value']['actual_returncode'] for z in current_closed.values()]==[1,0,1,0,0]
    new_contracts={}
    for key,count in [('kanana_complete_observer_tests',14),('kanana_independent_reconstruction_tests',12)]:
        home=Path(q[key]['home']);rp=home/'review.json';assert sha(rp)==q[key]['review_SHA256'];rv=js(rp)
        assert rv['passed'] and rv['check_count']==count and len(rv['checks'])==count
        assert rv['learned_CPU_GPU_NPU_forwards']==0 and not rv['formal_NEW_NPU3'] and not rv['formal_fullNF_or_200_or_qualification']
        assert all(not n.startswith('/') and sha(R/n)==h for n,h in rv['source_sha256'].items())
        cp=home/'actual_completion.json';assert js(cp)['actual_returncode']==0 and js(cp)['no_model_NN_or_fullNF']
        new_contracts[key]=dict(review=str(rp),review_SHA256=sha(rp),complete_value=rv,actual_completion=str(cp),actual_completion_SHA256=sha(cp),actual_completion_value=js(cp))
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
    compact['current_Kanana_completed_frontend200_input_only_acceptance']=dict(review=str(fp),review_SHA256=sha(fp),complete_value=fv)
    compact['current_Kanana_untrained_NPU_operator_interface_only']=dict(review=str(opproof),review_SHA256=sha(opproof),complete_value=ov)
    compact['current_Kanana_frontend_source_scope_operator_actual_completions']=current_closed
    compact['old_local_control_transport_naturally_closed255_remote_actual1']=q['old_local_transport_actual']
    compact['no_new_model_NPU3_fullNF_or200_despite_input200_and_untrained_NPU_operators']=True
    compact['current_Kanana_complete_native_observer_and_independent_reconstruction_contract_tests']=new_contracts
    compact['this_round_14_plus12_contract_checks_not_model_NPU3_or_fullNF_or_score']=True
    compact['transient_permission_and_local_sandbox_errors_exact_recovery']=q['transient_errors_and_recovery']
    facts=note/'facts.json';put(facts,compact)
    full=note/'full_actual_verified185_Kanana_complete_observer_reconstruction26_resource_v95_20261123.txt'
    text=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_FACTS_SHA256',sha(facts)).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False))
    full.write_text(text,encoding='utf8')
    pending=C/'pending_after_daily_20261123_verified185_Kanana_complete_observer_reconstruction26_resource_all_primary_v95_20261123.json'
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
    upd['current_Kanana_completed_frontend200_input_only_acceptance']=compact['current_Kanana_completed_frontend200_input_only_acceptance']
    upd['current_Kanana_untrained_NPU_operator_interface_only']=compact['current_Kanana_untrained_NPU_operator_interface_only']
    upd['current_owned_candidate']='kakaocorp/kanana-1.5-v-3b-instruct'
    upd['current_owned_candidate_stage']='Own input200 and six untrained NPU RegStage controls previously complete; new native full-value observer 14 tests and independent reconstruction 12 tests passed, but model arithmetic verifier/NPU loader/YAML/NEW NPU3 still incomplete'
    upd['old_local_control_transport_naturally_closed255_remote_actual1']=q['old_local_transport_actual']
    upd['current_Kanana_complete_native_observer_and_independent_reconstruction_contract_tests']=new_contracts
    upd['transient_permission_and_local_sandbox_errors_exact_recovery']=q['transient_errors_and_recovery']
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

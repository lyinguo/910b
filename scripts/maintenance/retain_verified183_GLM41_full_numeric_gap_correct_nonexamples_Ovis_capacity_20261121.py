"""Prune an exact accepted-GLM correct-only NPZ list; retain all other evidence.

No model, score, report, source, configuration, or log is changed. The user's
existing cleanup authorization applies only after an independently reviewed
plan, foreign-reference checks and complete protected-file SHA comparisons.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import fcntl, hashlib, json, os, re, shutil, stat, traceback

R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
S = R / 'workspace/diagnostics/model_runs'
H = B / 'verified183_GLM41_full_numeric_gap_correct_nonexample_minimal_Ovis_capacity_recovery_20261121'
CFG = R / 'configs/retention/verified183_GLM41_full_numeric_gap_correct_nonexamples_Ovis_capacity_20261121.json'
LED = D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
MID = 'zai-org/GLM-4.1V-9B-Thinking'
sys.path[:0] = [str(R), str(R / 'scripts')]
from maintenance import retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119 as base
from maintenance.audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119 import preserve_and_remove_indices, full_head
sha, js, put = base.sha, base.js, base.put
now = lambda: datetime.now(timezone.utc).isoformat()
FIVE = {'eval_report.json', 'sample_inputs.jsonl', 'sample_outputs.jsonl', 'run.log', 'operator_lineage.json'}

def quiet(owned=False):
    from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
    from eager_corrective_control_safety_2203 import core_unchanged
    core_unchanged()
    others = [p for p in project_processes() if int(p['PID']) != os.getpid()]
    allowed='scripts/maintenance/watch_batch_910b_immediate_disk_recovery_cleanup_v1_20261120.py'
    watcher=[p for p in others if p.get('cwd')==str(R) and str(R/allowed) in p.get('argv',[])]
    assert not watcher or sha(R/allowed)=='32bd7209129b3cf22bac0056584847262e9793b3fab6b52dfaccc7b3de801401'
    others=[p for p in others if p not in watcher]
    owners = runner_lock_owners()
    assert not others, ('Unknown or active project process', others)
    assert (not owners if not owned else owners and all(int(p['PID']) == os.getpid() for p in owners)), owners

def exact_file(row):
    home, p = Path(row['home']), Path(row['path'])
    assert home.resolve() == home and home.is_relative_to(S) and home.name == 'GLM41Thinking_own_original_BF16_8192_20261120_144537_053387_1624585'
    assert p.parent == home and p.resolve() == p and p.name == 'sample_%05d.npz' % row['index']
    st = p.lstat()
    assert stat.S_ISREG(st.st_mode) and not p.is_symlink() and st.st_nlink == 1
    assert st.st_size == row['bytes'] and sha(p) == row['SHA256']

def plan():
    quiet(); cfg=js(CFG);assert not (H/'plan.json').exists()
    assert sha(LED)==cfg['ledger_before_SHA256']=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3'
    assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
    for n,h in cfg['canonical_SHA256'].items():assert sha(n)==h
    state,reg=js(D/'state.json'),js(D/'registry.json')
    assert sum(m['status']=='done' for m in state['models'].values())==183 and sum(m['status']=='needs_adaptation' for m in state['models'].values())==207
    assert state['models'][MID]['status']=='needs_adaptation' and state['models'][MID]['score'] is None and MID not in reg
    audit_path=Path(cfg['current183_full_target_audit']);assert sha(audit_path)==cfg['current183_full_target_audit_SHA256']=='2fe185cc1b6926e8e45adca502e4793168bea2261112b60f00738c5ed6150114'
    baseline_audit=js(audit_path);assert baseline_audit['passed'] and baseline_audit['qualified_delivered']==69 and baseline_audit['numeric_within5pct']==168
    assert baseline_audit['local_and_server_book_SHA256']==cfg['current_book_SHA256']=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
    J=Path(cfg['measurement_job']);assert sha(J/'job.json')=='756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163'
    job=js(J/'job.json');done=js(J/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0 and done['model_returncode']==1 and [a['sample_count'] for a in done['accepted']]==[3,200]
    from A100_alignment_evidence_runner_20261103 import unchanged
    import batch_910b as batch
    from batch_910b_support import validate_report,workbook_rows
    unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec'])
    approval=js(job['spec']['alignment_approval_review']);assert sha(job['spec']['alignment_approval_review'])==job['spec']['alignment_approval_review_sha256'] and approval['approved'] and approval['real_NPU3_independent_proof_verified'] and approval['numerical_review_verified']
    proof_path=Path(done['accepted'][-1]['proof']);assert sha(proof_path)==done['accepted'][-1]['proof_sha256']=='af10ad5191779c0732038c95b369fa6cef27ec2478e813eafa59f7c87c97c101'
    pf=js(proof_path);assert pf['passed'] and pf['actual_NPU_inference'] and not pf['historical_record_reuse'] and pf['sample_count']==200 and pf['actual_forward_count']==55771 and pf['metrics']=={'Accuracy':.555}
    assert pf['source_tensor_count']==704 and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed'] and full_head(pf) and not pf['error_indices']
    assert not pf['formal_alignment_within_5pct'] and not pf['original_protocol_final_qualification_complete'] and set(pf['relative_error_to_current_A100_D'].values())=={'0.26'}
    folder=Path(pf['report_dir']);assert set(p.name for p in folder.iterdir())==FIVE
    assert validate_report(R,folder,MID,'image-text-to-text',200,pf)==.555
    for n,h in pf['report_sha256'].items():assert sha(folder/n)==h
    cache={};checked=[]
    def checkmap(label,m):
        for n,h in m.items():
            p=Path(n) if Path(n).is_absolute() else R/n
            if str(p) not in cache:cache[str(p)]=sha(p)
            if cache[str(p)]!=h:
                assert p==LED and h=='06f870990721bdb4e58a409f842980b7b4578eddc23367263d041c052f8e43ce' and cache[str(p)]==cfg['ledger_before_SHA256'],(label,n,'Unreviewed external change')
                old=B/'verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120'
                assert sha(old/'retention_actual_completion.json')=='a81e1801bf7fe73297eafe8865589cd30e635baddc0e25910c34d985c274cc01' and js(old/'retention_actual_completion.json')['all_protected_SHA256_unchanged']
                assert sha(old/'ledger_before.original')==h and js(LED)['events'][-1]['receipt_SHA256']==sha(old/'retention_actual_completion.json')
        checked.append(dict(label=label,entries=len(m)))
    assert sha(approval['true3_proof'])==approval['true3_proof_SHA256']
    for label,p in [('original_true3',js(approval['true3_proof']))]+[(str(a['sample_count']),js(a['proof'])) for a in done['accepted']]:
        assert p['passed'] and p['actual_NPU_inference'] and not p['historical_record_reuse'];validate_report(R,p['report_dir'],MID,'image-text-to-text',p['sample_count'],p)
        checkmap(label,p['external_evidence_sha256'])
    checkmap('source',job['spec']['source_sha256'])
    rows,_,_=workbook_rows(Path(cfg['workbook']));byid={r['model_id']:r for r in rows}
    assert len(rows)==700 and sum(r['score'] is not None for r in rows)==183 and byid[MID]['row']==696 and byid[MID]['task']=='image-text-to-text' and byid[MID]['score'] is None
    old=js(Path(cfg['export183']));assert sha(cfg['export183'])=='59400a4f9eeacd2746088fab2952957f5b1802ef38ee9b415aac6e2f6a49e5a0'
    for r in old['records']:
        a=byid[r['model_id']];assert a['row']==r['current_row'] and a['task']==r['task'] and str(a['score'])==r['current_G'] or (a['row']==r['current_row'] and a['task']==r['task'] and __import__('decimal').Decimal(str(a['score']))==__import__('decimal').Decimal(r['current_G']))
    stats=js(cfg['stat_baseline183']);assert sha(cfg['stat_baseline183'])=='218c75abd8d84caced7308f3a521504eb6b76682eb2a655d8a068ee63ec59fb7'
    for m,files in stats['reports'].items():
        for n,oldstat in files.items():
            st=(Path(state['models'][m]['report_dir'])/n).stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==oldstat,(m,n,'Prior complete report changed')
    report=js(folder/'eval_report.json');details=report['detailed_results']
    special=set().union(*(set(pf[k]) for k in ('error_indices','empty_indices','invalid_indices','truncated_indices')))
    sel=preserve_and_remove_indices(details,report['pipeline_config']['algorithm']['batch_size'],special)
    assert len(sel['correct'])==111 and len(sel['incorrect'])==89 and set(sel['incorrect'])<=set(sel['keep'])
    assert set(range(10))|set(range(90,110))|set(range(190,200))<=set(sel['keep'])
    line=js(folder/'operator_lineage.json')['actual_execution'];home=Path(line['evidence_root']).resolve()
    trace_path=Path(line['sample_trace']);trace=[json.loads(s) for s in trace_path.read_text().splitlines()]
    assert [r['index'] for r in trace]==list(range(200)) and sum(r['actual_native_forwards'] for r in trace)==55771
    candidates={};all_meta=[]
    for row in trace:
        ep=Path(row['actual_execution_file']);assert sha(ep)==row['actual_execution_SHA256'];e=js(ep);i=row['index'];p=Path(e['archive']).resolve()
        assert p.parent==home and not e['error'] and pf['external_evidence_sha256'][str(p)]==e['archive_SHA256']
        st=p.stat();v=dict(index=i,logical_indices=[i],path=str(p),home=str(home),model_id=MID,bytes=st.st_size,allocated_bytes=st.st_blocks*512,SHA256=e['archive_SHA256'],proof=str(proof_path),proof_SHA256=sha(proof_path));all_meta.append(v)
        if i in sel['remove'] and st.st_nlink==1:exact_file(v);candidates[str(p)]=v
    helpers=base.old_helpers(cfg);accepted={}
    for m,sp in reg.items():
        if isinstance(sp,dict) and sp.get('accepted_formal200_proof'):accepted[m]=dict(proof=sp['accepted_formal200_proof'],report_dir=sp['accepted_formal200_report'])
    accepted[MID]=dict(proof=str(proof_path),report_dir=str(folder))
    foreign,metadata=base.sources_and_dependencies(reg,accepted,candidates,helpers)
    for p in foreign:candidates.pop(p,None)
    target=max(0,cfg['required_free_bytes']+512*1024**2-shutil.disk_usage(R).free);chosen={};reclaim=0
    for p,row in sorted(candidates.items(),key=lambda kv:-kv[1]['allocated_bytes']):
        if reclaim>=target:break
        chosen[p]=row;reclaim+=row['allocated_bytes']
    candidates=chosen;assert candidates and len(candidates)<=100 and sum(v['bytes'] for v in candidates.values())<=20*1024**3
    assert reclaim>=target,'Eligible correct nonexample values insufficient; no deletion permitted'
    protected=set(metadata)|{Path(n) for n in cfg['canonical_SHA256']}|{CFG,proof_path,audit_path,trace_path,Path(cfg['export183']),Path(cfg['stat_baseline183'])}
    for m,row in state['models'].items():
        if row['status']=='done':protected.update(Path(row['report_dir'])/n for n in FIVE)
    protected.update(p for p in home.rglob('*') if p.is_file() and str(p) not in candidates)
    protected.update(p for p in folder.iterdir());protected.update(p for p in J.rglob('*') if p.is_file())
    for a in [js(approval['true3_proof']),js(done['accepted'][0]['proof'])]:
        protected.update(Path(n) for n in a['external_evidence_sha256'] if Path(n).is_relative_to(S) and Path(n).is_file())
        protected.update(Path(a['report_dir'])/n for n in FIVE)
    failed=S/'MetaMllama_own_original_SDPA2048_20261120_195030_614654_1637048'
    protected.update(p for p in failed.rglob('*') if p.is_file())
    for root in (R/'scripts',R/'configs',R/'universal_eval'):protected.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml','.mjs','.ir'))
    assert not set(candidates)&{str(p) for p in protected}
    hashes={}
    for i,p in enumerate(sorted(protected)):
        hashes[str(p)]=sha(p)
        if i%200==0:put(H/'progress.json',dict(UTC=now(),phase='hash_all_protected_reports_sources_logs_failed_values',checked=i+1,total=len(protected)))
    quiet();(H/'ledger_before.original').write_bytes(LED.read_bytes())
    audit=dict(passed=True,UTC=now(),full_current183_target_audit=cfg['current183_full_target_audit'],all183_C_B_D_G_and_complete_five_stats_preserved=True,all_original3_fresh3_formal200_full_source_and_external_checks=checked,distinct_SHA_files=len(cache),measurement_job=str(J),formal_proof=str(proof_path),formal_proof_SHA256=sha(proof_path),sample_count=200,score=.555,three_D_relative_error=pf['relative_error_to_current_A100_D'],full_numeric_acceptance=True,original_protocol_qualified=False,G_publication_permitted=False,all89_wrong_76invalid_and_original_outer10_full_examples_retained=True,no_deletion_yet=True)
    put(H/'measurement_target_audit.json',audit)
    p=dict(UTC=now(),config_SHA256=sha(CFG),ledger_before_SHA256=sha(LED),canonical_SHA256=cfg['canonical_SHA256'],proof_SHA256=sha(proof_path),formal_proof=str(proof_path),selection=sel,all200_metadata=all_meta,remove=list(candidates.values()),remove_count=len(candidates),remove_bytes=sum(v['bytes'] for v in candidates.values()),remove_allocated_bytes=reclaim,foreign_references_preserved=foreign,protected_SHA256=hashes,protected_count=len(hashes),free_before_bytes=shutil.disk_usage(R).free,full_measurement_target_audit=str(H/'measurement_target_audit.json'),full_measurement_target_audit_SHA256=sha(H/'measurement_target_audit.json'),all_wrong_full_groups_small_failed_unverified_reports_sources_configs_logs_retained=True)
    put(H/'plan.json',p);return dict(plan_SHA256=sha(H/'plan.json'),remove_count=p['remove_count'],remove_bytes=p['remove_bytes'],projected_free_bytes=p['free_before_bytes']+reclaim,protected_count=p['protected_count'],foreign_files_preserved=len(foreign),wrong_retained=89,numeric_example_count=200-len(candidates),full_measurement_target_audit_SHA256=sha(H/'measurement_target_audit.json'),no_deletion_yet=True)


def apply(reviewed_plan):
    from eager_corrective_control_safety_2203 import kernel_lock_owned
    quiet(); cfg=js(CFG); assert sha(H/'plan.json') == reviewed_plan
    p=js(H/'plan.json'); assert sha(CFG)==p['config_SHA256'] and sha(LED)==p['ledger_before_SHA256']
    assert not (H/'actual_completion.json').exists()
    with (D/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');quiet(True)
        for i,(n,h) in enumerate(p['protected_SHA256'].items()):
            assert sha(n)==h,('Protected changed before deletion',n)
            if i%200==0:put(H/'progress.json',dict(UTC=now(),phase='fresh_protected_before_delete',checked=i+1,total=p['protected_count']))
        for row in p['remove']:
            assert row['index'] not in p['selection']['keep'] and row['index'] not in p['selection']['incorrect']
            exact_file(row); assert sha(row['proof'])==row['proof_SHA256']
        helpers=base.old_helpers(cfg)
        state,reg=js(D/'state.json'),js(D/'registry.json')
        accepted={m:dict(proof=s['accepted_formal200_proof'],report_dir=s['accepted_formal200_report']) for m,s in reg.items() if isinstance(s,dict) and s.get('accepted_formal200_proof')}
        foreign,_=base.sources_and_dependencies(reg,accepted,{r['path']:r for r in p['remove']},helpers)
        assert not foreign,('New foreign reference appeared',foreign)
        assert all(sha(n)==h for n,h in p['canonical_SHA256'].items());quiet(True)
        manifest=H/'exact_removed_correct_nonexample_NPZ_SHA256.jsonl'; assert not manifest.exists()
        removed=[];failure=None;free=shutil.disk_usage(R).free
        try:
            with manifest.open('x',encoding='utf-8') as out:
                for row in p['remove']:
                    exact_file(row);Path(row['path']).unlink();removed.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
        except BaseException: failure=traceback.format_exc()
        bad=[n for n,h in p['protected_SHA256'].items() if sha(n)!=h]
        ok=not failure and not bad and len(removed)==p['remove_count']
        actual=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=reviewed_plan,removed_files=len(removed),
          removed_bytes=sum(r['bytes'] for r in removed),removed_allocated_bytes=sum(r['allocated_bytes'] for r in removed),
          manifest=str(manifest),manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not bad,
          protected_errors=bad,error=failure,free_before_bytes=free,free_after_bytes=shutil.disk_usage(R).free,
          all89_wrong_and_full_original_outer_groups_retained=True,all_reports_source_config_logs_small_failure_unverified_preserved=True,
          canonical_scores_both_workbooks_state_registry_binding_unchanged=True,exact_pruned_arrays_no_longer_available_original_proof_bytes_retained=True)
        put(H/'actual_completion.json',actual);assert sha(LED)==p['ledger_before_SHA256'];ledger=js(LED)
        ledger['events'].append(dict(UTC=actual['UTC'],authorization=cfg['user_authorization'],receipt=str(H/'actual_completion.json'),
          receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=len(removed),
          removed_bytes=actual['removed_bytes'],all_original_records_kept=True,all_incorrect_cases_preserved=True,
          intentional_nonexample_matrix_unavailability=True,actual_returncode=actual['actual_returncode']))
        put(LED,ledger);assert all(sha(n)==h for n,h in p['canonical_SHA256'].items() if n!=str(LED));assert ok
        return dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED))

if __name__ == '__main__':
    print(json.dumps(plan() if sys.argv[1]=='plan' else apply(sys.argv[2]),ensure_ascii=False))

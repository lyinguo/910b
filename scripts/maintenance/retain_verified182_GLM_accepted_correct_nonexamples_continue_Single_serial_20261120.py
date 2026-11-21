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
H = B / 'verified182_GLM_accepted_correct_nonexamples_user_serial_Single_capacity_recovery_20261120'
CFG = R / 'configs/retention/verified182_GLM_accepted_correct_nonexamples_continue_Single_serial_20261120.json'
LED = D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
MID = 'zai-org/GLM-4.6V-Flash'
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
    owners = runner_lock_owners()
    assert not others, ('Unknown or active project process', others)
    assert (not owners if not owned else owners and all(int(p['PID']) == os.getpid() for p in owners)), owners

def exact_file(row):
    home, p = Path(row['home']), Path(row['path'])
    assert home.resolve() == home and home.is_relative_to(S) and home.name.startswith('GLM46Flash_own_BF16_thinking_20261119_180502_')
    assert p.parent == home and p.resolve() == p and p.name == 'sample_%05d.npz' % row['index']
    st = p.lstat()
    assert stat.S_ISREG(st.st_mode) and not p.is_symlink() and st.st_nlink == 1
    assert st.st_size == row['bytes'] and sha(p) == row['SHA256']

def plan():
    quiet(); cfg = js(CFG)
    assert not (H / 'plan.json').exists()
    assert sha(LED) == cfg['ledger_before_SHA256'] == '7978cbde39e32b7d28ecb05cf69148e451cbe262493b39d6b07e6fdb785b9b71'
    assert sha(D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json') == '66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
    state, reg = js(D / 'state.json'), js(D / 'registry.json')
    assert sum(m['status'] == 'done' for m in state['models'].values()) == 182
    assert sum(m['status'] == 'needs_adaptation' for m in state['models'].values()) == 208
    assert state['models'][MID]['original_protocol_final_qualification_complete'] is True
    for n, h in cfg['canonical_SHA256'].items(): assert sha(n) == h, n
    audit_path = Path(cfg['current182_full_target_audit'])
    assert sha(audit_path) == cfg['current182_full_target_audit_SHA256'] == '68b43b7c08f7095e6086a26b1813752572d4f1320e06121df5329a28d3aca9e2'
    audit = js(audit_path)
    assert audit['passed'] and audit['qualified_delivered'] == 68 and audit['numeric_within5pct'] == 167
    assert audit['local_and_server_book_SHA256'] == cfg['current_book_SHA256'] == '0b2fcf21fe5a56e5b33111c4e65c2ec0ddf53dae43d451dec756a342df810340'
    spec = reg[MID]; proof_path = Path(spec['accepted_formal200_proof'])
    assert sha(proof_path) == spec['accepted_formal200_proof_sha256'] == 'cf8d72c9296330ed55e520356c82a8225f0448724c11f405cb640e0858fe62a0'
    pf = js(proof_path)
    assert pf['passed'] and pf['actual_NPU_inference'] and pf['NEW_real_NPU_inference'] and not pf['historical_record_reuse']
    assert pf['sample_count'] == 200 and pf['actual_forward_count'] == 44636 and pf['metrics'] == {'Accuracy': .95}
    assert pf['original_protocol_final_qualification_complete'] and pf['formal_alignment_within_5pct'] and full_head(pf)
    assert not pf['error_indices'] and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed']
    folder = Path(state['models'][MID]['report_dir']); assert set(p.name for p in folder.iterdir()) == FIVE
    assert str(folder) == spec['accepted_formal200_report'] == pf['report_dir']
    for name, digest in pf['report_sha256'].items(): assert sha(folder / name) == digest
    report = js(folder / 'eval_report.json'); details = report['detailed_results']
    special = set().union(*(set(pf[k]) for k in ('error_indices', 'empty_indices', 'invalid_indices', 'truncated_indices')))
    sel = preserve_and_remove_indices(details, report['pipeline_config']['algorithm']['batch_size'], special)
    assert len(sel['incorrect']) == 10 and set(sel['incorrect']) <= set(sel['keep'])
    assert set(range(10)) | set(range(90, 110)) | set(range(190, 200)) <= set(sel['keep'])
    line = js(folder / 'operator_lineage.json')['actual_execution']; home = Path(line['evidence_root']).resolve()
    trace_path = Path(line['sample_trace']); trace = [json.loads(s) for s in trace_path.read_text().splitlines()]
    assert [e['index'] for e in trace] == list(range(200)) and sum(e['actual_native_forwards'] for e in trace) == 44636
    candidates = {}
    all_meta = []
    for e in trace:
        path = Path(e['actual_execution_file']); assert sha(path) == e['actual_execution_SHA256']
        v = js(path); i = e['index']; p = Path(v['archive']).resolve(); assert p.parent == home and not v['error']
        assert pf['external_evidence_sha256'][str(p)] == v['archive_SHA256']
        st = p.stat(); row = dict(index=i, logical_indices=[i], path=str(p), home=str(home), model_id=MID, bytes=st.st_size,
            allocated_bytes=st.st_blocks*512, SHA256=v['archive_SHA256'], proof=str(proof_path), proof_SHA256=sha(proof_path))
        all_meta.append(row)
        if i in sel['remove']: exact_file(row); candidates[str(p)] = row
    helpers = base.old_helpers(cfg)
    accepted = {}
    for mid, row in state['models'].items():
        if row['status'] != 'done': continue
        sp = reg.get(mid, {}); proof = sp.get('accepted_formal200_proof')
        if not proof:
            arr = [a for a in row.get('attempts', []) if a.get('count') == 200 and a.get('independent_verification')]
            proof = arr[-1]['independent_verification'] if arr else None
        if proof and Path(proof).is_file(): accepted[mid] = dict(proof=proof, report_dir=row['report_dir'])
    assert MID in accepted and 'zai-org/AutoGLM-Phone-9B-Multilingual' in accepted
    foreign, metadata = base.sources_and_dependencies(reg, accepted, candidates, helpers)
    for n in foreign: candidates.pop(n, None)
    assert candidates and len(candidates) <= 160 and sum(v['bytes'] for v in candidates.values()) <= 80*1024**3
    protected = set(metadata) | {Path(n) for n in cfg['canonical_SHA256']} | {CFG, proof_path, audit_path, trace_path}
    for mid, row in state['models'].items():
        if row['status'] == 'done': protected.update(Path(row['report_dir']) / n for n in FIVE)
    protected.update(p for p in home.rglob('*') if p.is_file() and str(p) not in candidates)
    protected.update(p for p in folder.iterdir())
    job_home = proof_path.parent.parent
    protected.update(p for p in job_home.rglob('*') if p.is_file())
    own3 = B / 'AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v2_exact_frontend_path_real_FLOCK_20261120/fresh3/verification_stdout.json'
    assert sha(own3) == 'da99ef8ddab138186eb4368204a6917041f126d3fb03730fe8163352a211c259'
    protected.add(own3)
    protected.update(Path(n) for n in js(own3)['external_evidence_sha256'] if Path(n).is_relative_to(S) and Path(n).is_file())
    for root in (R/'scripts', R/'configs', R/'universal_eval'):
        protected.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml','.mjs','.ir'))
    assert not set(candidates) & {str(p) for p in protected}
    hashes = {}
    for i, p in enumerate(sorted(protected)):
        hashes[str(p)] = sha(p)
        if i % 200 == 0: put(H/'progress.json', dict(UTC=now(), phase='hash_all_protected', checked=i+1, total=len(protected)))
    quiet(); (H/'ledger_before.original').write_bytes(LED.read_bytes())
    p = dict(UTC=now(),config_SHA256=sha(CFG),ledger_before_SHA256=sha(LED),canonical_SHA256=cfg['canonical_SHA256'],
             proof_SHA256=sha(proof_path),formal_proof=str(proof_path),selection=sel,all200_metadata=all_meta,
             remove=list(candidates.values()),remove_count=len(candidates),remove_bytes=sum(v['bytes'] for v in candidates.values()),
             remove_allocated_bytes=sum(v['allocated_bytes'] for v in candidates.values()),foreign_references_preserved=foreign,
             protected_SHA256=hashes,protected_count=len(hashes),free_before_bytes=shutil.disk_usage(R).free,
             all_wrong_full_groups_small_failed_unverified_reports_sources_configs_logs_retained=True)
    put(H/'plan.json',p)
    return dict(plan_SHA256=sha(H/'plan.json'),remove_count=p['remove_count'],remove_bytes=p['remove_bytes'],
        projected_free_bytes=p['free_before_bytes']+p['remove_allocated_bytes'],protected_count=p['protected_count'],
        foreign_files_preserved=len(foreign),wrong_retained=10,numeric_example_count=200-len(candidates),no_deletion_yet=True)

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
          all10_wrong_and_full_original_outer_groups_retained=True,all_reports_source_config_logs_small_failure_unverified_preserved=True,
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

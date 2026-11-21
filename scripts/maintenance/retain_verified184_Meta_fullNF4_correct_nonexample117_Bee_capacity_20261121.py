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
H = B / 'verified184_Meta_fullNF4_correct_nonexample117_user_Bee_capacity_recovery_20261121'
CFG = R / 'configs/retention/verified184_Meta_fullNF4_correct_nonexample117_user_Bee_capacity_20261121.json'
LED = D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
MID = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
sys.path[:0] = [str(R), str(R / 'scripts')]
from maintenance import retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119 as base
from maintenance.audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119 import preserve_and_remove_indices, full_head
sha, js, put = (base.sha, base.js, base.put)
now = lambda: datetime.now(timezone.utc).isoformat()
FIVE = {'eval_report.json', 'sample_inputs.jsonl', 'sample_outputs.jsonl', 'run.log', 'operator_lineage.json'}

def quiet(owned=False):
    from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
    from eager_corrective_control_safety_2203 import core_unchanged
    core_unchanged()
    others = [p for p in project_processes() if int(p['PID']) != os.getpid()]
    allowed = []
    for p in others:
        if p.get('cwd') != str(R):
            continue
        watcher = R / 'scripts/maintenance/watch_batch_910b_immediate_disk_recovery_cleanup_v1_20261120.py'
        bad = R / 'scripts/continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_20261121.py'
        if str(watcher) in p.get('argv', []):
            assert sha(watcher) == '32bd7209129b3cf22bac0056584847262e9793b3fab6b52dfaccc7b3de801401'
            allowed.append(p)
        elif str(bad) in p.get('argv', []):
            assert sha(bad) == '16241ed4e429572f4d1fedc49d31bb89f79e9603c1ee357507a51739e2970678'
            cfg = js(R / 'configs/queues/BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_20261121.json')
            assert not Path(cfg['previous_dual_publication_home']).is_absolute() and (not Path(cfg['previous_dual_publication_home']).exists())
            allowed.append(p)
    others = [p for p in others if p not in allowed]
    assert not others, ('Unknown or active project process', others)
    owners = runner_lock_owners()
    assert not owners if not owned else owners and all((int(p['PID']) == os.getpid() for p in owners))

def exact_file(row):
    home, p = (Path(row['home']), Path(row['path']))
    assert home.resolve() == home and home.is_relative_to(S) and (home.name == 'MetaMllama_own_original_SDPA2048_20261120_195030_614654_1637048')
    assert p.parent == home and p.resolve() == p and (p.name == 'sample_%05d.npz' % row['index'])
    st = p.lstat()
    assert stat.S_ISREG(st.st_mode) and (not p.is_symlink()) and (st.st_nlink == 1)
    assert st.st_size == row['bytes'] and sha(p) == row['SHA256']

def plan():
    quiet()
    cfg = js(CFG)
    assert not (H / 'plan.json').exists()
    assert sha(LED) == cfg['ledger_before_SHA256'] == '6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c'
    assert sha(D / 'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json') == '66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
    for n, h in cfg['canonical_SHA256'].items():
        assert sha(n) == h
    state, reg = (js(D / 'state.json'), js(D / 'registry.json'))
    assert sum((v['status'] == 'done' for v in state['models'].values())) == 184 and sum((v['status'] == 'needs_adaptation' for v in state['models'].values())) == 206
    assert state['models'][MID]['status'] == 'needs_adaptation' and state['models'][MID]['score'] is None and (MID not in reg)
    audit_path = Path(cfg['current184_full_target_audit'])
    assert sha(audit_path) == '033e43ca89d3f6ba87a8924490c98b696e8a96565014dd1d498e2dd8cc3a06a2'
    pub = js(audit_path)
    assert pub['passed'] and pub['qualified_delivered'] == 70 and (pub['numeric_within5pct'] == 169)
    current = pub['canonical_SHA256']
    previous = js(audit_path.parent / 'publication_receipt.json')['canonical_before_sha256']
    M = B / 'MetaMllama_completed_genuine200_formal_NF4_original_empty_invalid_schema_v2_actual_file_SHA_after_Ovis_serial_no_NN_replay_20261121'
    assert sha(M / 'formal_acceptance.json') == 'adb80b13cd349a23932cc18c608d342b6225ff51a8aed6fa0f3d5fde74179e3b' and js(M / 'actual_completion.json')['actual_returncode'] == 0
    accept = js(M / 'formal_acceptance.json')
    assert accept['passed'] and accept['actual_NEW_model_forwards'] == 0 and accept['original_job_remains_failed'] and (accept['score'] == 0.705) and (not accept['original_protocol_qualified'])
    J = Path(accept['original_genuine_NPU200_job'])
    assert sha(J / 'job.json') == '581c7c7c9a4f6a270f5c9021ee9c6824144dc7c437473b7eb3ab81fce6141757'
    job = js(J / 'job.json')
    done = js(J / 'completion.json')
    assert done['actual_returncode'] == done['verifier_returncode'] == 1 and done['model_returncode'] == 1
    proof_path = Path(accept['proof'])
    assert sha(proof_path) == accept['proof_SHA256'] == 'c54902717b16dd42b1b02f851c02652ea77d9ad31884eae238d38dcc3c08cb13'
    pf = js(proof_path)
    assert pf['passed'] and pf['sample_count'] == 200 and (pf['actual_forward_count'] == 2823) and (pf['source_tensor_count'] == 906) and (pf['metrics'] == {'Accuracy': 0.705}) and (not pf['error_indices'])
    assert pf['all906_own_source_cast_actual_NPU_SHA_verified'] and pf['all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward'] and (pf['all_full128256_native_greedy_argmax_replayed'] == 2823)
    assert pf['invalid_indices'] == [14, 95, 97, 100, 105, 153] and (not pf['formal_alignment_within_5pct']) and (not pf['original_protocol_final_qualification_complete'])
    from batch_910b_support import validate_report
    folder = Path(pf['report_dir'])
    assert {p.name for p in folder.iterdir()} == FIVE and validate_report(R, folder, MID, 'image-text-to-text', 200, pf) == 0.705
    cache = {}
    checked = []

    def checkmap(label, m):
        for n, h in m.items():
            p = Path(n) if Path(n).is_absolute() else R / n
            if str(p) not in cache:
                cache[str(p)] = sha(p)
            actual = cache[str(p)]
            if actual != h:
                if str(p) in previous:
                    assert previous[str(p)] == h and current[str(p)] == actual
                else:
                    ret = B / 'verified183_GLM41_full_numeric_gap_correct_nonexample_minimal_Ovis_capacity_recovery_20261121'
                    assert p == LED and h == '4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3' and (actual == cfg['ledger_before_SHA256']) and (sha(ret / 'actual_completion.json') == '15e6800acd2a727af9465787cbb61e76e3257e3e01b2bee886f430a9033f5e5c') and (sha(ret / 'ledger_before.original') == h)
        checked.append(dict(label=label, entries=len(m)))
    for n, h in pf['report_sha256'].items():
        assert sha(folder / n) == h
    checkmap('current Meta fullNF4 complete external', pf['external_evidence_sha256'])
    checkmap('current Meta original model sources', job['spec']['source_sha256'])
    checkmap('actual accepted Meta independent sources', pf['source_sha256'])
    approval = js(job['spec']['alignment_approval_review'])
    assert sha(job['spec']['alignment_approval_review']) == job['spec']['alignment_approval_review_sha256'] and approval['approved']
    controls = [js(approval['true3_proof'])] + [js(x['proof']) for x in done['accepted']]
    for i, p in enumerate(controls):
        assert p['passed'] and p['sample_count'] == 3 and (not p['historical_record_reuse'])
        checkmap('original3/fresh3 ' + str(i), p['external_evidence_sha256'])
    discovery = js(cfg['discovery'])
    assert sha(cfg['discovery']) == cfg['discovery_SHA256'] == '2cf8a74bdee79d3234c459d8da192f5655a3a168361994e85be355dc2406b57d'
    sc, = [x for x in discovery['scopes'] if x['model_id'] == MID]
    sel = sc['all_wrong_invalid_empty_truncated_and_whole_groups_preserved']
    assert len(sel['incorrect']) == 59 and len(sc['remove']) == 117 and (set(sel['incorrect']) <= set(sel['keep'])) and (set(range(10)) | set(range(90, 110)) | set(range(190, 200)) <= set(sel['keep']))
    candidates = {}
    for x in sc['remove']:
        row = dict(x, index=x['logical_indices'][0])
        assert len(row['logical_indices']) == 1 and row['index'] not in sel['keep']
        exact_file(row)
        candidates[row['path']] = row
    helpers = base.old_helpers(cfg)
    accepted = {m: dict(proof=s['accepted_formal200_proof'], report_dir=s['accepted_formal200_report']) for m, s in reg.items() if isinstance(s, dict) and s.get('accepted_formal200_proof')}
    accepted[MID] = dict(proof=str(proof_path), report_dir=str(folder))
    foreign, metadata = base.sources_and_dependencies(reg, accepted, candidates, helpers)
    for p in foreign:
        candidates.pop(p, None)
    assert candidates and len(candidates) <= 117 and (sum((x['bytes'] for x in candidates.values())) < 20 * 1024 ** 3)
    home = Path(sc['home'])
    protected = set(metadata) | {Path(x) for x in cfg['canonical_SHA256']} | {CFG, proof_path, audit_path, Path(cfg['discovery']), M / 'formal_acceptance.json', M / 'actual_completion.json'}
    for m, e in state['models'].items():
        if e['status'] == 'done':
            protected.update((Path(e['report_dir']) / n for n in FIVE))
    protected.update((p for p in home.rglob('*') if p.is_file() and str(p) not in candidates))
    protected.update(folder.iterdir())
    protected.update((p for p in J.rglob('*') if p.is_file()))
    protected.update((p for p in M.rglob('*') if p.is_file()))
    for p in controls:
        protected.update((Path(n) for n in p['external_evidence_sha256'] if Path(n).is_relative_to(S) and Path(n).is_file()))
        protected.update((Path(p['report_dir']) / n for n in FIVE))
    for root in (R / 'scripts', R / 'configs', R / 'universal_eval'):
        protected.update((p for p in root.rglob('*') if p.is_file() and p.suffix in ['.py', '.json', '.yaml', '.yml', '.mjs', '.ir']))
    assert not set(candidates) & {str(x) for x in protected}
    hashes = {}
    for i, p in enumerate(sorted(protected)):
        hashes[str(p)] = sha(p)
        if i % 200 == 0:
            put(H / 'progress.json', dict(UTC=now(), phase='hash_all_preserved_reports_sources_logs_wrong_examples_controls', checked=i + 1, total=len(protected)))
    quiet()
    (H / 'ledger_before.original').write_bytes(LED.read_bytes())
    targetaudit = dict(passed=True, UTC=now(), formal_fullNF4=accept, formal_proof_SHA256=sha(proof_path), all_original3_fresh3_full200_source_external=checked, distinct_SHA_files=len(cache), all184_report_bytes_and_current_G_preserved=True, all59wrong_6invalid_original_complete_groups_small_failed_controls_preserved=True, qualification_false_and_no_G_publication=True, no_deletion_yet=True)
    put(H / 'measurement_target_audit.json', targetaudit)
    p = dict(UTC=now(), config_SHA256=sha(CFG), ledger_before_SHA256=sha(LED), canonical_SHA256=cfg['canonical_SHA256'], proof_SHA256=sha(proof_path), formal_proof=str(proof_path), selection=sel, remove=list(candidates.values()), remove_count=len(candidates), remove_bytes=sum((x['bytes'] for x in candidates.values())), remove_allocated_bytes=sum((x['allocated_bytes'] for x in candidates.values())), foreign_references_preserved=foreign, protected_SHA256=hashes, protected_count=len(hashes), free_before_bytes=shutil.disk_usage(R).free, full_measurement_target_audit=str(H / 'measurement_target_audit.json'), full_measurement_target_audit_SHA256=sha(H / 'measurement_target_audit.json'), all_wrong_full_groups_small_failed_unverified_reports_sources_configs_logs_retained=True, cleanup_not_claimed_sufficient_for_Bee_full200=True)
    put(H / 'plan.json', p)
    return dict(plan_SHA256=sha(H / 'plan.json'), remove_count=p['remove_count'], remove_bytes=p['remove_bytes'], protected_count=len(hashes), projected_free_bytes=p['free_before_bytes'] + p['remove_allocated_bytes'], full_measurement_target_audit_SHA256=sha(H / 'measurement_target_audit.json'), no_deletion_yet=True)

def apply(reviewed_plan):
    from eager_corrective_control_safety_2203 import kernel_lock_owned
    quiet()
    cfg = js(CFG)
    assert sha(H / 'plan.json') == reviewed_plan
    p = js(H / 'plan.json')
    assert sha(CFG) == p['config_SHA256'] and sha(LED) == p['ledger_before_SHA256']
    assert not (H / 'actual_completion.json').exists()
    with (D / 'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        kernel_lock_owned(os.getpid(), D / 'runner.lock')
        quiet(True)
        for i, (n, h) in enumerate(p['protected_SHA256'].items()):
            assert sha(n) == h, ('Protected changed before deletion', n)
            if i % 200 == 0:
                put(H / 'progress.json', dict(UTC=now(), phase='fresh_protected_before_delete', checked=i + 1, total=p['protected_count']))
        for row in p['remove']:
            assert row['index'] not in p['selection']['keep'] and row['index'] not in p['selection']['incorrect']
            exact_file(row)
            assert sha(row['proof']) == row['proof_SHA256']
        helpers = base.old_helpers(cfg)
        state, reg = (js(D / 'state.json'), js(D / 'registry.json'))
        accepted = {m: dict(proof=s['accepted_formal200_proof'], report_dir=s['accepted_formal200_report']) for m, s in reg.items() if isinstance(s, dict) and s.get('accepted_formal200_proof')}
        foreign, _ = base.sources_and_dependencies(reg, accepted, {r['path']: r for r in p['remove']}, helpers)
        assert not foreign, ('New foreign reference appeared', foreign)
        assert all((sha(n) == h for n, h in p['canonical_SHA256'].items()))
        quiet(True)
        manifest = H / 'exact_removed_correct_nonexample_NPZ_SHA256.jsonl'
        assert not manifest.exists()
        removed = []
        failure = None
        free = shutil.disk_usage(R).free
        try:
            with manifest.open('x', encoding='utf-8') as out:
                for row in p['remove']:
                    exact_file(row)
                    Path(row['path']).unlink()
                    removed.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    out.flush()
        except BaseException:
            failure = traceback.format_exc()
        bad = [n for n, h in p['protected_SHA256'].items() if sha(n) != h]
        ok = not failure and (not bad) and (len(removed) == p['remove_count'])
        actual = dict(UTC=now(), actual_returncode=0 if ok else 1, plan_SHA256=reviewed_plan, removed_files=len(removed), removed_bytes=sum((r['bytes'] for r in removed)), removed_allocated_bytes=sum((r['allocated_bytes'] for r in removed)), manifest=str(manifest), manifest_SHA256=sha(manifest), protected_files=p['protected_count'], all_protected_SHA256_unchanged=not bad, protected_errors=bad, error=failure, free_before_bytes=free, free_after_bytes=shutil.disk_usage(R).free, all59_wrong_and_full_original_outer_groups_retained=True, all_reports_source_config_logs_small_failure_unverified_preserved=True, canonical_scores_both_workbooks_state_registry_binding_unchanged=True, exact_pruned_arrays_no_longer_available_original_proof_bytes_retained=True)
        put(H / 'actual_completion.json', actual)
        assert sha(LED) == p['ledger_before_SHA256']
        ledger = js(LED)
        ledger['events'].append(dict(UTC=actual['UTC'], authorization=cfg['user_authorization'], receipt=str(H / 'actual_completion.json'), receipt_SHA256=sha(H / 'actual_completion.json'), manifest=str(manifest), manifest_SHA256=sha(manifest), removed_files=len(removed), removed_bytes=actual['removed_bytes'], all_original_records_kept=True, all_incorrect_cases_preserved=True, intentional_nonexample_matrix_unavailability=True, actual_returncode=actual['actual_returncode']))
        put(LED, ledger)
        assert all((sha(n) == h for n, h in p['canonical_SHA256'].items() if n != str(LED)))
        assert ok
        return dict(actual=actual, actual_SHA256=sha(H / 'actual_completion.json'), ledger_SHA256=sha(LED))
if __name__ == '__main__':
    print(json.dumps(plan() if sys.argv[1] == 'plan' else apply(sys.argv[2]), ensure_ascii=False))

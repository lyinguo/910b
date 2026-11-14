"""Continue one reviewed own-checkpoint candidate after the old real lock exits.

No signals, takeover, active-job changes or publication. Genuine3 is reviewed
before approving a NEWfresh3/fixed200 measurement. Small-sample precision is
reported, not used as an extra 5% gate. Final workbook publication is separate.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import copy, fcntl, hashlib, json, os, shutil, subprocess, time, traceback
R = Path('/root/projects/comperror')
sys.path[:0] = [str(R), str(R / 'scripts')]
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B / 'UnslothE2Bit_after_old_numeric_real_lock_continuation_0105_20261115'
READY = B / 'UnslothE2Bit_own_chat_template_native_candidate_0105_20261115/readiness.json'
J3 = B / 'UnslothE2Bit_own_chat_native_true3_real_FLOCK_0105_20261115'
J200 = B / 'UnslothE2Bit_own_chat_native_approved_fresh3_fixed200_real_FLOCK_0105_20261115'
OLD = B / 'E4B_capacity_recovered_complete_failure_records_approved_fresh3_fixed200_v10_real_FLOCK_1605_20261113'
MID = 'unsloth/gemma-3n-E2B-it'
RUNNER = R / 'scripts/A100_alignment_evidence_runner_20261103.py'
RUNNER_SHA = '58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
OWNED_CONTROLLER = False
def js(p):
    return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def now():
    return datetime.now(timezone.utc).isoformat()
def save(name, value):
    p = H / name
    t = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(t, p)
def state(phase, **kwargs):
    save('state.json', dict(UTC=now(), PID=os.getpid(), model_id=MID, phase=phase,
                           no_G_or_canonical_write=True, **kwargs))
def run_job(home, log):
    assert not (home / 'completion.json').exists() and not (home / 'state.json').exists()
    assert sha(RUNNER) == RUNNER_SHA
    with (H / log).open('xb') as f:
        child = subprocess.Popen([sys.executable, '-B', str(RUNNER), str(home)], cwd=R,
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        state('actual_runner_started', actual_job=str(home), child_PID=child.pid)
        rc = child.wait()
    completion = js(home / 'completion.json')
    assert rc == completion['actual_returncode'] == 0, str(home)
    return completion
def main():
    global OWNED_CONTROLLER
    from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
    import batch_910b as batch
    from batch_910b_support import validate_report
    H.mkdir(exist_ok=True)
    lock = (H / 'continuation.guard').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OWNED_CONTROLLER = True
    assert not (H / 'actual_completion.json').exists()
    ready = js(READY)
    assert ready['passed'] and ready['ready_for_real_runner_after_old_lock_release']
    assert sha(J3 / 'job.json') == ready['job_SHA256']
    job = js(J3 / 'job.json')
    while True:
        batch.check_sources(job['spec'])
        for path, digest in job['protected_canonical_hashes'].items():
            assert sha(path) == digest, path
        locks = runner_lock_owners()
        active = [p for p in project_processes() if p['role'] in ['alignment_runner', 'CPU_verifier_or_audit', 'model_inference']]
        if locks or active or not (OLD / 'completion.json').exists():
            state('waiting_for_original_live_numeric_and_runner_release', actual_FLOCK=locks,
                  actual_active_processes=active, old_completion_exists=(OLD / 'completion.json').exists())
            time.sleep(30)
            continue
        if not batch.device_idle(job['spec']):
            state('waiting_for_original_NPU4_idle')
            time.sleep(30)
            continue
        free = shutil.disk_usage(D).free
        assert free >= 12 * 1024**3, 'Own true3 space estimate including original reserves not met'
        state('old_runner_released_starting_own_true3', old_actual_completion=js(OLD / 'completion.json'), free_bytes=free)
        break
    completion = run_job(J3, 'true3_runner.log')
    accepted = completion['accepted'][-1]
    proof = js(accepted['proof'])
    assert proof['passed'] and proof['sample_count'] == 3 and not proof['historical_record_reuse']
    assert sha(accepted['proof']) == accepted['proof_sha256']
    folder = Path(accepted['report_dir'])
    validate_report(R, folder, MID, 'image-text-to-text', 3, proof)
    line = js(folder / 'operator_lineage.json')['actual_execution']
    trace = [json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]
    executions = [js(t['actual_execution_file']) for t in trace]
    # Empirical reserve, not a worst-case guarantee. Per-sample hard guards remain
    # in the model; no unverified matrices or protected files are pruned.
    estimate = sum(e['archive_bytes'] for e in executions) * 200 // 3 * 4 + 6 * 1024**3
    free = shutil.disk_usage(D).free
    review = dict(approved=True, UTC=now(), model_id=MID, real_NPU3_independent_proof_verified=True,
        numerical_review_verified=True, true3_report=str(folder), true3_proof=str(accepted['proof']),
        true3_proof_SHA256=accepted['proof_sha256'], source_SHA256=job['spec']['source_sha256'],
        small_sample_relative_error_to_D=proof['relative_error_to_A100_table_D'],
        small_sample_5pct_is_not_an_extra_first200_gate=True, first200_is_measurement_not_predetermined_alignment=True,
        empirical_full200_space_estimate_bytes=estimate, actual_free_bytes=free,
        estimate_is_not_worst_case_guarantee=True, original_hard_log_and_sample_guards_preserved=True)
    save('first200_measurement_review.json', review)
    if free < estimate:
        state('true3_passed_formal200_capacity_blocked_no_runner_held', actual_free_bytes=free, empirical_estimate_bytes=estimate)
        save('actual_completion.json', dict(actual_returncode=0, UTC=now(), true3_passed=True,
             formal200_started=False, reason='empirical capacity gate; continue another ready model'))
        return
    assert not J200.exists()
    J200.mkdir()
    formal = copy.deepcopy(job)
    formal['stages'] = [3, 200]
    formal['spec'].update(approved=True, alignment_approval_review=str(H / 'first200_measurement_review.json'),
                          alignment_approval_review_sha256=sha(H / 'first200_measurement_review.json'))
    (J200 / 'job.json').write_text(json.dumps(formal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = run_job(J200, 'fresh3_fixed200_runner.log')
    save('actual_completion.json', dict(actual_returncode=0, UTC=now(), formal_result=result,
         next='Review actual fixed200 three-D errors and all evidence before separate protected publication', no_G_or_canonical_write=True))
    state('completed_evidence_only', formal_job=str(J200))
if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if OWNED_CONTROLLER and not (H / 'actual_completion.json').exists():
            save('actual_completion.json', dict(actual_returncode=1, UTC=now(), error=traceback.format_exc(), no_G_or_canonical_write=True))
        raise

"""Queue publication behind this exact live runner; never launch or alter a model."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import contextlib, fcntl, hashlib, json, os, runpy, time, traceback

R = Path('/root/projects/comperror')
D = R/'workspace/diagnostics/batch_910b'
B = D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0] = [str(R), str(R/'scripts')]
from batch_910b_support import atomic_json, validate_report
from eager_corrective_control_safety_2203 import core_unchanged, kernel_lock_owned


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        raw = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if raw[0] == 'Z':
            return None
        return dict(PID=int(pid), PPid=int(raw[1]), startticks=int(raw[19]),
                    argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],
                    cwd=str((p/'cwd').resolve()))
    except (FileNotFoundError, ProcessLookupError):
        return None


def qualified(p):
    errors = p.get('relative_error_to_current_A100_D', {})
    return (p.get('passed') is True and p.get('sample_count') == 200
            and p.get('actual_NPU_inference') is True
            and p.get('historical_record_reuse') is False
            and p.get('formal_alignment_within_5pct') is True
            and p.get('original_protocol_final_qualification_complete') is True
            and set(errors) == {'server', 'local', 'preserved_original'}
            and all(Decimal(str(x)) <= Decimal('.05') for x in errors.values()))


def canonical_unchanged(expected):
    for path, expected_sha in expected.items():
        assert sha(path) == expected_sha, (path, 'Canonical change before publication')


def main():
    config = Path(sys.argv[1])
    assert sha(config) == sys.argv[2]
    q = read(config)
    assert q['model_id'] == 'zai-org/GLM-4.1V-9B-Thinking'
    J, C, H = (Path(q[k]) for k in ['job', 'controller', 'reservation_home'])
    assert all(p.resolve().is_relative_to(B) for p in [J, C, H])
    assert sha(J/'job.json') == q['job_SHA256'] == '756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163'
    job = read(J/'job.json')
    assert job['stages'] == [3, 200] and not job['write_workbook']
    canonical_unchanged(job['protected_canonical_hashes'])
    core_unchanged()
    kernel_lock_owned(q['initial_runner_PID'], D/'runner.lock')
    runner = proc(q['initial_runner_PID'])
    assert runner and str(J/'job.json') in runner['argv']
    controller = proc(q['initial_controller_PID'])
    assert controller and any(Path(x).name == q['controller_source_name'] for x in controller['argv'])
    H.mkdir(exist_ok=False)
    start = now()
    atomic_json(H/'state.json', dict(UTC=start, PID=os.getpid(), phase='queued_behind_same_formal200_runner',
                initial_runner=runner, initial_controller=controller, no_NN_or_NF_launched=True,
                no_current_source_job_deadline_signal_or_canonical_change=True))
    try:
        with (D/'runner.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            kernel_lock_owned(os.getpid(), D/'runner.lock')
            # The original controller writes its actual completion just after the runner exits.
            controller_deadline = time.monotonic()+120
            while proc(q['initial_runner_PID']) or proc(q['initial_controller_PID']):
                assert time.monotonic() < controller_deadline, 'Original runner/controller did not naturally exit'
                time.sleep(.2)
            canonical_unchanged(job['protected_canonical_hashes'])
            done, sup = read(J/'completion.json'), read(C/'actual_completion.json')
            assert done['actual_returncode'] == done['verifier_returncode'] == sup['actual_returncode'] == 0
            assert [x['sample_count'] for x in done['accepted']] == [3, 200]
            run = done['accepted'][-1]
            assert sha(run['proof']) == run['proof_sha256']
            proof = read(run['proof'])
            if not qualified(proof):
                atomic_json(H/'actual_completion.json', dict(UTC=now(), actual_returncode=0,
                    phase='original200_measurement_not_qualified_no_publication_lock_released',
                    formal_proof=run['proof'], formal_proof_SHA256=run['proof_sha256'],
                    three_D_errors=proof.get('relative_error_to_current_A100_D'), no_canonical_write=True))
                return
            assert done['model_returncode'] == 0
            validate_report(R, run['report_dir'], q['model_id'], 'image-text-to-text', 200, proof)
            atomic_json(H/'state.json', dict(UTC=now(), PID=os.getpid(),
                phase='qualified_full200_real_original_EX_reserved_for_reviewed_dual_publication',
                original_runner_and_controller_naturally_exited=True, same_original_global_EX=True,
                formal_proof=run['proof'], formal_proof_SHA256=run['proof_sha256'],
                three_D_errors=proof['relative_error_to_current_A100_D'], no_NN_or_NF_launched=True))
            deadline = time.monotonic()+1800
            request = H/'reviewed_publication_ready.json'
            while not request.exists():
                assert time.monotonic() < deadline, 'Qualified publication review did not arrive within1800s'
                time.sleep(1)
            req = read(request)
            source, review = (R/req[k] for k in ['publisher', 'reviewed_inputs'])
            assert source.parent == R/'scripts' and source.name.startswith('publish_GLM41Thinking_verified184_')
            assert review.resolve().is_relative_to(B)
            assert sha(source) == req['publisher_SHA256'] and sha(review) == req['reviewed_inputs_SHA256']
            reviewed = read(review)
            assert len(reviewed['changes']) == 1
            change = reviewed['changes'][0]
            assert change['model_id'] == q['model_id'] and change['cell'] == 'G696'
            assert change['formal_proof_sha256'] == run['proof_sha256']
            assert Decimal(str(change['new_value'])) == Decimal(str(proof['metrics']['Accuracy']))
            canonical_unchanged(job['protected_canonical_hashes'])
            kernel_lock_owned(os.getpid(), D/'runner.lock')
            atomic_json(H/'reviewed_publication_admission.json', dict(UTC=now(), PID=os.getpid(),
                same_original_FLOCK_retained=True, request_SHA256=sha(request), **req))
            old_argv = sys.argv
            try:
                sys.argv = [str(source), str(review)]
                with (H/'publication_stdout.json').open('x') as out, (H/'publication_stderr.log').open('x') as err:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        runpy.run_path(str(source), init_globals={'_reviewed_original_lock': lock}, run_name='__main__')
            finally:
                sys.argv = old_argv
            atomic_json(H/'actual_completion.json', dict(UTC=now(), started_UTC=start, actual_returncode=0,
                phase='reviewed_qualified_dual_publication_completed_same_original_lock',
                original_runner_and_controller_naturally_exited=True, no_NN_replay=True, **req))
    except BaseException:
        atomic_json(H/'actual_completion.json', dict(UTC=now(), started_UTC=start, actual_returncode=1,
            phase='publication_reservation_failed_evidence_preserved_lock_released', traceback=traceback.format_exc(),
            no_model_restart_or_signal=True))
        raise


if __name__ == '__main__':
    main()

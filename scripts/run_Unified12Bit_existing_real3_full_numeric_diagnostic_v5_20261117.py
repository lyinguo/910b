"""Read-only full numeric audit of current failed true3; no model rerun.

This does not change the old runner exit1 or grant formal qualification.
The actual global lock excludes every new model while this audit is alive.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,fcntl,subprocess,traceback,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Gemma4Unified12Bit_current_true3_full_numeric_diagnostic_v5_exact_stdlib_20261117'
J=B/'Gemma4Unified12Bit_own_original_HF_BF16_true3_v4_exact_loading_metadata_real_FLOCK_20261117'
REPORT=R/'workspace/reports/google_gemma-4-12B-it/image-text-to-text/20261117_011127'
V=R/'scripts/verify_gemma4_unified_12bit_own_BF16_clean_worker_v5_exact_stdlib_20261117.py'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
    p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()

def main():
    from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
    assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
    done=json.loads((J/'completion.json').read_bytes());assert done['actual_returncode']==1 and done['model_returncode']==1 and done['verifier_returncode']==1 and done['report_dir']==str(REPORT)
    assert {x.name for x in REPORT.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','operator_lineage.json','run.log'}
    original={str(p):sha(p) for p in REPORT.iterdir()};original[str(J/'completion.json')]=sha(J/'completion.json')
    H.mkdir(exist_ok=False);lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush();os.fsync(lock.fileno())
    child=None
    try:
        job=dict(UTC=now(),operation='current_failed_numeric_stage_full_readonly_diagnostic',original_report=str(REPORT),original_job=str(J),original_SHA256=original,verifier=str(V),verifier_SHA256=sha(V),no_new_NPU_or_qualification_or_G_write=True);put('job.json',job)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (H/'verification_stdout.json').open('xb') as out,(H/'verification_stderr.log').open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B',str(V),str(REPORT),'3','--diagnostic-current-failed-report'],cwd=R,env=env,stdout=out,stderr=err,start_new_session=True)
            start=time.monotonic();over=False
            while child.poll() is None:
                over|=time.monotonic()-start>21600;put('state.json',dict(UTC=now(),PID=os.getpid(),child_PID=child.pid,phase='independent_full_numeric_diagnostic_current_real3_no_new_NPU',timeout_review_required_no_signal=over));time.sleep(3)
            rc=child.wait()
        assert all(sha(p)==h for p,h in original.items())
        put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,original_report_and_failure_unchanged=True,no_new_NPU_or_qualification_or_G_write=True,timeout_review_required_no_signal=over,stdout_SHA256=sha(H/'verification_stdout.json'),stderr_SHA256=sha(H/'verification_stderr.log')))
        assert rc==0 and not over
    except BaseException:
        if child is not None:
            while child.poll() is None:time.sleep(3)
        if not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_new_NPU_or_qualification_or_G_write=True))
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()

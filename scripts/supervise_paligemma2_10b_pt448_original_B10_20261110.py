"""Run one exact original ten-request diagnostic with the real shared lock."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
R = Path("/root/projects/comperror")
D = R / "workspace/diagnostics/batch_910b"
H = Path(sys.argv[1]).resolve()
assert H.is_relative_to(D/"A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct")
sys.path[:0] = [str(R),str(R/"scripts")]
import batch_910b as batch
from A100_alignment_evidence_runner_20261103 import unchanged
from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
from batch_910b_support import validate_completed_exit
from eager_corrective_control_safety_2203 import kernel_lock_owned
from validate_alignment_adapter_spec_1813_20261103 import validate_alignment_adapter_spec


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save():
    state["updated_UTC"]=now()
    tmp=H/"state.tmp.json"
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,H/"state.json")


def guard():
    unchanged(job["protected_canonical_hashes"])
    batch.check_sources(spec)
    validate_alignment_adapter_spec(spec,mid)
    assert read(D/"PAUSE")["owner"]=="A100_relative5pct_alignment_20261103_explicit_user_resume"
    kernel_lock_owned(os.getpid(),D/"runner.lock")
    assert sha(spec["protocol_file"])==spec["protocol_sha256"]


def quiet():
    assert not [p for p in project_processes() if p["PID"]!=os.getpid() and p["role"] in {
        "model","alignment_runner","CPU_verifier_or_audit","legacy_controller",
    }]
    assert not Path("/proc/363384").exists()


job=read(H/"job.json")
spec=job["spec"]
mid=job["model_id"]
assert job["scope"]=="original_B10_compatibility_diagnostic_only" and job["write_workbook"] is False
assert spec["approved"] is False and spec["npu_devices"]==[4]
cfg=read(R/spec["config"])
assert cfg["model"]["name"]==mid and cfg["sample_count"]==10 and cfg["sample_range"]=="0:9"
assert cfg["algorithm"]["batch_size"]==10 and cfg["baseline"]["score"]==job["baseline"]
state={"PID":os.getpid(),"current":mid,"phase":"preflight","child_PID":None,
       "sample_count":10,"approved":False,"started_UTC":now()}
folder=R/"workspace/reports"/mid.replace("/","_")/cfg["task_type"]
owned=None
assert not runner_lock_owners()
quiet()
with (D/"runner.lock").open("r") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        guard()
        quiet()
        assert batch.device_idle(spec)
        before={p for p in folder.iterdir() if p.is_dir()}
        env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",
                 MODELSCOPE_OFFLINE="1",EVAL_OFFLINE_ONLY="1",PYTHONUNBUFFERED="1",
                 PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false")
        with (H/"console.log").open("xb") as log:
            cmd=["python3","run.py","--config",spec["config"]]
            owned=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,env=env)
            state.update(phase="running_original_B10",child_PID=owned.pid,argv=cmd)
            save()
            actual=owned.wait()
            owned=None
        report,=[p for p in folder.iterdir() if p.is_dir() and p not in before]
        state.update(model_returncode=actual,child_PID=None,report_dir=str(report))
        validate_completed_exit(actual,report)
        guard()
        proof_path=H/"verification_stdout.json"
        with proof_path.open("xb") as out,(H/"verification_stderr.log").open("xb") as err:
            cmd=["python3",spec["verifier"],str(report),"10"]
            owned=subprocess.Popen(cmd,cwd=R,stdout=out,stderr=err,env=env)
            state.update(phase="independent_original_B10_verification",child_PID=owned.pid,verifier_argv=cmd)
            save()
            verified=owned.wait()
            owned=None
        state.update(verifier_returncode=verified,child_PID=None)
        assert verified==0
        proof=read(proof_path)
        assert proof["passed"] and proof["sample_count"]==10 and proof["actual_NPU_inference"]
        assert not proof["historical_record_reuse"] and not proof["formal_alignment_within_5pct"]
        assert proof["NPU_profiler_kernel_events"]>0 and len(proof["report_sha256"])==5
        assert all(sha(report/name)==digest for name,digest in proof["report_sha256"].items())
        unchanged(proof["external_evidence_sha256"])
        guard()
        state.update(phase="completed_original_B10_gate",actual_returncode=0,proof=str(proof_path),
                     proof_SHA256=sha(proof_path),score=proof["metrics"]["ROUGE-L"],
                     generated_tokens=proof["generated_token_count"],finished_UTC=now())
        save()
    except BaseException:
        if owned is not None:
            owned.wait()
        state.update(phase="failed_original_B10_evidence_preserved",child_PID=None,actual_returncode=1,
                     error=traceback.format_exc(),finished_UTC=now())
        save()
        raise
    finally:
        with (H/"actual_completion.json").open("x",encoding="utf-8") as stream:
            json.dump(state,stream,ensure_ascii=False,indent=2)
            stream.write("\n")

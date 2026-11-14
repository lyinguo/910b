"""Dispatch verified A100 deployment scopes through the existing 910B runners."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

sys.dont_write_bytecode = True
R = Path("/root/projects/comperror")
D = R / "workspace/diagnostics/batch_910b"
BASE = D / "A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct"
STORAGE = R / "workspace/diagnostics/model_runs"
RUNNER = R / "scripts/A100_alignment_evidence_runner_20261103.py"
PAUSE_OWNER = "A100_relative5pct_alignment_20261103_explicit_user_resume"
THRESHOLD = Decimal(".05")
FIVE = {"eval_report.json", "sample_inputs.jsonl", "sample_outputs.jsonl", "run.log", "operator_lineage.json"}
ACTIVE_ROLES = {"model", "alignment_runner", "CPU_verifier_or_audit", "legacy_controller"}
RUNNER_SHA256 = "58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90"
ORIGINAL10_SUPERVISOR_SHA256 = "c4f5b6f40e5545bb474eb548c048fdd7e08e96ef8876578165014f9de1bc1285"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def check(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError(f"Changed input: {path}")
    return path


def scoped(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError(f"Job outside the project scope: {path}")
    return path


def save(path, record):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        try:
            stream.write((json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(reference, observed):
    reference, observed = Decimal(str(reference)), Decimal(str(observed))
    if not reference.is_finite() or not observed.is_finite() or reference == 0:
        raise ValueError("Same-sample reference has no qualified relative denominator")
    return abs(reference - observed) / abs(reference)


def receipt(home, stage):
    name = "actual_completion.json" if stage == "original10" else "completion.json"
    terminal = home / name
    controller = home / "controller_actual_completion.json"
    if terminal.is_file() and controller.is_file():
        end, actual = read(terminal), read(controller)
        if end["actual_returncode"] == actual["actual_returncode"] == 0 and end["verifier_returncode"] == 0:
            return "verified", end
        return "failed_evidence_preserved", end
    if controller.is_file() and read(controller)["actual_returncode"] != 0:
        return "failed_evidence_preserved", read(controller)
    if any((home / name).exists() for name in
           ("pipeline_dispatch.json", "pipeline_worker.json", "launch.json", "state.json", terminal.name)):
        return "started_or_interrupted", None
    return "unstarted", None


def load_job(home, digest, model_id, stages):
    from A100_alignment_evidence_runner_20261103 import scope, unchanged
    from validate_alignment_adapter_spec_1813_20261103 import validate_alignment_adapter_spec
    import batch_910b as batch

    job = read(check(scoped(home) / "job.json", digest))
    spec = job["spec"]
    assert job["model_id"] == model_id and job["write_workbook"] is False
    assert job["operation"] == "A100_D_column_alignment_evidence_only" and job["no_takeover_no_signals_no_main_restart"]
    assert job["stages"] == stages and spec["approved"] is (stages == [3, 200])
    scope(spec, stages, False)
    unchanged(job["protected_canonical_hashes"])
    unchanged({str(R / rel): value for rel, value in job["runner_source_sha256"].items()})
    check(spec["protocol_file"], spec["protocol_sha256"])
    validate_alignment_adapter_spec(spec, model_id)
    batch.check_sources(spec)
    return job


def verified_proof(record, count, spec):
    path = check(record["proof"], record.get("proof_sha256") or record["proof_SHA256"])
    proof = read(path)
    assert proof["sample_count"] == count and proof["passed"] and proof["NEW_real_NPU_inference"]
    assert proof["historical_record_reuse"] is False and not proof["error_indices"]
    assert proof["new_independent_verifier_SHA256"] == spec["source_sha256"][spec["verifier"]]
    report = Path(record["report_dir"]).resolve()
    assert report.is_relative_to(R / "workspace/reports") and proof["report_dir"] == str(report)
    assert {file.name for file in report.iterdir()} == FIVE == set(proof["report_sha256"])
    for name, digest in proof["report_sha256"].items():
        check(report / name, digest)
    return path, proof


def numerical_gate(home, stage3, spec, model_id):
    record, = stage3["accepted"]
    path, proof = verified_proof(record, 3, spec)
    review_path = home / "pipeline_actual_3_numerical_review.json"
    if not review_path.is_file():
        return None
    review = read(review_path)
    assert review["model_id"] == model_id and review["current_NPU3_proof_SHA256"] == sha(path)
    assert review["passed"] and review["diagnostic_only"] and review["paired_3_within5pct"]
    assert review["actual_same_own_three_A100_NPU_pixels_equal"]
    assert review["all883_source_cast_AA_CUDA_and_actual_NPU_SHA_equal"]
    assert review["original_A100_own_prefix_vision_control"] == proof["original_A100_own_prefix_vision_control"]
    assert cohort_error(record, 3, spec) == Decimal(review["same_A100_three_exact_relative"]) <= THRESHOLD
    for item, digest in review["evidence_SHA256"].items():
        check(item, digest)
    return {"path": str(review_path), "SHA256": sha(review_path)}


def cohort_error(record, count, spec):
    from strict_mc_independent_reference_910b import independent_choice

    _, proof = verified_proof(record, count, spec)
    p = read(check(spec["protocol_file"], spec["protocol_sha256"]))
    if count == 3:
        controls = read(check(p["actual_A100_native_control_review"], p["actual_A100_native_control_review_SHA256"]))
        actual, = [item["all_genuine_outputs"] for item in controls if item["name"] == "fullchat_real3"]
        prompt_key, raw_key = "prompt_token_ids", "raw_output"
    else:
        binding = p["original_A100_own_prefix_vision_control"]
        exit_ref, raw_ref = binding["A100_B10_exit"], binding["A100_B10_raw"]
        terminal = read(check(exit_ref["path"], exit_ref["SHA256"]))
        assert terminal["actual_returncode"] == 0 and terminal["stdout_SHA256"] == raw_ref["SHA256"]
        source, = [item for row in check(raw_ref["path"], raw_ref["SHA256"]).read_bytes().splitlines()
                   if row.startswith(b"{") and (item := json.loads(row))["kind"] == "persisted_genuine_A100_original_PrefixLM_scope"]
        actual = source["actual_B10"]
        prompt_key, raw_key = "actual_prompt_ids", "raw"
    outputs = [json.loads(row) for row in (Path(record["report_dir"]) / "sample_outputs.jsonl").read_bytes().splitlines()]
    inputs = [json.loads(row) for row in check(p["dataset"], p["dataset_SHA256"]).read_bytes().splitlines()[:count]]
    assert proof["original_batch10_request_counts"] == [count] and len(outputs) == len(actual) == count
    aa_hits = npu_hits = 0
    for i, (a, n, data) in enumerate(zip(actual, outputs, inputs, strict=True)):
        assert a["index"] == n["index"] == i and a[prompt_key] == p["input_records"][i]["ids"]
        label = str(data["reference"]).strip().upper()
        assert label == str(n["reference"]).strip().upper()
        aa_hits += independent_choice(a[raw_key]) == label
        npu_hits += independent_choice(n["raw_output"]) == label
    assert proof["metrics"]["Strict_MC_Accuracy"] == npu_hits / count
    return relative(aa_hits, npu_hits)


def plan(row):
    from A100_alignment_evidence_runner_20261103 import unchanged

    mid = row["model_id"]
    smoke = scoped(row["smoke_home"])
    b10 = scoped(row["original10_home"])
    formal = scoped(row["formal_home"])
    status, end = receipt(smoke, "true3")
    if status not in {"unstarted", "verified"}:
        return {"phase": status, "stage": "true3", "home": str(smoke)}
    job = load_job(smoke, row["smoke_job_SHA256"], mid, [3])
    spec = job["spec"]
    storage = row["true3_capacity_review"]
    needed = read(check(storage["path"], storage["SHA256"]))["minimum_small_scope_estimate_bytes"]
    if status == "unstarted":
        return {"phase": "ready", "stage": "true3", "home": str(smoke), "minimum_free_bytes": needed,
                "job_SHA256": row["smoke_job_SHA256"], "spec": spec}
    numerics = numerical_gate(smoke, end, spec, mid)
    if not numerics:
        return {"phase": "await_same_A100_3_and_independent_numerics", "home": str(smoke)}
    status, end = receipt(b10, "original10")
    if status == "unstarted":
        return {"phase": "ready", "stage": "original10", "home": str(b10), "minimum_free_bytes": needed + row["original10_extra_bytes"],
                "spec": spec, "smoke_job": job, "numerical_review": numerics}
    if status != "verified":
        return {"phase": status, "stage": "original10", "home": str(b10)}
    error = cohort_error(end, 10, spec)
    if error > THRESHOLD:
        return {"phase": "outside5pct_original10", "home": str(b10), "relative_error": str(error)}
    approved = formal / "approved_fresh3_fixed200_preflight_receipt.json"
    if not approved.is_file():
        return {"phase": "await_approved_fresh3_fixed200", "home": str(formal), "original10_relative_error": str(error)}
    status, end = receipt(formal, "fresh3_fixed200")
    if status not in {"unstarted", "verified"}:
        return {"phase": status, "stage": "fresh3_fixed200", "home": str(formal)}
    gate = read(approved)
    assert gate["approved"] and gate["stages"] == [3, 200]
    formal_job = load_job(formal, gate["job_SHA256"], mid, [3, 200])
    approval = read(check(formal_job["spec"]["alignment_approval_review"], formal_job["spec"]["alignment_approval_review_sha256"]))
    assert approval["real_NPU3_independent_proof_verified"] and approval["numerical_review_verified"]
    pairs = approval["exact_same_A100_vs_910B_small_cohorts"]
    assert set(pairs) == {"3", "10"} and all(0 <= Decimal(p["exact_relative"]) <= THRESHOLD for p in pairs.values())
    unchanged({p["proof"]: p["SHA256"] for p in pairs.values()})
    assert pairs["3"]["SHA256"] == read(smoke / "completion.json")["accepted"][0]["proof_sha256"]
    assert pairs["10"]["SHA256"] == read(b10 / "actual_completion.json")["proof_SHA256"]
    if status == "verified":
        assert [p["sample_count"] for p in end["accepted"]] == [3, 200]
        _, proof = verified_proof(end["accepted"][-1], 200, formal_job["spec"])
        return {"phase": "await_three_D_and_publication" if proof["formal_alignment_within_5pct"] else "outside5pct_fixed200",
                "home": str(formal), "table_D_relative_error": proof["relative_error_to_A100_table_D"],
                "score": end["accepted"][-1]["score"], "proof": end["accepted"][-1]["proof"]}
    return {"phase": "ready", "stage": "fresh3_fixed200", "home": str(formal),
            "minimum_free_bytes": approval["storage_preflight"]["minimum_preflight_bytes"],
            "job_SHA256": gate["job_SHA256"], "spec": formal_job["spec"]}


def prepare_original10(row, action, supervisor):
    from validate_alignment_adapter_spec_1813_20261103 import validate_alignment_adapter_spec
    import batch_910b as batch

    home, spec = scoped(action["home"]), action["spec"]
    protocol = read(check(spec["protocol_file"], spec["protocol_sha256"]))
    template = check(supervisor["path"], supervisor["SHA256"])
    owner = protocol.get("exact_own_original_B10_supervisor")
    name = Path(owner["source"]).name if owner else "supervise_Gemma3_original_B10_20261109.py"
    source = home / name
    if owner:
        assert str(source) == owner["source"] and supervisor["SHA256"] == owner["SHA256"]
    cfg_path = R / spec["config"]
    cfg = read(cfg_path)
    assert cfg["algorithm"]["batch_size"] == 10 and cfg["sample_range"] == "0:199"
    assert cfg["model"]["name"] == row["model_id"] and Decimal(str(cfg["baseline"]["threshold"])) == THRESHOLD
    config = cfg_path.with_name(cfg_path.stem + "-pipeline-B10.yaml")
    review = action["numerical_review"]
    cfg.update(name=config.stem, sample_count=10, sample_range="0:9")
    cfg["model"]["kwargs"].update(numerical_review=review["path"], numerical_review_sha256=review["SHA256"])
    b10 = dict(spec, config=str(config.relative_to(R)), source_sha256=dict(spec["source_sha256"]))
    value = {"scope": "original_B10_compatibility_diagnostic_only", "model_id": row["model_id"],
             "write_workbook": False, "baseline": cfg["baseline"]["score"], "spec": b10,
             "protected_canonical_hashes": action["smoke_job"]["protected_canonical_hashes"],
             "current_3_pair_review": review["path"], "current_3_pair_review_SHA256": review["SHA256"]}
    home.mkdir(exist_ok=True)
    assert {path.name for path in home.iterdir()} <= {source.name, "job.json"}
    if not source.exists():
        with source.open("xb") as stream:
            stream.write(template.read_bytes())
    check(source, supervisor["SHA256"])
    if not config.exists():
        save(config, cfg)
    assert read(config) == cfg
    b10["source_sha256"][b10["config"]] = sha(config)
    validate_alignment_adapter_spec(b10, row["model_id"])
    batch.check_sources(b10)
    job_path = home / "job.json"
    if not job_path.exists():
        save(job_path, value)
    assert read(job_path) == value
    return source, sha(job_path)


def worker(dispatch):
    from A100_alignment_evidence_runner_20261103 import unchanged
    from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
    import batch_910b as batch

    home = scoped(dispatch.parent)
    assert dispatch.name == "pipeline_dispatch.json"
    record = read(dispatch)
    save(home / "pipeline_worker.json", {"pid": os.getpid(), "UTC": now()})
    actual = None
    try:
        job = read(check(home / "job.json", record["job_SHA256"]))
        if record["stage"] == "original10":
            source = check(record["source"], ORIGINAL10_SUPERVISOR_SHA256)
            assert source.parent == home and job["scope"] == "original_B10_compatibility_diagnostic_only"
        else:
            source = check(RUNNER, RUNNER_SHA256)
            assert record["source"] == str(source)
            assert job["stages"] == {"true3": [3], "fresh3_fixed200": [3, 200]}[record["stage"]]
        assert job["write_workbook"] is False
        unchanged(job["protected_canonical_hashes"])
        assert read(D / "PAUSE")["owner"] == PAUSE_OWNER
        assert not runner_lock_owners() and not [p for p in project_processes() if p["role"] in ACTIVE_ROLES]
        assert shutil.disk_usage(STORAGE).free >= record["minimum_free_bytes"] and batch.device_idle(job["spec"])
        # The existing runner owns the real lock throughout inference and verification.
        actual = subprocess.run([sys.executable, str(source), str(home)], cwd=R,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).returncode
    finally:
        save(home / "controller_actual_completion.json",
             {"UTC": now(), "actual_returncode": actual, "dispatch_SHA256": sha(dispatch), "stage": record["stage"]})
    return actual


def pick(rows, free_bytes, busy):
    if busy or any(action["phase"] == "started_or_interrupted" for _, action in rows):
        return None
    return next(((row, action) for row, action in rows
                 if action["phase"] == "ready" and action["minimum_free_bytes"] <= free_bytes), None)


def dispatch_ready(row, action, queue):
    import fcntl
    from A100_alignment_incremental_monitor_20261103 import project_processes
    import batch_910b as batch

    home = scoped(action["home"])
    with (D / "runner.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"phase": "runner_lock_became_busy"}
        assert read(D / "PAUSE")["owner"] == PAUSE_OWNER and not Path("/proc/363384").exists()
        assert not [p for p in project_processes() if p["role"] in ACTIVE_ROLES]
        for model in queue["models"]:
            for field, stage in (("smoke_home", "true3"), ("original10_home", "original10"), ("formal_home", "fresh3_fixed200")):
                if receipt(scoped(model[field]), stage)[0] == "started_or_interrupted":
                    return {"phase": "another_job_claimed"}
        if receipt(home, action["stage"])[0] != "unstarted":
            return {"phase": "already_dispatched_or_completed"}
        if shutil.disk_usage(STORAGE).free < action["minimum_free_bytes"] or not batch.device_idle(action["spec"]):
            return {"phase": "device_or_capacity_changed_before_dispatch"}
        if action["stage"] == "original10":
            source, digest = prepare_original10(row, action, queue["original10_supervisor"])
        else:
            source = check(RUNNER, queue["runner_SHA256"])
            digest = action["job_SHA256"]
            check(home / "job.json", digest)
        claim = home / "pipeline_dispatch.json"
        save(claim, {"UTC": now(), "stage": action["stage"], "job_SHA256": digest, "source": str(source),
                     "source_SHA256": sha(source), "minimum_free_bytes": action["minimum_free_bytes"]})
    with (home / "pipeline_console.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(claim)],
                                   cwd=R, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"phase": "dispatched", "stage": action["stage"], "model_id": row["model_id"],
            "home": str(home), "worker_pid": process.pid, "dispatch_SHA256": sha(claim)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", nargs="?")
    parser.add_argument("--sha256")
    parser.add_argument("--advance", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.queue or not args.sha256:
        parser.error("queue and --sha256 are required")
    from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners

    queue = read(check(scoped(args.queue), args.sha256))
    assert Decimal(queue["maximum_relative_error"]) == THRESHOLD
    assert read(D / "PAUSE")["owner"] == PAUSE_OWNER and not Path("/proc/363384").exists()
    assert queue["runner_SHA256"] == RUNNER_SHA256
    assert queue["original10_supervisor"]["SHA256"] == ORIGINAL10_SUPERVISOR_SHA256
    check(RUNNER, RUNNER_SHA256)
    assert len({row["model_id"] for row in queue["models"]}) == len(queue["models"])
    free_bytes = shutil.disk_usage(STORAGE).free
    owners = runner_lock_owners()
    processes = [p for p in project_processes() if p["role"] in ACTIVE_ROLES]
    rows = []
    for row in queue["models"]:
        try:
            action = plan(row)
        except (AssertionError, KeyError, ValueError, FileNotFoundError) as exc:
            action = {"phase": "blocked_evidence_or_configuration", "detail": f"{type(exc).__name__}: {exc}"}
        rows.append((row, action))
    active = pick(rows, free_bytes, bool(owners or processes))
    public = []
    for row, action in rows:
        item = {key: value for key, value in action.items() if key not in {"spec", "smoke_job", "numerical_review"}}
        if action["phase"] == "ready" and action["minimum_free_bytes"] > free_bytes:
            item.update(phase="blocked_storage", short_bytes=action["minimum_free_bytes"] - free_bytes)
        public.append(dict(model_id=row["model_id"], A100_deployment=row["A100_deployment"], **item))
    result = {"UTC": now(), "free_project_bytes": free_bytes, "maximum_relative_error": str(THRESHOLD),
              "kernel_FLOCK": owners, "running_project_evaluations": processes, "queue": public,
              "dispatch": dispatch_ready(*active, queue) if active and args.advance else None}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

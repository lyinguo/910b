ORIGINAL_RESOURCE_GATE = b'"""Resource preflight for the existing immutable three-model A100 deployment queue."""\nfrom datetime import datetime, timezone\nimport hashlib\nimport json\nfrom pathlib import Path\nimport shutil\nimport stat\nimport sys\n\nsys.dont_write_bytecode = True\nR = Path("/root/projects/comperror")\nD = R / "workspace/diagnostics/batch_910b"\nB = D / "A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct"\nS = R / "workspace/diagnostics/model_runs"\npipe_file = R / "scripts/A100_deploy_test_pipeline_20261109.py"\nassert hashlib.sha256(pipe_file.read_bytes()).hexdigest() == "5bd69062bd638ff251d869ade247fc04dc4a738a6fb13c2c0b42ec25b70244da"\nsys.path[:0] = [str(pipe_file.parent), str(R)]\nimport A100_deploy_test_pipeline_20261109 as pipe\nfrom A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners\nqueue = pipe.read(pipe.check(B / "A100_deploy_test_pipeline_queue_0837_20261109/queue.json",\n                             "558956b0c3ad1d85df76953fd10cc0e9f361cbd596301d4d63648ce220e0ef59"))\nstate = pipe.read(D / "state.json")\nassert pipe.read(D / "PAUSE")["owner"] == pipe.PAUSE_OWNER\nentries = []\ndataset_digests = {}\nfor row in queue["models"]:\n    model = row["model_id"]\n    home = pipe.scoped(row["smoke_home"])\n    job = pipe.read(pipe.check(home / "job.json", row["smoke_job_SHA256"]))\n    spec = job["spec"]\n    protocol = pipe.read(pipe.check(spec["protocol_file"], spec["protocol_sha256"]))\n    config = pipe.read(pipe.check(R / spec["config"], spec["source_sha256"][spec["config"]]))\n    root = Path(config["model"]["model_path"])\n    assert str(root) in state["models"][model]["weight_paths"] and root.is_relative_to("/mnt/afs")\n    assert protocol["model_id"] == model and protocol["weight_path"] == str(root.resolve())\n    errors = []\n    required = protocol["source_files"]\n    if not root.is_dir():\n        errors.append({"path": str(root), "reason": "original_registered_directory_unavailable"})\n    else:\n        for name, record in required.items():\n            assert name == Path(name).name and name.endswith(".safetensors")\n            try:\n                m = (root / name).stat()\n                if not stat.S_ISREG(m.st_mode) or m.st_size != record["bytes"]:\n                    errors.append({"file": name, "reason": "original_size_mismatch"})\n            except OSError as exc:\n                errors.append({"file": name, "errno": exc.errno})\n        if len(required) > 1:\n            try:\n                index = pipe.read(root / "model.safetensors.index.json")["weight_map"]\n                if set(index) != set(protocol["source_tensors"]) or set(index.values()) != set(required):\n                    errors.append({"file": "model.safetensors.index.json", "reason": "original_index_layout_mismatch"})\n            except (OSError, ValueError, KeyError) as exc:\n                errors.append({"file": "model.safetensors.index.json", "reason": type(exc).__name__})\n    dataset = Path(protocol["dataset"]).resolve()\n    assert dataset.is_relative_to(R / "workspace/datasets") and dataset == Path(config["data"]["dataset_path"]).resolve()\n    if dataset not in dataset_digests:\n        dataset_digests[dataset] = pipe.sha(dataset)\n    assert dataset_digests[dataset] == protocol["dataset_SHA256"]\n    minimum = pipe.read(pipe.check(row["true3_capacity_review"]["path"],row["true3_capacity_review"]["SHA256"]))["minimum_small_scope_estimate_bytes"]\n    phase, terminal = pipe.receipt(home, "true3")\n    entries.append({"model_id": model, "phase": phase, "minimum_true3_bytes": minimum,\n                    "required_weight_files": len(required), "original_weight_layout_errors": errors,\n                    "resource_metadata_gate_passed": not errors, "old_attempt_exit": None if terminal is None else terminal["actual_returncode"],\n                    "needs_new_private_retry_if_old_failed": phase == "failed_evidence_preserved"})\nprint(json.dumps({"UTC": datetime.now(timezone.utc).isoformat(), "free_bytes": shutil.disk_usage(S).free,\n                  "queue_preflight": entries, "exact_full_weight_digest_still_required_at_real_load": True,\n                  "current_runner_locks": runner_lock_owners(), "active_project_processes": [p for p in project_processes() if p["role"] in pipe.ACTIVE_ROLES],\n                  "A100_mount_mnt2_path_exists_on_910B": Path("/mnt2/afs").is_dir(),\n                  "read_only_no_dispatch": True}, ensure_ascii=False))\n\r\n'
ORIGINAL_RESOURCE_GATE_SHA256 = 'bc810ce69760512bde98329e7085d1e458a7c074ce0326b41f0edb95d8a7771b'
"""Filter the immutable 910B queue by the live original-resource gate."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0] = [str(R / 'scripts'), str(R)]


def eligible_actions(rows, gate):
    result = []
    for row, planned in rows:
        item = dict(planned)
        resource = gate[row['model_id']]
        item['original_weight_gate_passed'] = resource['resource_metadata_gate_passed']
        if not item['original_weight_gate_passed'] and item['phase'] != 'started_or_interrupted':
            item['prior_job_phase'] = item['phase']
            item['phase'] = 'skipped_currently_original_weights_unavailable'
            item['resource_errors'] = resource['original_weight_layout_errors']
        result.append((row, item))
    return result


def main():
    from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
    import A100_deploy_test_pipeline_20261109 as pipe

    assert pipe.sha(R/'scripts/A100_deploy_test_pipeline_20261109.py') == '5bd69062bd638ff251d869ade247fc04dc4a738a6fb13c2c0b42ec25b70244da'
    queue = pipe.read(pipe.check(B/'A100_deploy_test_pipeline_queue_0837_20261109/queue.json',
                                 '558956b0c3ad1d85df76953fd10cc0e9f361cbd596301d4d63648ce220e0ef59'))
    assert hashlib.sha256(ORIGINAL_RESOURCE_GATE).hexdigest() == ORIGINAL_RESOURCE_GATE_SHA256
    gate_output = io.StringIO()
    with contextlib.redirect_stdout(gate_output):
        exec(compile(ORIGINAL_RESOURCE_GATE, 'A100_own_original_resource_gate_1053_remote.py', 'exec'), {})
    checked = json.loads(gate_output.getvalue())
    gate = {row['model_id']:row for row in checked['queue_preflight']}
    rows = []
    for row in queue['models']:
        try:
            action = pipe.plan(row)
        except (AssertionError, KeyError, ValueError, FileNotFoundError) as exc:
            action = {'phase':'blocked_evidence_or_configuration','detail':f'{type(exc).__name__}: {exc}'}
        rows.append((row,action))
    rows = eligible_actions(rows, gate)
    owners = runner_lock_owners()
    active = [x for x in project_processes() if x['role'] in pipe.ACTIVE_ROLES]
    free = shutil.disk_usage(pipe.STORAGE).free
    next_item = pipe.pick(rows, free, bool(owners or active))
    picked = next_item[0]['model_id'] if next_item else None
    if next_item:
        assert gate[picked]['resource_metadata_gate_passed']
    result = {'checked_UTC':checked['UTC'],'pending_eligible_original_queue_model':picked,
              'currently_skipped_original_weights':[row['model_id'] for row,item in rows
                   if item['phase'] == 'skipped_currently_original_weights_unavailable'],
              'queue':[{'model_id':row['model_id'],
                         **{k:v for k,v in item.items() if k not in ('spec','smoke_job','numerical_review')}}
                       for row,item in rows],
              'current_actual_project_processes':active,'kernel_FLOCK_owners':owners,
              'busy_with_actual_job':bool(owners or active),
              'no_eligible_fixed_queue_stage_continue_other_original_weight_candidates':not next_item and not (owners or active),
              'dispatch':pipe.dispatch_ready(*next_item,queue) if next_item else None}
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()

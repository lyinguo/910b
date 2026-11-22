"""Preserve immutable serial model jobs and explicit useful source additions."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone, timedelta
import base64, hashlib, json, os, subprocess

R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
C = D / 'code_publication_source_layout_20261114'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def js(p):
    return json.loads(Path(p).read_bytes())

def put(p, value):
    p = Path(p)
    temp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temp, p)

def main():
    plan = Path(sys.argv[1])
    assert sha(plan) == sys.argv[2]
    q = js(plan)
    pointer = D / 'A100_ALIGNMENT_HANDOFF_CURRENT.json'
    before = pointer.read_bytes()
    assert hashlib.sha256(before).hexdigest() == q['expected_pointer_SHA256']
    ptr = json.loads(before)
    H = Path(ptr['handoff_directory'])
    closing_path = H / 'closing_addendum.json'
    closing_before = closing_path.read_bytes()
    assert hashlib.sha256(closing_before).hexdigest() == q['expected_closing_SHA256']
    parent = Path(ptr['new_primary_pending'])
    assert sha(parent) == ptr['new_primary_pending_SHA256'] == q['expected_pending_SHA256']
    canonical = ptr['canonical_SHA256']
    assert ptr['current_verified_count'] == 185
    assert ptr['current_retention_ledger_SHA256'] == '5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0'
    assert all(sha(n) == h for n, h in canonical.items())
    rows = {r['path']: r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows) == q['expected_parent_source_count']
    assert all(sha(n) == r['SHA256'] for n, r in rows.items())
    note = H / 'late_actual_notes' / q['round_name']
    note.mkdir(exist_ok=False)
    (note / 'previous_pointer.json').write_bytes(before)
    (note / 'previous_closing_addendum.json').write_bytes(closing_before)
    for record in q['local_original_records']:
        raw = base64.b64decode(record['base64'], validate=True)
        assert hashlib.sha256(raw).hexdigest() == record['SHA256']
        dest = note / 'local_original_records' / record['name']
        assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    paths = list(q['new_primary_paths']) + [str(Path(__file__).relative_to(R))]
    assert all(sha(R / n) == h for n, h in q['new_primary_expected_SHA256'].items())
    runtime = []
    job_stems = [
        'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122',
        'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122',
        'XtunerLlava3_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v2_IModel_real_FLOCK_20261122',
        'LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122',
        'LlavaMore_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122',
    ]
    for stem in job_stems:
        p = B / stem / 'job.json'
        if not p.is_file():
            continue
        job = js(p)
        declared = job['spec']['source_sha256']
        assert all(sha(R / n) == h for n, h in declared.items())
        paths.extend(declared)
        runtime.append(dict(job=str(p), SHA256=sha(p), source_SHA256=declared))
        config = Path(job['spec']['config'])
        for variant in [config, config.with_stem(config.stem + '-smoke')]:
            if (R / variant).is_file():
                paths.append(str(variant))
    for n in sorted(set(paths)):
        p = R / n
        assert p.is_file() and p.resolve().is_relative_to(R)
        assert n == 'run.py' or n.startswith(('scripts/', 'configs/', 'universal_eval/', 'tests/'))
        h = sha(p)
        if str(p) in rows:
            assert rows[str(p)]['SHA256'] == h, n
        else:
            dest = note / 'primary_source_snapshots' / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            rows[str(p)] = dict(path=str(p), relative=n, SHA256=h, bytes=p.stat().st_size,
                snapshot=str(dest), role='Own original protocol native implementation/full numeric verifier/real config consumer, complete lossless CPU persistence, tests or exact source and serial-state maintenance')
    reader = R / 'scripts/maintenance/read_verified185_Xtuner_LlavaMore_serial_ready_frozen_monitor_actual_20261122_remote.py'
    result = subprocess.run(['python3', '-B', str(reader)], cwd=R, capture_output=True,
        text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode == 0, result.stderr
    current = json.loads(result.stdout)
    assert current['all_current_canonical_fresh_SHA256'] == canonical
    actual = current['current_Xtuner_same_serial_actual']
    assert actual['npu_smi']['returncode'] == 0
    monitor = next(v for k, v in current['complete_monitor_latest_checkpoint'].items() if k.endswith('/latest.json'))
    assert not monitor['source_errors']
    owners = current['actual_original_lock_owner_proc']
    owner_ids = {v['owner']['PID'] for v in owners if 'owner' in v}
    specific = [e for e in monitor.get('inspection_errors', []) if e.get('reason') == 'runner_lock_owner_not_in_project_process_snapshot']
    closed = bool(specific and all(
        {o['PID'] for o in e['owners']}.issubset(owner_ids) for e in specific) and all(
        v.get('cwd') == '/root' and len(v.get('argv', [])) == 4 and
        Path(v['argv'][2]).name == 'A100_alignment_evidence_runner_20261103.py' and
        v['argv'][3].endswith('/XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122')
        for v in owners))
    unknown = [e for e in monitor.get('inspection_errors', []) if e not in specific] if closed else monitor.get('inspection_errors', [])
    assert not unknown, unknown
    assert not monitor.get('new_log_error_candidates'), monitor.get('new_log_error_candidates')
    model = [p for p in actual['actual_processes'] if 'run.py' in p['argv']]
    nf = [p for p in actual['actual_processes'] if any(Path(a).name.startswith('verify_') for a in p['argv'])]
    captures = [{k: c.get(k) for k in ['home', 'committed_step_counts', 'total_committed_native_forwards',
        'last_committed_step', 'pending_partial_line', 'NPZ']} for c in actual['captures']]
    forwards = sum(c['total_committed_native_forwards'] for c in captures)
    last = max((c['last_committed_step']['UTC'] for c in captures if c.get('last_committed_step')), default=None)
    age = (datetime.fromisoformat(actual['UTC']) - datetime.fromisoformat(last)).total_seconds() if last else None
    currently_submitting = bool(model and age is not None and age < 10)
    writing_fds = [{ 'PID': p['PID'], 'fd': {k: v for k, v in p.get('fd', {}).items() if v.endswith('.npz')}} for p in model]
    writing_fds = [v for v in writing_fds if v['fd']]
    persistence = bool(model and writing_fds and not currently_submitting)
    current['current_specific_root_cwd_monitor_owner_check_closed'] = closed
    current['current_unknown_inspection_errors'] = unknown
    current['current_formal_model_task_active'] = bool(model or nf)
    current['current_recent_NPU_forward_submission_active'] = currently_submitting
    current['current_complete_numeric_persistence_active_not_new_NN_or_NF'] = persistence
    current['current_complete_numeric_persistence_open_FDs'] = writing_fds
    current['current_committed_native_forwards'] = forwards
    current['current_last_committed_step_age_seconds'] = age
    current['last_qualified_full200_NF_Beijing'] = '2026-11-22T03:34:24.274295+08:00'
    current['last_qualified_dual_audit_Beijing'] = '2026-11-22T03:47:22.926705+08:00'
    current['current_counts'] = dict(delivered=185, numeric_within=170, numeric_outside=15,
        complete_weights_pending=205, confirmed_missing=310, original_full_qualified=71,
        newly_complete200_this_round=0, newly_delivered_G_this_round=0, newly_qualified_this_round=0)
    snapshot = note / 'Xtuner_same_v2_and_More_original_probability_serial_complete_actual.json'
    put(snapshot, current)
    more = current['LlavaMore_new_source_ready_and_serial_waiter']
    more_ready = next(v for k, v in more.items() if '/LlavaMore_own_original_FP16_letter_source_readiness_v1_' in k and k.endswith('/readiness.json'))
    more_state = next(v for k, v in more.items() if '/LlavaMore_own_original_probability_ready_after_' in k and k.endswith('/state.json'))
    assert more_ready['passed'] and more_ready['actual_NEW_model_forwards'] == 0
    compact = dict(UTC=current['UTC'], counts=current['current_counts'], Xtuner_captures=captures,
        Xtuner_controller=actual['controller'], Xtuner_current_job=actual['job'],
        current_committed_native_forwards=forwards, current_model_task_active=bool(model), current_fullNF_active=bool(nf),
        recent_forward_submission_active=currently_submitting, numeric_persistence_not_new_NN_or_NF=persistence,
        numeric_persistence_open_FDs=writing_fds, last_committed_step_age_seconds=age,
        actual_original_lock_owners=owners, actual_free_bytes=actual['free_bytes'],
        specific_root_cwd_monitor_owner_check_closed=closed, unknown_inspection_errors=unknown,
        new_log_error_candidates=monitor.get('new_log_error_candidates', []),
        More_source_readiness={k: more_ready[k] for k in ['UTC', 'passed', 'model_id', 'job', 'job_SHA256', 'YAML', 'YAML_SHA256', 'protocol', 'protocol_SHA256', 'three_D_values']},
        More_actual_serial_waiter=more_state, actual_runtime_jobs=runtime,
        full_actual_snapshot=str(snapshot), full_actual_snapshot_SHA256=sha(snapshot))
    full = q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT', str(len(rows))).replace('CURRENT_ACTUAL_JSON', json.dumps(compact, ensure_ascii=False))
    full_path = note / q['full_note_name']
    full_path.write_text(full, encoding='utf8')
    pending = C / q['pending_name']
    assert not pending.exists()
    put(pending, dict(UTC=datetime.now(timezone.utc).isoformat(), parent_pending=str(parent),
        parent_pending_SHA256=sha(parent), source_count=len(rows),
        all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],
        Code_Day_already_pushed='2026-11-22', Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',
        no_second_same_day_commit=True, next_daily_review='Day23 review all new useful source/config consumers/dependencies/Windows control/tests; fresh SHA, full diff/all outgoing and reviewed publisher plan before one normal push'))
    updates = dict(UTC=datetime.now(timezone.utc).isoformat(), new_primary_pending=str(pending),
        new_primary_pending_SHA256=sha(pending), latest_pending=str(pending), latest_pending_SHA256=sha(pending),
        current_primary_source_count=len(rows), current_source_count=len(rows), latest_actual_note=str(full_path),
        latest_actual_note_SHA256=sha(full_path), latest_full_actual_summary=str(full_path),
        latest_full_actual_summary_SHA256=sha(full_path), latest_actual_snapshot=str(snapshot),
        latest_actual_snapshot_SHA256=sha(snapshot), current_actual_stage='Xtuner_same_v2_serial_complete_numeric_persistence_and_More_original_probability_source_ready_waiter',
        current_formal_model_stage='Actual complete numeric persistence after committed NPU steps is not new NN/fullNF; same real runner/model remains active, More waits actual release',
        current_model_process_active=bool(model), current_full_numerical_verifier_active=bool(nf),
        current_formal_model_task_active=bool(model or nf), current_recent_NPU_forward_submission_active=currently_submitting,
        current_complete_numeric_persistence_active_not_new_NN_or_NF=persistence,
        current_model_NN_or_NF_active=bool(currently_submitting or nf),
        current_actual_model_NN_or_NF_active=bool(currently_submitting or nf),
        current_NN_or_NF_active=bool(currently_submitting or nf),
        current_actual_model_process_or_NF_active=bool(model or nf),
        current_genuine_committed_native_forwards=forwards, current_inspection_owner_specific_root_cwd_filter_closed=closed,
        current_next_original_model='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning',
        current_next_original_model_source_ready=bool(more_ready['passed']),
        current_next_original_model_actual_NPU_started=not more_state.get('genuine_own_NPU3_not_started', True),
        canonical_SHA256=canonical)
    assert pointer.read_bytes() == before and closing_path.read_bytes() == closing_before
    assert all(sha(n) == h for n, h in canonical.items())
    assert all(sha(n) == r['SHA256'] for n, r in rows.items())
    closing = json.loads(closing_before)
    closing.update(updates)
    put(closing_path, closing)
    ptr.update(updates)
    ptr['closing_addendum_SHA256'] = sha(closing_path)
    put(pointer, ptr)
    receipt = dict(passed=True, UTC=datetime.now(timezone.utc).isoformat(), pointer_SHA256=sha(pointer),
        closing_SHA256=sha(closing_path), pending=str(pending), pending_SHA256=sha(pending),
        source_count=len(rows), full_actual_note=str(full_path), full_actual_note_SHA256=sha(full_path),
        actual_snapshot=str(snapshot), actual_snapshot_SHA256=sha(snapshot),
        current_committed_native_forwards=forwards, current_captures=captures,
        current_model_task_active=bool(model), current_fullNF_process_active=bool(nf),
        complete_numeric_persistence_not_new_NN_or_NF=persistence,
        More_source_ready=more_ready['passed'], More_actual_state=more_state,
        unknown_inspection_errors=unknown, no_active_source_job_config_deadline_signal_mutation=True,
        no_canonical_mutation=True, no_deletion=True, no_second_Day22_commit=True)
    put(note / 'round_preservation_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()

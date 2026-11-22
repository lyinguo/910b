"""Preserve finite actual handoff and every explicit useful source without changing live jobs."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, os, subprocess
R = Path('/root/projects/comperror')
D = R/'workspace/diagnostics/batch_910b'
C = D/'code_publication_source_layout_20261114'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def js(p):
    return json.loads(Path(p).read_bytes())

def put(p, value):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    os.replace(tmp, p)

def main():
    plan = Path(sys.argv[1])
    assert sha(plan) == sys.argv[2]
    q = js(plan)
    pointer = D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'
    pb = pointer.read_bytes()
    assert hashlib.sha256(pb).hexdigest() == q['expected_pointer_SHA256']
    ptr = json.loads(pb)
    closing = Path(ptr['handoff_directory'])/'closing_addendum.json'
    cb = closing.read_bytes()
    assert hashlib.sha256(cb).hexdigest() == q['expected_closing_SHA256']
    parent = Path(ptr['new_primary_pending'])
    assert sha(parent) == q['expected_pending_SHA256']
    rows = {r['path']: r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows) == 2902
    assert all(sha(n) == r['SHA256'] for n, r in rows.items())
    assert all(sha(n) == h for n, h in ptr['canonical_SHA256'].items())
    note = Path(ptr['handoff_directory'])/'late_actual_notes'/q['round_name']
    note.mkdir(exist_ok=False)
    (note/'previous_pointer.json').write_bytes(pb)
    (note/'previous_closing_addendum.json').write_bytes(cb)
    local = []
    for r in q['local_original_records']:
        b = base64.b64decode(r['base64'], validate=True)
        assert hashlib.sha256(b).hexdigest() == r['SHA256']
        dest = note/'local_original_records'/r['name']
        assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b)
        local.append(dict(path=str(dest), SHA256=r['SHA256'], bytes=len(b)))
    sources = dict(q['new_primary_expected_SHA256'])
    assert str(Path(__file__).relative_to(R)) in sources
    for n, h in sorted(sources.items()):
        p = R/n
        assert n.startswith(('scripts/', 'configs/', 'universal_eval/', 'tests/'))
        assert p.is_file() and p.resolve().is_relative_to(R) and sha(p) == h, n
        if str(p) in rows:
            assert rows[str(p)]['SHA256'] == h
        else:
            dest = note/'primary_source_snapshots'/n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            rows[str(p)] = dict(path=str(p), relative=n, SHA256=h, bytes=p.stat().st_size,
                snapshot=str(dest), role='Own Turkish original FP16 input/source/constructor/native/full numerical verifier, exact serial continuation, failed Base full-value diagnostic, true runtime YAML/protocol/smoke and Windows control dependencies; all original failed stages preserved')
    reader = R/'scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py'
    run = subprocess.run(['python3', '-B', str(reader)], cwd=R, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert run.returncode == 0, run.stderr
    actual = json.loads(run.stdout)
    for n, z in actual['current_runtime_config_SHA256'].items():
        p = R/n
        assert sha(p) == z['SHA256']
        if str(p) in rows:
            assert rows[str(p)]['SHA256'] == z['SHA256']
        else:
            dest = note/'primary_source_snapshots'/n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            rows[str(p)] = dict(path=str(p), relative=n, SHA256=z['SHA256'], bytes=z['bytes'],
                snapshot=str(dest), role='Actual direct run.py independent Turkish own original YAML/protocol/smoke config, exact runtime consumer')
    x = actual['all_original_and_current_actual']['same_GLM41Base_only_actual_serial_model']
    assert x['cards_returncode'] == 0
    assert not x['frozen_monitor_summary']['source_errors']
    snapshot = note/'Turkish_same_serial_Base_original_failed_NF_CPU_diagnostic_complete_actual.json'
    put(snapshot, actual)
    records = actual['Turkish_serial_v2_and_Base_CPU_diagnostic_complete_records']
    states = {p: r['complete_value'] for p, r in records.items()
              if p.endswith(('state.json', 'completion.json', 'actual_completion.json', 'actual_failure.json', 'progress.json'))}
    diagnostics = {p: r for p, r in records.items() if p.endswith('diagnostic_review.json')}
    for r in diagnostics.values():
        z = r['complete_value']
        assert z['formal_acceptance'] is False and z['qualification'] is False
        assert z['original_fixed_relative_RMSE_limit_unchanged'] == .02
    base_records = x['current_own_GLM41Base_complete_records']
    base_done = {p: r for p, r in base_records.items()
                 if p.endswith(('completion.json', 'actual_completion.json', 'verification_stderr.log'))}
    processes = x['actual_proc_argv_cwd_PPid_startticks_FD_io']
    models = [p for p in processes if 'run.py' in p['argv']]
    nf = [p for p in processes if any(Path(a).name.startswith('verify_') for a in p['argv'])]
    assert len(models) <= 1
    progress = actual['all_original_and_current_actual']['Turkish_current_actual_native_capture_progress']
    forwards = sum(z['total_committed_native_forwards'] for z in progress)
    errors = {p: r for p, r in records.items()
              if p.endswith(('stderr.log', 'verification_stderr.log')) and
              any(t in str(r['complete_value']) for t in ('Traceback (most recent call last)', 'AssertionError:', 'RuntimeError:'))}
    compact = dict(UTC=actual['UTC'], counts=dict(delivered=185, numeric_within=170, numeric_outside=15,
        complete_weights_pending=205, confirmed_missing=310, full_original_qualification=71),
        current_model_id='ytu-ce-cosmos/Turkish-LLaVA-v0.1', current_unique_C_row=690,
        three_D_values=dict(server='.7050', local='.7050', preserved_original='.705'),
        same_Turkish_private_job='TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122',
        same_private_job_SHA256='60b82a03905e82b79e6e0429a0958f85dbfa0c42f2ec54fce67c164d6da355ab',
        current_actual_states=states, genuine_native_forwards=forwards, complete_current_capture_progress=progress,
        current_model_processes=models, current_full_numerical_verifier_processes=nf,
        actual_original_kernel_EX=x['actual_original_lock_owners'], actual_original_disk_free_bytes=x['actual_original_disk_free_bytes'],
        Base_failed_original_completion_and_full_error=base_done, Base_visual_diagnostic_reviews=diagnostics,
        Base_original_fixed_relative_RMSE_limit=.02, Base_failed_NF_not_accepted_or_threshold_relaxed=True,
        current_actual_error_contexts=errors, complete_frozen_monitor_summary=x['frozen_monitor_summary'],
        full_actual_snapshot=str(snapshot), full_actual_snapshot_SHA256=sha(snapshot),
        all_current_primary_sources_count=len(rows), parent_pending=str(parent), parent_pending_SHA256=sha(parent),
        local_original_records=local, canonical_SHA256=ptr['canonical_SHA256'],
        no_new_complete200_G_or_qualification=True, no_deletion=True, no_second_Day22_commit=True)
    facts = note/'facts.json'
    put(facts, compact)
    text = q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT', str(len(rows))).replace('CURRENT_FACTS_SHA256', sha(facts))
    text = text.replace('CURRENT_ACTUAL_JSON', json.dumps(compact, ensure_ascii=False))
    full = note/q['full_note_name']
    full.write_text(text, encoding='utf8')
    pending = C/q['pending_name']
    assert not pending.exists()
    put(pending, dict(UTC=actual['UTC'], parent_pending=str(parent), parent_pending_SHA256=sha(parent), source_count=len(rows),
        all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],
        Code_Day_already_pushed='2026-11-22', Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',
        no_second_same_day_commit=True,
        next_daily_review='Day23 review all explicit useful own Turkish and Base sources, original failed versions, true runtime YAML/protocol/smoke and real queue consumers, tests, Windows builders/readers/official controls, complete dependency diff/outgoing, actual publisher plan_SHA before one normal push'))
    updates = dict(UTC=actual['UTC'], new_primary_pending=str(pending), new_primary_pending_SHA256=sha(pending),
        latest_pending=str(pending), latest_pending_SHA256=sha(pending), current_primary_source_count=len(rows), current_source_count=len(rows),
        latest_late_actual_note=str(full), latest_late_actual_note_SHA256=sha(full), current_full_actual_note=str(full),
        current_full_actual_note_SHA256=sha(full), current_full_facts=str(facts), current_full_facts_SHA256=sha(facts),
        latest_actual_snapshot=str(snapshot), latest_actual_snapshot_SHA256=sha(snapshot),
        current_model_process_active=bool(models), current_full_numerical_verifier_active=bool(nf),
        current_model_NN_or_NF_active=bool(forwards and models or nf), current_actual_model_NN_or_NF_active=bool(forwards and models or nf),
        current_NN_or_NF_active=bool(forwards and models or nf), current_formal_model_task_active=bool(models or nf),
        current_actual_model_process_or_NF_active=bool(models or nf), current_next_original_model='ytu-ce-cosmos/Turkish-LLaVA-v0.1',
        current_next_original_model_source_ready=True, current_next_original_model_stage='Same own genuine3/fullNF serial v2 after exact reviewed Base failure; actual natural completion and capacity override old running state',
        current_Turkish_actual_states=states, current_Turkish_committed_native_forwards=forwards,
        current_Base_failed_original_completion_and_error=base_done, current_Base_CPU_diagnostic_not_acceptance=diagnostics,
        current_unknown_inspection_errors=x['frozen_monitor_summary'].get('inspection_errors', []))
    assert pointer.read_bytes() == pb and closing.read_bytes() == cb
    assert all(sha(n) == h for n, h in ptr['canonical_SHA256'].items())
    assert all(sha(n) == r['SHA256'] for n, r in rows.items())
    co = json.loads(cb)
    co.update(updates)
    put(closing, co)
    ptr.update(updates)
    ptr['closing_addendum_SHA256'] = sha(closing)
    put(pointer, ptr)
    result = dict(passed=True, UTC=actual['UTC'], pointer_SHA256=sha(pointer), closing_SHA256=sha(closing),
        pending=str(pending), pending_SHA256=sha(pending), source_count=len(rows), full_actual_note=str(full),
        full_actual_note_SHA256=sha(full), facts=str(facts), facts_SHA256=sha(facts), actual_snapshot=str(snapshot),
        actual_snapshot_SHA256=sha(snapshot), genuine_Turkish_forwards=forwards, models=models, full_NF=nf,
        current_actual_states=states, current_actual_error_contexts=errors, no_canonical_mutation=True,
        no_active_source_job_config_deadline_signal_mutation=True, no_deletion=True, no_second_Day22_commit=True)
    put(note/'round_preservation_receipt.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()

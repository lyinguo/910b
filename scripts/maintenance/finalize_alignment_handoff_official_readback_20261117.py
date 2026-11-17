"""Close an official-update receipt without changing any evaluation artifacts."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import shutil


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def main(plan_path, expected):
    data = plan_path.read_bytes()
    assert digest(data) == expected
    plan = json.loads(data)
    R = Path(plan['project_root'])
    D = R / 'workspace/diagnostics/batch_910b'
    H = Path(plan['handoff_directory'])
    assert H.resolve().is_relative_to((D / 'heartbeat_handoffs').resolve())
    pointer = D / 'A100_ALIGNMENT_HANDOFF_CURRENT.json'
    pointer_before = pointer.read_bytes()
    assert digest(pointer_before) == plan['pointer_SHA256']
    p = json.loads(pointer_before)
    assert p['handoff_directory'] == str(H)
    for alias in ('handoff', 'handoff_path', 'handoff_txt'):
        assert p[alias] == str(H / 'handoff.txt')
    for alias in ('facts', 'facts_path', 'facts_json'):
        assert p[alias] == str(H / 'facts.json')
    for alias in ('closing_addendum', 'closing_addendum_path'):
        assert p[alias] == str(H / 'closing_addendum.json')
    for alias in ('parent_handoff', 'parent_handoff_path', 'parent_handoff_txt'):
        assert str(Path(p[alias]).parent) == p['parent_handoff_directory']
        assert digest(Path(p[alias]).read_bytes()) == p['parent_handoff_SHA256']
    for alias in ('parent_facts', 'parent_facts_path', 'parent_facts_json'):
        assert str(Path(p[alias]).parent) == p['parent_handoff_directory']
        assert digest(Path(p[alias]).read_bytes()) == p['parent_facts_SHA256']
    assert digest((H / 'handoff.txt').read_bytes()) == p['handoff_SHA256']
    assert digest((H / 'facts.json').read_bytes()) == p['facts_SHA256']
    late = H / 'late_actual_notes/official_readback_source_closure_current_actual_20261117'
    assert not (late / 'actual.json').exists()
    late.mkdir(parents=True, exist_ok=True)
    (late / 'prior_pointer_original.json').write_bytes(pointer_before)
    old_closing = (H / 'closing_addendum.json').read_bytes()
    (late / 'prior_closing_original.json').write_bytes(old_closing)
    assert digest(old_closing) == p['closing_addendum_SHA256']
    previous = Path(p['new_primary_pending'])
    previous_bytes = previous.read_bytes()
    assert digest(previous_bytes) == p['new_primary_pending_SHA256']
    pending = json.loads(previous_bytes)
    sources = {row['path']: dict(row) for row in pending['all_current_primary_source_and_true_consumer_configs']}
    for row in sources.values():
        assert digest(Path(row['path']).read_bytes()) == row['SHA256'], row['path']
    for source in plan['new_primary_sources']:
        path = (R / source['relative']).resolve()
        assert path.is_relative_to(R.resolve()) and source['relative'].startswith('scripts/')
        raw = base64.b64decode(source['base64'], validate=True)
        assert digest(raw) == source['SHA256']
        ast.parse(raw.decode('utf-8-sig'), filename=str(path))
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        snap = late / 'primary_source_snapshots' / source['relative']
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(raw)
        sources[str(path)] = dict(path=str(path), relative=source['relative'], SHA256=digest(raw), bytes=len(raw), snapshot=str(snap), role=source['role'])
    originals = []
    for item in plan['local_original_artifacts']:
        raw = base64.b64decode(item['base64'], validate=True)
        assert digest(raw) == item['SHA256']
        target = late / 'local_original_packets' / item['archive_name']
        assert target.resolve().is_relative_to(late.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        originals.append({k: v for k, v in item.items() if k != 'base64'} | {'snapshot': str(target)})
    sys.path[:0] = [str(R / 'scripts'), str(R)]
    monitor_path = R / 'scripts/A100_alignment_incremental_monitor_20261103.py'
    assert digest(monitor_path.read_bytes()) == '2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
    import A100_alignment_incremental_monitor_20261103 as m
    with contextlib.redirect_stdout(io.StringIO()):
        m.main()
    latest = json.loads((D / 'incremental_monitor/latest.json').read_bytes())
    checkpoint = json.loads((D / 'incremental_monitor/checkpoint.json').read_bytes())
    save(late / 'frozen_monitor_complete_latest.json', latest)
    save(late / 'frozen_monitor_complete_checkpoint.json', checkpoint)
    actual = dict(
        UTC=datetime.now(timezone.utc).isoformat(),
        actual_project_processes=m.project_processes(), actual_original_kernel_FLOCK=m.runner_lock_owners(),
        actual_free_bytes=shutil.disk_usage(R).free, current_canonical_counts=p['canonical_counts'],
        all_monitor_source_errors=latest['source_errors'], all_monitor_inspection_errors=latest['inspection_errors'],
        all_monitor_new_log_error_candidates=latest['new_log_error_candidates'], work_counters=latest['work_counters'],
        new_or_modified_source_config_paths=latest['new_or_modified_source_config_paths'],
        npu_smi=latest['npu_smi'], actual_stoppage_first_proven_BJT='2026-11-17T16:58:07.780211+08:00',
        current_formal_NN_or_NF_active=latest['inference_or_verification_active'],
        official_update_readback=plan['official_readback'],
        local_original_packets=originals, no_new_evaluation_or_matrix_deletion_in_metadata_closure=True,
        all_previous_source_SHA_rechecked_no_weight_arrays_or_old_CPU200_rerun=True,
    )
    assert actual['all_monitor_source_errors'] == []
    assert actual['all_monitor_inspection_errors'] == []
    assert actual['all_monitor_new_log_error_candidates'] == []
    canonical = {}
    for path, sha in p['canonical_SHA256'].items():
        canonical[path] = digest(Path(path).read_bytes())
        assert canonical[path] == sha, path
    actual['canonical_SHA256'] = canonical
    actual['handoff_pointer_all_path_aliases_and_parent_SHA_consistent'] = True
    new_pending = Path(plan['new_pending'])
    assert new_pending.parent == previous.parent and not new_pending.exists()
    pending.update(UTC=actual['UTC'], parent_pending=str(previous), parent_pending_SHA256=digest(previous_bytes),
                   all_current_primary_source_and_true_consumer_configs=sorted(sources.values(), key=lambda row: row['path']),
                   source_count=len(sources), official_update_maintenance_source_closure=True)
    save(new_pending, pending)
    actual.update(new_primary_pending=str(new_pending), new_primary_pending_SHA256=digest(new_pending.read_bytes()), current_primary_source_count=len(sources))
    save(late / 'actual.json', actual)
    closing = json.loads(old_closing)
    closing.update(UTC=actual['UTC'], latest_actual_note=str(late / 'actual.json'), latest_actual_note_SHA256=digest((late / 'actual.json').read_bytes()),
                   latest_pending=str(new_pending), latest_pending_SHA256=actual['new_primary_pending_SHA256'],
                   current_actual_model_NN_or_NF_active=actual['current_formal_NN_or_NF_active'],
                   official_readback=plan['official_readback'], current_source_count=len(sources),
                   prior_initial_current_snapshot_preserved=True)
    save(H / 'closing_addendum.json', closing)
    for alias in ('latest_actual_note', 'latest_actual_snapshot'):
        p[alias] = str(late / 'actual.json')
    for alias in ('latest_actual_note_SHA256', 'latest_actual_snapshot_SHA256'):
        p[alias] = digest((late / 'actual.json').read_bytes())
    p.update(UTC=actual['UTC'], closing_addendum_SHA256=digest((H / 'closing_addendum.json').read_bytes()),
             new_primary_pending=str(new_pending), new_primary_pending_SHA256=actual['new_primary_pending_SHA256'],
             current_primary_source_count=len(sources), current_actual_model_NN_or_NF_active=actual['current_formal_NN_or_NF_active'],
             latest_official_automation_readback_SHA256=plan['official_readback_SHA256'])
    assert pointer.read_bytes() == pointer_before
    temporary = pointer.with_suffix('.official_closure.tmp')
    assert not temporary.exists()
    save(temporary, p)
    os.replace(temporary, pointer)
    return {key: actual[key] for key in ('UTC', 'actual_project_processes', 'actual_original_kernel_FLOCK', 'actual_free_bytes', 'current_formal_NN_or_NF_active', 'new_primary_pending', 'new_primary_pending_SHA256', 'current_primary_source_count')} | {
        'passed': True, 'handoff_directory': str(H), 'closing_addendum_SHA256': p['closing_addendum_SHA256'],
        'latest_actual_note': str(late / 'actual.json'), 'latest_actual_note_SHA256': p['latest_actual_note_SHA256'],
        'pointer_SHA256': digest(pointer.read_bytes()), 'all_path_aliases_parent_SHA_and_canonical_SHA_consistent': True,
        'monitor_source_and_log_checks_clear': True,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.plan, args.plan_sha256), ensure_ascii=False))

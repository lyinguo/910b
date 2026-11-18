"""Archive explicitly reviewed sources and receipts, then atomically advance a handoff.

All content and model-specific facts are supplied by a separately hashed plan. This
utility never edits a workbook, runner, model configuration, report, or old proof.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import base64
import hashlib
import json
import os
import shutil


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def inside(path, root):
    path = path.resolve()
    assert path.is_relative_to(root.resolve()), str(path)
    return path


def execute(plan_path, expected_sha):
    raw_plan = plan_path.read_bytes()
    assert sha(raw_plan) == expected_sha
    plan = json.loads(raw_plan)
    root = Path(plan['project_root']).resolve()
    diagnostics = root / 'workspace/diagnostics/batch_910b'
    home = inside(Path(plan['handoff_directory']), diagnostics / 'heartbeat_handoffs')
    assert home.is_dir() and not (home / 'facts.json').exists()
    pointer = diagnostics / 'A100_ALIGNMENT_HANDOFF_CURRENT.json'
    old_pointer = pointer.read_bytes()
    assert sha(old_pointer) == plan['expected_pointer_SHA256']
    for path, expected in plan['canonical_SHA256'].items():
        assert sha(Path(path).read_bytes()) == expected, path
    parent = Path(plan['parent_handoff_directory'])
    for name, digest in plan['parent_files_SHA256'].items():
        assert sha((parent / name).read_bytes()) == digest, str(parent / name)
    old_pending_path = Path(plan['parent_pending'])
    pending_raw = old_pending_path.read_bytes()
    assert sha(pending_raw) == plan['parent_pending_SHA256']
    old_pending = json.loads(pending_raw)
    old_sources = old_pending['all_current_primary_source_and_true_consumer_configs']
    rows = {entry['path']: dict(entry) for entry in old_sources}
    assert len(rows) == len(old_sources)
    transition_receipt = Path(plan['reviewed_daily_transition_receipt'])
    assert sha(transition_receipt.read_bytes()) == plan['reviewed_daily_transition_receipt_SHA256']
    receipt = json.loads(transition_receipt.read_bytes())
    assert '02df1887796f49c830c63db01f4dcdb2bb8e4298' in json.dumps(receipt)
    exact_transitions = {
        str(root / 'scripts/publish_code_daily.py'): (
            '1cdd6adba1eebc601af4cdbd33075998d3cdc825302cfb79cc35e2804c2fe93f',
            'e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'),
        str(root / 'configs/publication/code_only.json'): (
            '0306b35125114be45659c919e1c576b98468dd7b86441df77d5d7e271bb93494',
            'a78329517e49e93433f428c3d2e68908c13a77a5d39eba080d4bb82d40df0151'),
    }
    transitions_used = []
    for entry in rows.values():
        actual_sha = sha(Path(entry['path']).read_bytes())
        if actual_sha != entry['SHA256']:
            pair = exact_transitions.get(entry['path'])
            assert pair == (entry['SHA256'], actual_sha), entry['path']
            transitions_used.append(dict(path=entry['path'], before_SHA256=entry['SHA256'], after_SHA256=actual_sha,
                reviewed_daily_receipt=str(transition_receipt), reviewed_daily_receipt_SHA256=sha(transition_receipt.read_bytes()),
                no_model_math_or_frozen_runner_change=True))
            entry['SHA256'] = actual_sha
    for item in plan['install_primary_sources']:
        path = inside(root / item['relative'], root)
        assert item['relative'].startswith(('scripts/', 'universal_eval/', 'configs/'))
        assert path.suffix.lower() in ('.py', '.mjs', '.json', '.yaml', '.yml')
        data = base64.b64decode(item['base64'], validate=True)
        assert sha(data) == item['SHA256']
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=str(path))
        if path.exists():
            assert path.read_bytes() == data, str(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        rows[str(path)] = {'path': str(path), 'relative': item['relative'], 'SHA256': sha(data), 'bytes': len(data), 'role': item['role']}
    for item in plan['additional_existing_primary_sources']:
        path = inside(root / item['relative'], root)
        data = path.read_bytes()
        if item.get('SHA256') is not None:
            assert sha(data) == item['SHA256'], str(path)
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=str(path))
        rows[str(path)] = {'path': str(path), 'relative': item['relative'], 'SHA256': sha(data), 'bytes': len(data), 'role': item['role']}
    source_rows = []
    for path, entry in sorted(rows.items()):
        data = Path(path).read_bytes()
        assert sha(data) == entry['SHA256'], path
        relative = Path(path).relative_to(root)
        snapshot = home / 'primary_source_snapshots' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
        assert sha(snapshot.read_bytes()) == entry['SHA256']
        entry.update(relative=str(relative), snapshot=str(snapshot), bytes=len(data))
        source_rows.append(entry)
    for artifact in plan['local_original_artifacts']:
        if artifact.get('base64') is not None:
            data = base64.b64decode(artifact['base64'], validate=True)
            assert sha(data) == artifact['SHA256']
            target = inside(home / 'local_original_packets' / artifact['archive_name'], home)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            artifact['snapshot'] = str(target)
        artifact.pop('base64', None)
    preserved = {}
    for name, source in plan['static_original_files'].items():
        path = Path(source['path'])
        data = path.read_bytes()
        if source.get('SHA256'):
            assert sha(data) == source['SHA256'], str(path)
        target = inside(home / 'static_original_evidence' / name, home)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        preserved[name] = {'path': str(path), 'SHA256': sha(data), 'bytes': len(data), 'snapshot': str(target)}
    (home / 'prior_current_pointer_original.json').write_bytes(old_pointer)
    (home / 'parent_pending_original.json').write_bytes(pending_raw)
    body = plan['handoff_text'].replace('\r\n', '\n')
    (home / 'handoff.txt').write_text(body, encoding='utf-8', newline='\n')
    new_pending_path = inside(Path(plan['new_pending']), diagnostics / 'code_publication_source_layout_20261114')
    assert not new_pending_path.exists()
    pending = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'reviewed_daily_source_transitions': transitions_used,
        'parent_pending': str(old_pending_path), 'parent_pending_SHA256': sha(pending_raw),
        'Code_Day_already_pushed': plan['Code_Day_already_pushed'],
        'no_second_same_day_commit': True,
        'all_current_primary_source_and_true_consumer_configs': source_rows,
        'source_count': len(source_rows),
        'next_daily_review': plan['next_daily_review'],
        'only_source_or_actual_consumer_configuration_not_run_artifacts': True,
        'handoff_directory': str(home),
    }
    save_json(new_pending_path, pending)
    facts = dict(plan['facts'])
    facts.update({
        'UTC': datetime.now(timezone.utc).isoformat(),
        'parent_handoff_directory': str(parent),
        'parent_files_SHA256': plan['parent_files_SHA256'],
        'canonical_SHA256': plan['canonical_SHA256'],
        'current_primary_source_count': len(source_rows),
        'new_primary_pending': str(new_pending_path),
        'new_primary_pending_SHA256': sha(new_pending_path.read_bytes()),
        'all_primary_original_byte_snapshots': source_rows,
        'reviewed_daily_source_transitions': transitions_used,
        'actual_local_original_packets': plan['local_original_artifacts'],
        'static_original_evidence': preserved,
        'all_ancestral_norms_failures_families_exact_pruning_exceptions_user_decisions_inherited': True,
    })
    save_json(home / 'facts.json', facts)
    closing = dict(plan['closing'])
    closing.update(UTC=datetime.now(timezone.utc).isoformat(), latest_pending=str(new_pending_path), latest_pending_SHA256=sha(new_pending_path.read_bytes()))
    save_json(home / 'closing_addendum.json', closing)
    updated = json.loads(old_pointer)
    updated.update(plan['pointer_updates'])
    updated['UTC'] = datetime.now(timezone.utc).isoformat()
    updated['handoff_directory'] = str(home)
    updated['parent_handoff_directory'] = str(parent)
    for key in ('handoff', 'handoff_path', 'handoff_txt'):
        updated[key] = str(home / 'handoff.txt')
    for key in ('handoff_SHA256', 'handoff_sha256', 'handoff_txt_SHA256', 'handoff_text_SHA256'):
        updated[key] = sha((home / 'handoff.txt').read_bytes())
    for key in ('facts', 'facts_path', 'facts_json'):
        updated[key] = str(home / 'facts.json')
    for key in ('facts_SHA256', 'facts_sha256', 'facts_json_SHA256'):
        updated[key] = sha((home / 'facts.json').read_bytes())
    for key in ('closing_addendum', 'closing_addendum_path'):
        updated[key] = str(home / 'closing_addendum.json')
    updated['closing_addendum_SHA256'] = sha((home / 'closing_addendum.json').read_bytes())
    for key in ('parent_handoff', 'parent_handoff_path', 'parent_handoff_txt'):
        updated[key] = str(parent / 'handoff.txt')
    for key in ('parent_facts', 'parent_facts_path', 'parent_facts_json'):
        updated[key] = str(parent / 'facts.json')
    updated['parent_handoff_SHA256'] = plan['parent_files_SHA256']['handoff.txt']
    updated['parent_facts_SHA256'] = plan['parent_files_SHA256']['facts.json']
    updated['canonical_SHA256'] = plan['canonical_SHA256']
    updated['new_primary_pending'] = str(new_pending_path)
    updated['new_primary_pending_SHA256'] = sha(new_pending_path.read_bytes())
    updated['current_primary_source_count'] = len(source_rows)
    for key in ('latest_actual_note', 'latest_actual_snapshot'):
        updated[key] = str(home / 'closing_addendum.json')
    for key in ('latest_actual_note_SHA256', 'latest_actual_snapshot_SHA256'):
        updated[key] = sha((home / 'closing_addendum.json').read_bytes())
    for path, expected in plan['canonical_SHA256'].items():
        assert sha(Path(path).read_bytes()) == expected, path
    for entry in source_rows:
        assert sha(Path(entry['path']).read_bytes()) == entry['SHA256'], entry['path']
    assert pointer.read_bytes() == old_pointer
    temporary = pointer.with_suffix('.verified_handoff.tmp')
    assert not temporary.exists()
    save_json(temporary, updated)
    os.replace(temporary, pointer)
    result = {key: updated[key] for key in ('handoff_directory', 'handoff_SHA256', 'facts_SHA256', 'closing_addendum_SHA256', 'new_primary_pending', 'new_primary_pending_SHA256', 'current_primary_source_count')}
    result.update(passed=True, pointer_SHA256=sha(pointer.read_bytes()), old_pointer_SHA256=sha(old_pointer), canonical_files_unchanged=True)
    save_json(home / 'archive_actual_receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan, args.plan_sha256), ensure_ascii=False))

from pathlib import Path
import ast
import hashlib
import json

P = Path('D:/code-lv/.tmp/batch_910b_20261030')
old = (P / 'archive_alignment_primary_handoff_snapshot_20261117.py').read_text(encoding='utf-8')
needle = "    for entry in rows.values():\n        assert sha(Path(entry['path']).read_bytes()) == entry['SHA256'], entry['path']\n"
replacement = '''    transition_receipt = Path(plan['reviewed_daily_transition_receipt'])
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
'''
assert old.count(needle) == 1
new = old.replace(needle, replacement)
new = new.replace("        'UTC': datetime.now(timezone.utc).isoformat(),\n        'parent_pending'", "        'UTC': datetime.now(timezone.utc).isoformat(),\n        'reviewed_daily_source_transitions': transitions_used,\n        'parent_pending'")
new = new.replace("        'all_primary_original_byte_snapshots': source_rows,", "        'all_primary_original_byte_snapshots': source_rows,\n        'reviewed_daily_source_transitions': transitions_used,")
ast.parse(new)
path = P / 'archive_alignment_reviewed_daily_transition_primary_handoff_20261118.py'
assert not path.exists()
path.write_bytes(new.encode('utf-8'))
print(json.dumps(dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), changed_scope='Only two exact reviewed publisher-policy source transitions; all model/frozen source SHA guards retained')))

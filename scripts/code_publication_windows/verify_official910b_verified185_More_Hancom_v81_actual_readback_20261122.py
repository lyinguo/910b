"""Read official heartbeat bytes and require all original nonprompt settings."""
from pathlib import Path
import hashlib, json, tomllib
V = Path(__file__).resolve().parent
path = Path('C:/Users/lyg/.codex/automations/910b/automation.toml')
before = (V / 'official_910b_before_verified185_More_Hancom_v81_update_20261122.toml').read_bytes()
old = tomllib.loads(before.decode('utf8'))
requested = json.loads((V / 'official_910b_verified185_More_Hancom_v81_requested_args_20261122.json').read_bytes())
raw = path.read_bytes()
now = tomllib.loads(raw.decode('utf8'))
actual = now['prompt']
exact = actual == requested['prompt']
one_LF = requested['prompt'].endswith('\n') and actual == requested['prompt'][:-1]
assert exact or one_LF, 'Official actual prompt differs beyond one previously proved terminal LF'
assert now['id'] == requested['id'] == '910b'
assert now['name'] == old['name'] == requested['name']
assert now['status'] == old['status'] == requested['status'] == 'ACTIVE'
assert now['kind'] == old['kind'] == requested['kind'] == 'heartbeat'
assert now['rrule'] == old['rrule'] == requested['rrule'] == 'RRULE:FREQ=MINUTELY;INTERVAL=300'
assert now['target_thread_id'] == old['target_thread_id'] == requested['targetThreadId']
ignored = {'prompt', 'model_prompt_sha256', 'updated_at'}
assert {k: v for k, v in old.items() if k not in ignored} == {k: v for k, v in now.items() if k not in ignored}
h = hashlib.sha256(actual.encode()).hexdigest()
assert now['model_prompt_sha256'] == h
book = Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
stat = book.stat()
bh = hashlib.sha256(book.read_bytes()).hexdigest()
assert bh == '2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
(V / 'official_910b_verified185_More_Hancom_v81_actual_after_update_20261122.toml').write_bytes(raw)
r = dict(passed=True, id=now['id'], status=now['status'], name=now['name'], rrule=now['rrule'],
    target_thread_id=now['target_thread_id'], all_original_nonprompt_fields_unchanged=True,
    requested_prompt_SHA256=hashlib.sha256(requested['prompt'].encode()).hexdigest(),
    prompt_SHA256=h, prompt_chars=len(actual), prompt_exact_requested_bytes=exact,
    official_normalization_exactly_one_trailing_LF=one_LF,
    previous_TOML_SHA256=hashlib.sha256(before).hexdigest(), actual_TOML_SHA256=hashlib.sha256(raw).hexdigest(),
    local_workbook_SHA256=bh, local_workbook_stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns),
    no_active_TOML_direct_write=True, no_new_automation=True)
(V / 'official_910b_verified185_More_Hancom_v81_actual_readback_20261122.json').write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(r, ensure_ascii=False))

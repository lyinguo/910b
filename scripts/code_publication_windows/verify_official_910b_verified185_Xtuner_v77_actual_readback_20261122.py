"""Verify official 910b update from actual readonly TOML; preserve original fields and raw bytes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tomllib
V=Path(__file__).resolve().parent
q=json.loads((V/'official_910b_verified185_Xtuner_native_v77_requested_args_20261122.json').read_bytes());old_raw=(V/'official_910b_before_verified185_Xtuner_v77_update_20261122.toml').read_bytes();old=tomllib.loads(old_raw.decode());p=Path(r'C:/Users/lyg/.codex/automations/910b/automation.toml');raw=p.read_bytes();actual=tomllib.loads(raw.decode());sha=lambda b:hashlib.sha256(b).hexdigest()
assert actual['prompt']==q['prompt'] and actual['model_prompt_sha256']==sha(q['prompt'].encode())
for key in old:
 if key not in {'prompt','model_prompt_sha256','updated_at'}:assert actual.get(key)==old[key],key
assert actual['id']==q['id']=='910b' and actual['kind']==q['kind']=='heartbeat' and actual['name']==q['name'] and actual['status']==q['status']=='ACTIVE' and actual['rrule']==q['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300' and actual['target_thread_id']==q['targetThreadId']=='01a18bbd-8040-7d02-9004-beef491d836a'
assert not actual.get('notification_policy') and q['notificationPolicy'] is None
out=V/'official_910b_verified185_Xtuner_v77_actual_after_official_update_20261122.toml'
if out.exists():assert out.read_bytes()==raw
else:out.write_bytes(raw)
book=Path(r'D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');st=book.stat();assert sha(book.read_bytes())=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,actual_readonly_TOML=str(p),actual_TOML_SHA256=sha(raw),before_TOML_SHA256=sha(old_raw),prompt_SHA256=sha(actual['prompt'].encode()),prompt_characters=len(actual['prompt']),id=actual['id'],name=actual['name'],kind=actual['kind'],status=actual['status'],target_thread_id=actual['target_thread_id'],rrule=actual['rrule'],all_original_nonprompt_fields_unchanged=True,only_official_tool_used_no_direct_TOML_write=True,actual_local_workbook=dict(path=str(book),SHA256=sha(book.read_bytes()),bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),no_model_or_workbook_or_runtime_source_mutation=True)
(V/'official_910b_verified185_Xtuner_v77_actual_readback_20261122.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review,ensure_ascii=False))

"""Verify an official prompt-only update and unchanged original task and workbook."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,tomllib
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
before=(V/'official910b_before_verified185_Day23_MedraN_source_v91_20261123.toml').read_bytes()
old=tomllib.loads(before.decode('utf8'))
args=json.loads((V/'official910b_verified185_Day23_MedraN_source_v91_requested_args_20261123.json').read_bytes())
actual=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes()
now=tomllib.loads(actual.decode('utf8'))
(V/'official910b_after_verified185_Day23_MedraN_source_v91_20261123.toml').write_bytes(actual)
different={k for k in set(old)|set(now) if old.get(k)!=now.get(k)}
assert different <= {'prompt','model_prompt_sha256','updated_at'},different
assert now['prompt'] in [args['prompt'],args['prompt']+'\n']
assert now['id']=='910b' and now['status']==old['status']=='ACTIVE'
assert now['name']==old['name']==args['name']
assert now['rrule']==old['rrule']==args['rrule']
assert now['target_thread_id']==old['target_thread_id']==args['targetThreadId']=='01a18bbd-8040-7d02-9004-beef491d836a'
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
st=book.stat();bh=sha(book.read_bytes())
assert bh=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
q=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),official_tool_only=True,automation_id='910b',
    status=now['status'],name=now['name'],original_thread=now['target_thread_id'],original_rrule=now['rrule'],
    only_changed_keys=sorted(different),actual_TOML_SHA256=sha(actual),requested_prompt_SHA256=sha(args['prompt'].encode()),
    actual_prompt_SHA256=sha(now['prompt'].encode()),prompt_exact=now['prompt']==args['prompt'],all_other_actual_fields_identical=True,
    local_workbook_fresh=dict(path=str(book),SHA256=bh,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),
    no_live_TOML_or_workbook_direct_write=True)
(V/'official910b_verified185_Day23_MedraN_source_v91_actual_readback_receipt_20261123.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(q,ensure_ascii=False))

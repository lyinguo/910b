"""Read back official-only prompt update and unchanged actual original heartbeat fields."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tomllib
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
before=(V/'official910b_before_verified185_GLM41Base_serial_v86_20261122.toml').read_bytes();old=tomllib.loads(before.decode('utf8'));args=json.loads((V/'official910b_verified185_GLM41Base_serial_v86_requested_args_20261122.json').read_bytes());actual=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes();now=tomllib.loads(actual.decode('utf8'));(V/'official910b_after_verified185_GLM41Base_serial_v86_20261122.toml').write_bytes(actual)
keys=set(old)|set(now);differences={k:dict(before=old.get(k),after=now.get(k)) for k in keys if old.get(k)!=now.get(k)};assert set(differences)<=set(['prompt','model_prompt_sha256','updated_at']),list(differences)
exact=now['prompt']==args['prompt'];one_terminal_lf=now['prompt']==args['prompt']+'\n';assert exact or one_terminal_lf
assert now['id']=='910b' and now['status']=='ACTIVE' and now['name']==old['name']==args['name'] and now['rrule']==old['rrule']==args['rrule'] and now['target_thread_id']==old['target_thread_id']==args['targetThreadId']=='01a18bbd-8040-7d02-9004-beef491d836a'
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');s=book.stat();bh=sha(book.read_bytes());assert bh=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),official_tool_only=True,automation_id='910b',status=now['status'],name=now['name'],original_thread=now['target_thread_id'],original_rrule=now['rrule'],only_changed_keys=sorted(differences),actual_TOML_SHA256=sha(actual),requested_prompt_SHA256=sha(args['prompt'].encode()),actual_prompt_SHA256=sha(now['prompt'].encode()),prompt_exact=exact,only_one_terminal_LF_added_by_official_tool=one_terminal_lf,all_other_actual_fields_identical=True,local_workbook_fresh=dict(path=str(book),SHA256=bh,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns),no_workbook_or_live_TOML_direct_write=True)
(V/'official910b_verified185_GLM41Base_serial_v86_actual_readback_receipt_20261122.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))

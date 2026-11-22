"""Read back the official original heartbeat; never edit its live TOML."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tomllib
V=Path(__file__).resolve().parent
args=json.loads((V/'official_910b_verified184_Llava_completed_Gemma4MLX8_serial_v74_requested_args_20261122.json').read_bytes())
raw=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes();actual=tomllib.loads(raw.decode('utf-8'))
assert actual['id']==args['id']=='910b' and actual['kind']==args['kind']=='heartbeat' and actual['status']==args['status']=='ACTIVE'
assert actual['name']==args['name'] and actual['target_thread_id']==args['targetThreadId']=='01a18bbd-8040-7d02-9004-beef491d836a' and actual['rrule']==args['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300'
assert actual['prompt']==args['prompt']
digest=hashlib.sha256(actual['prompt'].encode()).hexdigest();assert actual['model_prompt_sha256']==digest=='2e57580f8f82430b866933bc99e60445302ea5ccdfcca84086e24bc05ff67b80'
dest=V/'official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.toml'
with dest.open('xb') as f:f.write(raw)
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');st=book.stat()
with book.open('rb') as f:bookSHA=hashlib.file_digest(f,'sha256').hexdigest()
assert bookSHA=='0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1'
receipt=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),id='910b',kind='heartbeat',name=actual['name'],status='ACTIVE',target_thread_id=actual['target_thread_id'],rrule=actual['rrule'],actual_TOML_SHA256=hashlib.sha256(raw).hexdigest(),actual_prompt_SHA256=digest,actual_prompt_characters=len(actual['prompt']),full_prompt_TOML_actual_bytes_preserved=str(dest),official_tool_only_no_live_TOML_direct_edit=True,local_workbook_fresh_SHA256=bookSHA,local_workbook_fresh_stat=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),changes0_no_workbook_write=True)
(V/'official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))

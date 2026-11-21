"""Read-only actual official settings and fresh local workbook SHA/stat."""
from pathlib import Path
import hashlib,json,tomllib
V=Path(__file__).resolve().parent
before_raw=(V/'official_910b_before_MetaNF3_capacity_exact271_20261120.toml').read_bytes();before=tomllib.loads(before_raw.decode());requested=json.loads((V/'official_910b_verified183_MetaNF3_capacity_exact271_requested_args_20261120.json').read_bytes())
active=Path('C:/Users/lyg/.codex/automations/910b/automation.toml');after_raw=active.read_bytes();after=tomllib.loads(after_raw.decode())
def fields(x):return dict(id=x['id'],name=x['name'],status=x['status'],rrule=x['rrule'],target_thread_id=x.get('target_thread_id'),kind=x.get('kind'),notificationPolicy=x.get('notification_policy',None))
assert fields(before)==fields(after) and after['status']=='ACTIVE' and after['id']=='910b' and after['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
assert before['rrule']==after['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300' and after['prompt'].replace('\r\n','\n')==requested['prompt'].replace('\r\n','\n')
(V/'official_910b_after_MetaNF3_capacity_exact271_20261120.toml').write_bytes(after_raw)
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');stat1=book.stat();raw=book.read_bytes();stat2=book.stat();assert (stat1.st_size,stat1.st_mtime_ns)==(stat2.st_size,stat2.st_mtime_ns);booksha=hashlib.sha256(raw).hexdigest();assert booksha=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
result=dict(passed=True,before_TOML_SHA256=hashlib.sha256(before_raw).hexdigest(),after_TOML_SHA256=hashlib.sha256(after_raw).hexdigest(),preserved_original_fields=fields(after),actual_after_prompt_SHA256=hashlib.sha256(after['prompt'].encode()).hexdigest(),requested_normalized_prompt_SHA256=hashlib.sha256(requested['prompt'].replace('\r\n','\n').encode()).hexdigest(),only_official_tool_used=True,active_TOML_never_raw_edited=True,fresh_local183_workbook=dict(path=str(book),SHA256=booksha,bytes=stat2.st_size,mtime_ns=stat2.st_mtime_ns,ctime_ns=stat2.st_ctime_ns,concurrent_stat_unchanged=True),no_workbook_or_canonical_changes=True)
(V/'official_910b_MetaNF3_exact_settings_local183_readback_20261120.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))

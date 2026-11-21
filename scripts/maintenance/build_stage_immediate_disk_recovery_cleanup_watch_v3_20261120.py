from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;R='/root/projects/comperror';old=(V/'periodic_batch_910b_exact_storage_maintenance_20261120.py').read_text(encoding='utf-8')
n='periodic_batch_910b_exact_storage_maintenance_v3_immediate_budget_20261120.py';s=old.replace('batch_910b_periodic_exact_storage_retention_v2_20261120.json','batch_910b_immediate_disk_recovery_cleanup_v3_20261120.json')
needle="free=shutil.disk_usage(R).free;idle=quiet();base=";assert s.count(needle)==1;s=s.replace(needle,"required=int(sys.argv[sys.argv.index('--required-free-bytes')+1]) if '--required-free-bytes' in sys.argv else 0;assert 0<=required<=32*2**40;target_free=max(cfg['proactive_free_target_bytes'],required)\n free=shutil.disk_usage(R).free;idle=quiet();base=")
s=s.replace("target_free_bytes=cfg['proactive_free_target_bytes']","target_free_bytes=target_free").replace("free>=cfg['proactive_free_target_bytes']","free>=target_free").replace("target=max(0,cfg['proactive_free_target_bytes']-shutil.disk_usage(R).free)","target=max(0,target_free-shutil.disk_usage(R).free)")
ast.parse(s);(V/n).write_text(s,encoding='utf-8',newline='\n');watch='watch_batch_910b_immediate_disk_recovery_cleanup_v1_20261120.py';mapping={n:'scripts/maintenance/'+n,watch:'scripts/maintenance/'+watch};items={};hashes={}
for file,rel in mapping.items():
 raw=(V/file).read_bytes();ast.parse(raw);digest=hashlib.sha256(raw).hexdigest();hashes[rel]=digest;items[R+'/'+rel]=dict(SHA256=digest,base64=base64.b64encode(raw).decode())
cfg=json.loads((V/'batch_910b_periodic_exact_storage_retention_v2_20261120.json').read_text(encoding='utf-8'));cfg.update(user_additional_instruction='再阻塞的时候抓紧时间将那些中间结果进行清理，不要再等了，这个任务已经很久了',poll_seconds=20,active_state_max_age_seconds=120,no_new_values_retry_seconds=300,urgent_policy_consumer=mapping[n],react_immediately_when_actual_capacity_blocked_and_safe=True,no_wait_for_next_app_heartbeat=True,preserve_original_app_heartbeat_300min=True);cfg['exact_source_SHA256'].update(hashes)
raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();rel='configs/maintenance/batch_910b_immediate_disk_recovery_cleanup_v3_20261120.json';items[R+'/'+rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode());(V/Path(rel).name).write_bytes(raw)
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');items=json.loads(base64.b64decode('PAYLOAD'))
for n,x in items.items():
 b=base64.b64decode(x['base64']);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not Path(n).exists()
for n,x in items.items():
 p=Path(n);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(x['base64']))
H=R/'workspace/diagnostics/batch_910b/immediate_disk_recovery_cleanup_watch_20261120';assert not H.exists();script=R/'scripts/maintenance/watch_batch_910b_immediate_disk_recovery_cleanup_v1_20261120.py'
p=subprocess.run([sys.executable,'-B',str(script),'--once'],cwd=R,capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr);check=json.loads(p.stdout);assert check['phase']=='formal_NN_NF_or_original_EX_active_no_cleanup' and check['actual_NEW_model_forwards']==0
(H/'first_actual_noninterrupting_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\\n')
argv=[sys.executable,'-B',str(script)]
with (H/'watch_stdout.log').open('xb') as out,(H/'watch_stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
x=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,exact_source_SHA256={n:x['SHA256'] for n,x in items.items()},actual_first_check=check,no_model_interruption_or_canonical_write=True,no_new_app_automation_or_thread=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(x,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode());ast.parse(s);(V/'stage_immediate_disk_recovery_cleanup_watch_v3_20261120_remote.py').write_text(s,encoding='utf-8',newline='\n');print(json.dumps({p:x['SHA256'] for p,x in items.items()},indent=2))

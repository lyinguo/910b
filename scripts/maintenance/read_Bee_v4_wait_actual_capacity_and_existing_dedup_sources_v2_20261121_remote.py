import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
q=json.loads((R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v4_wait_actual_capacity_serial_20261121.json').read_bytes());W=Path(q['wait_home'])
out={'wait_files':{p.name:json.loads(p.read_bytes()) for p in W.glob('*.json')},'free_bytes':os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,'continuation_created':Path(q['home']).exists(),'J200_created':Path(q['new_job']).exists()}
names=[str(p) for p in (R/'scripts').rglob('*.py')]
out['dedup_source_names']=[n for n in names if any(t in n.lower() for t in ['dedup','hardlink','lossless_share'])]
du=subprocess.run(['du','-x','--max-depth=1',str(D)],capture_output=True,text=True,timeout=25)
out['batch_du']=du.stdout;out['batch_du_error']=du.stderr
for p in sorted((R/'scripts/maintenance').glob('*watch_batch_910b*')):
 out.setdefault('watcher_sources',[]).append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()})
print(json.dumps(out,ensure_ascii=False))

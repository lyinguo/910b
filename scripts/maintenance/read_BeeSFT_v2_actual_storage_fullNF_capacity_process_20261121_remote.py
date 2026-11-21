import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121';out={'UTC':datetime.now(timezone.utc).isoformat(),'free_bytes':shutil.disk_usage(R).free,'files':{},'file_stats':[],'processes':[]}
for p in H.iterdir():
 if not p.is_file():continue
 out['file_stats'].append({'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
 if p.suffix in ['.json','.log'] and p.name not in ['complete_storage_all_values_review.json']:
  b=p.read_bytes();v={'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
  try:v['JSON']=json.loads(b)
  except Exception:pass
  out['files'][p.name]=v
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if any(x in ' '.join(argv).lower() for x in ['continue_beesft','bee-8b-sft','verify_bee8bsft']):out['processes'].append({'PID':int(p.name),'argv':argv,'cwd':os.readlink(p/'cwd'),'stat':(p/'stat').read_text(),'io':(p/'io').read_text()})
 except OSError:pass
out['kernel_locks']=Path('/proc/locks').read_text();print(json.dumps(out,ensure_ascii=False))
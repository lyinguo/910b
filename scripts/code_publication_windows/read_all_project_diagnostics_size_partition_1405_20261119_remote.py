import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import os,stat,json,subprocess,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics';batch=D/'batch_910b'
q=subprocess.run(['du','-x','-B1','--max-depth=1',str(D)],capture_output=True,text=True,timeout=180);assert q.returncode==0,q.stderr
top=sorted([(int(s),p) for s,p in [v.split('\t',1) for v in q.stdout.splitlines()]],reverse=True)
types=defaultdict(lambda:dict(files=0,logical_bytes=0,allocated_bytes=0));tops={};large=[];errs=[];seen=set();dev=D.stat().st_dev
for size,name in top[1:31]:
 p=Path(name);x=subprocess.run(['du','-x','-B1','--max-depth=1',name],capture_output=True,text=True,timeout=120)
 assert x.returncode==0,(name,x.stderr);tops[name]=x.stdout
for parent,dirs,files in os.walk(D,followlinks=False):
 dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
 for name in files:
  p=Path(parent)/name
  try:
   s=p.lstat();identity=(s.st_dev,s.st_ino)
   if s.st_dev!=dev or not stat.S_ISREG(s.st_mode) or identity in seen:continue
   seen.add(identity);ext=p.suffix.lower() or '<none>';g=types[ext];g['files']+=1;g['logical_bytes']+=s.st_size;g['allocated_bytes']+=s.st_blocks*512
   if s.st_blocks*512>256*1024**2:large.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns))
  except OSError as e:errs.append(dict(path=str(p),error=repr(e)))
out=dict(UTC=datetime.now(timezone.utc).isoformat(),top_level_du=top,largest_30_directories_children=tops,extensions=dict(types),largest_files=sorted(large,key=lambda r:r['allocated_bytes'],reverse=True)[:100],errors=errs,free_bytes=shutil.disk_usage(R).free,read_only=True)
home=batch/('all_project_diagnostic_attribution_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'));home.mkdir();raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode();p=home/'complete_size_partition.json';p.write_bytes(raw);out['evidence_path']=str(p);out['evidence_SHA256']=hashlib.sha256(raw).hexdigest();print(json.dumps(out,ensure_ascii=False))

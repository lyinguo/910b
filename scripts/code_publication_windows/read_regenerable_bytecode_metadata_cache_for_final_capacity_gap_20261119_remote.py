import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import os,json,stat,shutil
roots=[Path('/root/.cache/pycache'),Path('/root/.cache/openwebui-910b/pycache'),Path('/root/.cache/uv/simple-v27'),Path('/root/.cache/uv/wheels-v7'),Path('/root/.cache/uv/sdists-v10')]
groups=defaultdict(lambda:dict(files=0,bytes=0,allocated_bytes=0,sample=[]));files=[];envrefs=[];live=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  env=(proc/'environ').read_bytes().split(b'\0');selected=[x.decode(errors='replace') for x in env if x.startswith((b'PYTHONPYCACHEPREFIX=',b'UV_CACHE_DIR='))]
  if selected:envrefs.append(dict(PID=int(proc.name),comm=(proc/'comm').read_text().strip(),cache_environment=selected))
  for fd in (proc/'fd').iterdir():
   try:
    n=os.readlink(fd)
    if any(n.startswith(str(r)+'/') for r in roots):live.append(dict(PID=int(proc.name),path=n,kind='open_fd'))
   except OSError:pass
  maps=(proc/'maps').read_text(errors='replace')
  for line in maps.splitlines():
   if any(str(r)+'/' in line for r in roots):live.append(dict(PID=int(proc.name),path=line.split()[-1],kind='mapped_file'))
 except OSError:pass
for root in roots:
 if not root.exists():continue
 for parent,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[x for x in dirs if not (Path(parent)/x).is_symlink()]
  for n in names:
   p=Path(parent)/n;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):continue
   row=dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,suffix=p.suffix)
   key=str(root)+':'+p.suffix;g=groups[key];g['files']+=1;g['bytes']+=s.st_size;g['allocated_bytes']+=s.st_blocks*512
   if len(g['sample'])<3:g['sample'].append(row)
   files.append(row)
tmp=sorted([dict(path=str(p),bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,suffix=p.suffix) for p in Path('/tmp').iterdir() if p.is_file() and not p.is_symlink()],key=lambda r:r['allocated_bytes'],reverse=True)[:25]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),groups=dict(groups),files=files,cache_environment_actual_refs=envrefs,live_cache_FD_or_maps=live,tmp_largest_direct_files=tmp,free_bytes=shutil.disk_usage('/root/projects/comperror').free,read_only=True),ensure_ascii=False))

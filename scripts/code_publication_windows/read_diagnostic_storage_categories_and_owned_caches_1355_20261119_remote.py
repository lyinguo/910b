import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,os,stat,subprocess,shutil,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def du(p,depth):
 q=subprocess.run(['du','-x','-B1','--max-depth='+str(depth),str(p)],capture_output=True,text=True,timeout=180)
 assert q.returncode==0,(str(p),q.stderr)
 return q.stdout
detail={str(D):du(D,2),str(B):du(B,1),'/root/.cache/pip':du('/root/.cache/pip',2)}
types=defaultdict(lambda:dict(files=0,logical_bytes=0,allocated_bytes=0));stage=defaultdict(lambda:dict(files=0,logical_bytes=0,allocated_bytes=0));largest=[];caches=[];tmp=[];errors=[];seen=set();rootdev=D.stat().st_dev
def bucket(rel):
 parts=rel.parts
 if 'heartbeat_handoffs' in parts:return 'handoff_archives'
 if any(s in parts for s in ['source_snapshot','source_snapshots','source_archive','primary_source_snapshots']):return 'source_snapshots'
 if rel.suffix.lower() in ['.npz','.npy','.pt','.safetensors','.bin']:return 'numeric_arrays_or_model_binary'
 if rel.suffix.lower() in ['.log','.out','.err']:return 'logs'
 if rel.suffix.lower() in ['.json','.jsonl','.yaml','.yml','.toml','.csv','.txt']:return 'reports_metadata_configs_or_traces'
 if rel.suffix.lower() in ['.py','.mjs','.js','.sh','.ps1']:return 'source'
 return 'other'
for parent,dirs,files in os.walk(D,followlinks=False):
 dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
 for name in files:
  p=Path(parent)/name
  try:
   s=p.lstat()
   if not stat.S_ISREG(s.st_mode) or s.st_dev!=rootdev:continue
   inode=(s.st_dev,s.st_ino)
   if inode in seen:continue
   seen.add(inode);rel=p.relative_to(D);row=dict(path=str(p),logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns)
   for target,key in [(types,bucket(rel)),(stage,rel.parts[0])]:
    r=target[key];r['files']+=1;r['logical_bytes']+=s.st_size;r['allocated_bytes']+=s.st_blocks*512
   if s.st_blocks*512>128*1024**2:largest.append(row)
   if '__pycache__' in rel.parts or name.endswith(('.pyc','.pyo')):caches.append(row)
   if name.endswith(('.tmp','.part','.temp','.download')):tmp.append(row)
  except OSError as e:errors.append(dict(path=str(p),error=repr(e)))
pip_files=[]
for root in [Path('/root/.cache/pip/http-v2'),Path('/root/.cache/pip/selfcheck')]:
 if not root.exists():continue
 for parent,dirs,files in os.walk(root,followlinks=False):
  for n in files:
   p=Path(parent)/n;s=p.lstat()
   if stat.S_ISREG(s.st_mode):pip_files.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
cache_process_refs=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');maps=(proc/'maps').read_text(errors='replace')
  if '/root/.cache/uv/archive-v0' in argv or '/root/.cache/uv/archive-v0' in maps:cache_process_refs.append(dict(PID=int(proc.name),comm=(proc/'comm').read_text().strip(),cache='/root/.cache/uv/archive-v0',in_argv='/root/.cache/uv/archive-v0' in argv,in_maps='/root/.cache/uv/archive-v0' in maps))
 except OSError:pass
out=dict(UTC=datetime.now(timezone.utc).isoformat(),project_realpath=str(R.resolve()),project_path_is_symlink=R.is_symlink(),du_complete=detail,diagnostic_categories=dict(types),diagnostic_top_level_files=dict(stage),large_files=sorted(largest,key=lambda r:r['allocated_bytes'],reverse=True)[:120],owned_python_bytecode_cache_files=caches,temporary_suffix_files=tmp,package_HTTP_cache_files=pip_files,uv_cache_runtime_process_refs=cache_process_refs,errors=errors,free_bytes=shutil.disk_usage(R).free,no_mutation=True)
home=D/('storage_attribution_detail_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'));home.mkdir();b=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode();p=home/'read_only_categories_complete.json';p.write_bytes(b);out['evidence_path']=str(p);out['evidence_SHA256']=hashlib.sha256(b).hexdigest();print(json.dumps(out,ensure_ascii=False))

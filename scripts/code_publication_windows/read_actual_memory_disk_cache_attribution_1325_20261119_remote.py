import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(argv,timeout=90):
 t=time.monotonic()
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
  return dict(argv=argv,rc=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-t)
 except subprocess.TimeoutExpired as e:return dict(argv=argv,timed_out=True,stdout=(e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else e.stdout,stderr=str(e),seconds=time.monotonic()-t)
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());H=Path(ptr['handoff_directory'])
files=[H/'handoff.txt',H/'facts.json',H/'closing_addendum.json',D/'A100_ALIGNMENT_HANDOFF_CURRENT.json']
notes=H/'late_actual_notes'
if notes.exists():files.extend(sorted(notes.glob('*/actual.json')))
records=[]
for p in files:
 b=p.read_bytes();records.append(dict(path=str(p),bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),text=b.decode('utf-8')))
commands=[['df','-B1','-T'],['free','-b'],['ps','-eo','pid,ppid,comm,rss,vsz,etime','--sort=-rss'],['du','-x','-B1','--max-depth=2',str(R)]]
for p in ['/root/.cache','/tmp','/var/tmp','/var/cache','/root/.ascend','/root/Ascend','/root/autotune','/root/projects']:
 if Path(p).exists():commands.append(['du','-x','-B1','--max-depth=2' if p!='/root/projects' else '--max-depth=1',p])
results=[run(a) for a in commands]
unlinked=[];errs=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  for fd in (proc/'fd').iterdir():
   try:
    target=os.readlink(fd);st=fd.stat()
    if st.st_nlink==0 and st.st_size>1024**2:unlinked.append(dict(PID=int(proc.name),fd=fd.name,target=target,logical_bytes=st.st_size,allocated_bytes=st.st_blocks*512,device=st.st_dev,inode=st.st_ino,comm=(proc/'comm').read_text().strip()))
   except (OSError,ProcessLookupError):pass
 except (OSError,ProcessLookupError):pass
out=dict(UTC=datetime.now(timezone.utc).isoformat(),memory_info=Path('/proc/meminfo').read_text(),project_disk_usage=shutil.disk_usage(R)._asdict(),handoff_complete_records=records,read_only_inventory=results,large_open_unlinked_files=unlinked,no_deletion_or_source_change=True)
home=D/('storage_memory_cache_user_diagnosis_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'));home.mkdir()
raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode();p=home/'read_only_inventory_complete.json';p.write_bytes(raw)
print(json.dumps(dict(**out,evidence_path=str(p),evidence_SHA256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False))

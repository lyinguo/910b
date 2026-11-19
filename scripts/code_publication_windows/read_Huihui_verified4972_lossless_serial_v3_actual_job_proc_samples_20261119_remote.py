import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v3_real_FLOCK_20261119';H=B/'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v3_20261119'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def maybe(p):return js(p) if p.is_file() else None
files={}
for home in [H,J]:
 files[str(home)]={p.name:maybe(p) for p in home.glob('*.json')} if home.exists() else None
stages={}
for stage in ['fresh3','formal200']:
 root=J/stage
 if root.exists():
  stages[stage]={}
  for p in root.iterdir():
   if p.is_file():
    if p.suffix=='.json' and p.stat().st_size<2*1024**2:stages[stage][p.name]=js(p)
    elif p.name.endswith(('.log','.jsonl')):
     with p.open('rb') as f:f.seek(max(0,p.stat().st_size-24000));raw=f.read()
     stages[stage][p.name]=dict(bytes=p.stat().st_size,tail=raw.decode(errors='replace'))
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
procs=project_processes();owners=runner_lock_owners();identities={}
for pid in {1565093,*[x['PID'] for x in procs],*[x['PID'] for x in owners]}:
 p=Path('/proc')/str(pid)
 if p.is_dir():
  identities[str(pid)]=dict(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text())
samples=[]
for proc in procs:
 if proc.get('role')!='model':continue
 pid=proc['PID']
 for folder in S.glob('HuihuiGLM46Flash_own_original_BF16_2048_*_'+str(pid)):
  trace=folder/'actual_sample_trace.jsonl';rows=[json.loads(x) for x in trace.read_text().splitlines()] if trace.is_file() else []
  samples.append(dict(evidence_root=str(folder),actual_model_PID=pid,completed_samples=len(rows),committed_forwards=sum(x['actual_native_forwards'] for x in rows),failed_records=[dict(path=str(p),record=js(p)) for p in folder.glob('*_failed.json')],archive_files=[dict(name=p.name,bytes=p.stat().st_size) for p in folder.glob('sample_*.npz')],all_own_source_actual_NPU_exists=(folder/'all_own_source_actual_NPU.json').exists()))
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert smi.returncode==0
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,stages=stages,current_project_processes=procs,current_original_FLOCK=owners,actual_process_identity=identities,actual_samples=samples,npu_smi=smi.stdout,free_bytes=shutil.disk_usage(R).free,original_job_and_live_sources_not_modified=True),ensure_ascii=False))

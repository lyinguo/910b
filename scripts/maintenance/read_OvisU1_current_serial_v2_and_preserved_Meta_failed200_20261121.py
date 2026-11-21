"""Read the current own Ovis serial job and preserved failed Meta measurement."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'OvisU1_completed_storage_fullNF_passed_after_exact_GLM41_23_capacity_serial_v2_20261121'
J=B/'OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121'
def file(p):
 p=Path(p)
 if not p.is_file():return dict(exists=False)
 b=p.read_bytes();v=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_text=b.decode(errors='replace'))
 try:v['complete_JSON']=json.loads(b)
 except ValueError:pass
 return v
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  if not args or not Path(args[0]).name.startswith('python') or not any(any(z in s.lower() for z in ['ovisu1','evidence_runner','retain_verified183_glm41','watch_batch_910b_immediate']) for s in args[1:]):continue
  stat=(p/'stat').read_text();t=stat.rsplit(')',1)[1].split()
  if t[0]=='Z':continue
  processes.append(dict(PID=int(p.name),PPid=int(t[1]),startticks=int(t[19]),argv=args,cwd=os.readlink(p/'cwd'),stat=stat,io=(p/'io').read_text()))
 except FileNotFoundError:pass
files={}
for home in [H,J,B/'GLM41_reviewed_exact23_apply_then_Ovis_passed_storage_v2_actual_20261121']:
 for name in ['state.json','job.json','actual_completion.json','completion.json','own_first200_measurement_approval.json','Ovis_resume_stderr.log','Ovis_resume_stdout.log']:
  p=home/name
  if p.is_file():files[str(p)]=file(p)
for stage in ['fresh3','formal200']:
 for name in ['console.log','verification_stdout.json','verification_stderr.log','accepted.json']:
  p=J/stage/name
  if p.is_file():files[str(p)]=file(p)
captures=[]
for home in sorted((R/'workspace/diagnostics/model_runs').glob('OvisU1_own_original_BF16_512_v2_guard_*')):
 p=home/'actual_sample_trace.jsonl';rows=[]
 if p.is_file():
  b=p.read_bytes();end=b.rfind(b'\n');rows=[json.loads(s) for s in b[:end+1].splitlines() if s] if end>=0 else []
 captures.append(dict(home=str(home),complete_samples=len(rows),committed_native_forwards=sum(v.get('actual_native_forwards',0) for v in rows),model_PID_matches_capture=any(home.name.endswith('_'+str(p['PID'])) and any(Path(a).name=='run.py' for a in p['argv']) for p in processes)))
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,actual_processes=processes,original_FLOCK=runner_lock_owners(),captures=captures,free_bytes=shutil.disk_usage(R).free,npu_smi=dict(actual_returncode=npu.returncode,stdout=npu.stdout,stderr=npu.stderr),canonical_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]},no_model_or_original_source_change=True),ensure_ascii=False))

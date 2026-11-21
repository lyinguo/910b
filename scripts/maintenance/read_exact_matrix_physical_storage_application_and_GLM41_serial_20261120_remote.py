import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
H=B/'user_authorized_periodic_cleanup_exact_matrix_bytes_all_paths_retained_first_serial_recovery_20261120';L=B/'user_periodic_exact_matrix_storage_application_launch_20261120';G=B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120';J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
files={}
for home in [H,L,G,J]:
 for n in ['state.json','progress.json','actual_completion.json','completion.json','launch.json','tests.json','runner_stdout.log','runner_stderr.log','stdout.log','stderr.log']:
  p=home/n
  if not p.is_file():continue
  raw=p.read_bytes();rec=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
  if p.suffix=='.json':rec['value']=json.loads(raw)
  else:rec['complete_text']=raw.decode(errors='replace')
  files[str(p)]=rec
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  if argv and Path(argv[0]).name.startswith('python') and any(Path(s).name=='run.py' or any(k in s.lower() for k in ['apply_exact_byte_identical','glm41thinking','metamllama','a100_alignment_evidence_runner']) for s in argv[1:]):procs.append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text()))
 except FileNotFoundError:continue
captures=[]
for home in sorted((R/'workspace/diagnostics/model_runs').glob('GLM41Thinking*')):
 rows={}
 for name in ['actual_sample_trace.jsonl','sample_execution.jsonl','generation_progress.jsonl','native_forward_progress.jsonl']:
  p=home/name
  if p.is_file():
   vals=[json.loads(s) for s in p.read_bytes().splitlines() if s];rows[name]=dict(rows=len(vals),last=vals[-1] if vals else None,sum_generation_count=sum(v.get('generation_count',0) for v in vals))
 captures.append(dict(home=str(home),traces=rows))
cards=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert cards.returncode==0
attempts={}
for home in [J/'fresh3',J/'formal200']:
 if home.is_dir():
  attempts[str(home)]=[dict(name=p.name,bytes=p.stat().st_size) for p in home.iterdir() if p.is_file()]
  for p in home.glob('*.log'):
   if p.stat().st_size<=8*2**20:files[str(p)]=dict(bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),complete_text=p.read_text(errors='replace'))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,original_FLOCK=runner_lock_owners(),files=files,actual_processes=procs,GLM41_captures=captures,npu_smi_complete=cards.stdout,current_attempt_files=attempts,new_formal200_job_exists=(J/'job.json').is_file()),ensure_ascii=False))

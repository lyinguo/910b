"""Frozen monitor plus actual own GLM Base process, committed forwards and full errors."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p,full=True):
 p=Path(p);s=p.stat();v=dict(SHA256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 if full:
  b=p.read_bytes()
  try:v['complete_value']=json.loads(b)
  except ValueError:v['complete_value']=b.decode('utf8',errors='replace')
 return v
monitor=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(monitor)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
run=subprocess.run(['python3','-B',str(monitor)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr;summary=json.loads(run.stdout)
mon={str(p):rec(p) for p in [Path(summary['report']),D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json'] if p.is_file()}
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  if not a or not Path(a[0]).name.startswith('python'):continue
  cwd=os.readlink(p/'cwd')
  if cwd!=str(R) and not any(str(R) in v for v in a):continue
  st=(p/'stat').read_text();tail=st[st.rfind(')')+2:].split();fds=[]
  for fd in (p/'fd').iterdir():
   try:
    dest=os.readlink(fd)
    if str(D) in dest or 'davinci' in dest or 'devmm' in dest:fds.append(dict(fd=fd.name,target=dest))
   except OSError:pass
  processes.append(dict(PID=int(p.name),PPid=int(tail[1]),startticks=int(tail[19]),argv=a,cwd=cwd,relevant_FD=fds,io=(p/'io').read_text()))
 except (OSError,ProcessLookupError):continue
records={};stems=['GLM41Base_own_original_probability_full_numeric_preflight_and_serial_launcher_v1_20261122','GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122','GLM41Base_own_original_option_probability_BF16_native_source_readiness_v1_20261122','GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122','GLM41Base_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122']
for stem in stems:
 for name in ['state.json','readiness.json','job.json','launch.json','actual_completion.json','actual_failure.json','completion.json','real_artifact_test_completion.json','full_value_artifact_tests_stdout.log','full_value_artifact_tests_stderr.log','controller_stdout.json','controller_stderr.log','readiness_stdout.log','readiness_stderr.log','own_genuine_NEWtrue3_original_probability_fullNF_active_stderr.log','own_genuine_NEWtrue3_original_probability_fullNF_active_stdout.log','fresh3/console.log','fresh3/verification_stderr.log','fresh3/verification_stdout.json','formal200/console.log','formal200/verification_stderr.log','formal200/verification_stdout.json']:
  p=B/stem/name
  if p.is_file():records[str(p)]=rec(p)
captures=[]
for h in sorted((R/'workspace/diagnostics/model_runs').glob('GLM41Base_own_original_option_logprob_BF16_*')):
 p=h/'committed_option_forwards.jsonl';text=p.read_text() if p.is_file() else '';lines=text.splitlines();pending=bool(text and not text.endswith('\n'));lines=lines[:-1] if pending else lines;rows=[json.loads(s) for s in lines if s.strip()]
 captures.append(dict(home=str(h),actual_committed_native_option_forwards=len(rows),committed_samples=len((h/'actual_sample_trace.jsonl').read_text().splitlines()) if (h/'actual_sample_trace.jsonl').is_file() else 0,complete_committed_forward_rows=rows,pending_partial_line=pending,NPZ_stat_only=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in h.glob('*.npz')],failure_complete_records={str(p):rec(p) for p in h.glob('*failed.json')}))
canonical={str(p):rec(p,False) for p in [R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx',D/'state.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'] if p.is_file()}
cards=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),frozen_monitor_summary=summary,complete_latest_report_checkpoint=mon,actual_proc_argv_cwd_PPid_startticks_FD_io=processes,project_processes=project_processes(),actual_original_lock_owners=runner_lock_owners(),cards_returncode=cards.returncode,cards_stdout=cards.stdout,cards_stderr=cards.stderr,actual_original_disk_free_bytes=shutil.disk_usage(R).free,current_own_GLM41Base_complete_records=records,current_own_GLM41Base_committed_capture_progress=captures,canonical_SHA_stat=canonical,no_active_NPZ_open_or_source_job_config_deadline_signal_mutation=True),ensure_ascii=False))

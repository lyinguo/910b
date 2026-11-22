"""Full current attempt evidence; completed capacity evidence is reused by SHA/stat."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p,full=True):
 p=Path(p);s=p.stat();x=dict(path=str(p),SHA256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 if full:
  b=p.read_bytes()
  try:x['complete_value']=json.loads(b)
  except ValueError:x['complete_value']=b.decode('utf8',errors='replace')
 return x
monitor=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(monitor)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
run=subprocess.run(['python3','-B',str(monitor)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr
summary=json.loads(run.stdout);mon={str(p):rec(p) for p in [Path(summary['report']),D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json'] if p.is_file()}
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  if not argv or not any(str(R) in a for a in argv):continue
  st=(p/'stat').read_text();tail=st[st.rfind(')')+2:].split();fds=[]
  for fd in (p/'fd').iterdir():
   try:
    dest=os.readlink(fd)
    if str(D) in dest or 'davinci' in dest or 'devmm' in dest:fds.append(dict(fd=fd.name,target=dest))
   except OSError:pass
  processes.append(dict(PID=int(p.name),PPid=int(tail[1]),startticks=int(tail[19]),argv=argv,cwd=os.readlink(p/'cwd'),io=(p/'io').read_text(),relevant_FD=fds))
 except (OSError,ProcessLookupError):continue
records={}
stems=['HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v3_original_registration_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_source_readiness_v3_original_registration_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v3_original_registration_real_FLOCK_20261122','HancomHiLlava31_private_original_registration_v3_actual_launcher_20261122']
for stem in stems:
 for name in ['state.json','readiness.json','job.json','actual_completion.json','actual_failure.json','completion.json','launch.json','stderr.log','readiness_stderr.log','exact_zero_forward_original_registration_recovery.json','fresh3/console.log','fresh3/verification_stderr.log','fresh3/verification_stdout.json']:
  p=B/stem/name
  if p.is_file():records[str(p)]=rec(p)
captures=[]
for home in sorted((R/'workspace/diagnostics/model_runs').glob('HancomHiLlava31_own_original_FP16_2048_*')):
 p=home/'generation_progress.jsonl';text=p.read_text() if p.is_file() else '';lines=text.splitlines();pending=bool(text and not text.endswith('\n'));lines=lines[:-1] if pending else lines;rows=[json.loads(z) for z in lines if z.strip()];counts={}
 for row in rows:counts[str(row['index'])]=max(counts.get(str(row['index']),0),int(row['step']))
 captures.append(dict(home=str(home),committed_counts=counts,total_committed_native_forwards=len(rows),last_committed_step=rows[-1] if rows else None,pending_partial_line=pending,committed_sample_execution=(home/'sample_execution.jsonl').read_text() if (home/'sample_execution.jsonl').is_file() else None,NPZ=[dict(path=str(q),bytes=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns) for q in home.glob('groups/*/*.npz')]))
previous={}
for stem in ['HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_own_BF16_guard_20261122','LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122']:
 p=B/stem/'actual_completion.json'
 if p.is_file():previous[str(p)]=rec(p)
 p=B/stem/'own_passed3_complete_all200_shape_capacity.json'
 if p.is_file():previous[str(p)]=rec(p,False)
canon={str(p):rec(p,False) for p in [R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx',D/'state.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'] if p.is_file()}
cards=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),frozen_monitor_summary=summary,complete_latest_report_checkpoint=mon,lock_owners=runner_lock_owners(),project_processes=project_processes(),actual_proc_argv_cwd_PPid_startticks_FD_io=processes,card_returncode=cards.returncode,card_stdout=cards.stdout,card_stderr=cards.stderr,actual_original_disk_free_bytes=shutil.disk_usage(R).free,Hancom_private_v3_current_complete_records=records,Hancom_all_capture_real_progress=captures,completed_previous_actual=previous,canonical_SHA_stat=canon,no_model_or_old_job_source_mutation=True),ensure_ascii=False))

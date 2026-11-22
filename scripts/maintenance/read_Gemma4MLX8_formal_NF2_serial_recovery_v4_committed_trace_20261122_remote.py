"""Read actual own serial worker, relative argv, true locks and complete small attempt files."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
PREFIX='Gemma4E4BMLX8_own_original_Transformers_BF16'
H=B/'Gemma4E4BMLX8_completed_genuine3_formal_NF2_actual_profiler_schema_then_capacity_serial_v2_20261122'
J3=B/(PREFIX+'_true3_v1_real_FLOCK_20261122');J200=B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def file(p,limit=2**20):
 p=Path(p)
 if not p.is_file():return dict(path=str(p),exists=False)
 s=p.stat();o=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,SHA256=sha(p))
 if s.st_size<=limit:o['text']=p.read_text(encoding='utf-8',errors='replace')
 else:
  with p.open('rb') as f:f.seek(max(0,s.st_size-65536));o['tail_last65536_bytes']=f.read().decode(errors='replace')
 return o
result=dict(UTC=datetime.now(timezone.utc).isoformat(),disk_free_bytes=shutil.disk_usage(R).free,control=[],jobs=[],processes=[],captures=[])
for p in [H/'state.json',H/'actual_completion.json',H/'formal_NF2_stdout.json',H/'formal_NF2_stderr.log',H/'formal_NF2_actual_process_returncode.json',H/'formal_same_genuine3_acceptance.json',H/'own_completed3_all_ZIP_all200_shape_capacity.json',H/'first200_measurement_approval.json',B/'Gemma4E4BMLX8_completed3_formal_NF2_profiler_serial_v2_launch_20261122.log']:result['control'].append(file(p))
for job in [J3,J200]:
 rec=dict(job=str(job),exists=job.exists(),files=[])
 if job.exists():
  for p in sorted(job.rglob('*')):
   if p.is_file() and p.suffix in ['.json','.jsonl','.log'] and (p.name in ['job.json','state.json','completion.json','returncode.json','attempts.jsonl','attempt_trace.jsonl'] or p.name=='console.log' or 'stderr' in p.name or 'stdout' in p.name):rec['files'].append(file(p))
 result['jobs'].append(rec)
for path in Path('/proc').iterdir():
 if not path.name.isdigit() or int(path.name)==os.getpid():continue
 try:args=[a.decode(errors='replace') for a in (path/'cmdline').read_bytes().split(b'\0') if a];cwd=str((path/'cwd').resolve());status=(path/'status').read_text();stat=(path/'stat').read_text()
 except OSError:continue
 if not args or not Path(args[0]).name.startswith('python'):continue
 if not any(any(tag in a for tag in ['Gemma4E4BMLX8','Gemma4MLX8','gemma4_e4b_MLX8','A100_alignment_evidence_runner','run.py','verify_']) for a in args):continue
 rec=dict(PID=int(path.name),argv=args,cwd=cwd,PPid=next((s.split(':',1)[1].strip() for s in status.splitlines() if s.startswith('PPid:')),None),stat=stat,status=status,io=(path/'io').read_text() if (path/'io').exists() else None,FD=[])
 for fd in (path/'fd').iterdir():
  try:target=os.readlink(fd)
  except OSError:continue
  if any(t in target for t in [str(R),'runner.lock','.npz','run.log']):rec['FD'].append(dict(fd=fd.name,target=target))
 result['processes'].append(rec)
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
result['runner_lock_owners']=runner_lock_owners();result['kernel_locks']=Path('/proc/locks').read_text()
root=R/'workspace/diagnostics/model_runs'
for c in sorted(root.glob('Gemma4E4BMLX8_own_original_BF16_*')):
 row=dict(path=str(c),trace=file(c/'actual_sample_trace.jsonl'),records=[])
 committed=[];trace_path=c/'actual_sample_trace.jsonl'
 if trace_path.is_file():
  for raw_line in trace_path.read_bytes().splitlines(keepends=True):
   if raw_line.endswith(b'\n'):committed.append(json.loads(raw_line))
   else:row['trace_last_incomplete_bytes_pending_next_incremental_read']=len(raw_line)
 for committed_row in committed:
  p=Path(committed_row['actual_execution_file']);assert sha(p)==committed_row['actual_execution_SHA256'];s=p.stat();x=json.loads(p.read_bytes());row['records'].append(dict(path=str(p),SHA256=sha(p),index=x.get('index'),forwards=x.get('actual_native_forwards',x.get('committed_native_forwards')),raw=x.get('raw_output'),error=x.get('error'),archive=x.get('archive'),archive_bytes=x.get('archive_bytes'),mtime_ns=s.st_mtime_ns))
 for n in ['completed_trace.json','independent_report_verification.json','independent_report_verification_profiler_v2.json']:
  p=c/n
  if p.exists():row[n]=file(p,65536)
 result['captures'].append(row)
result['actual_report_folders']=[]
reportroot=R/'workspace/reports/unsloth_gemma-4-E4B-it-MLX-8bit/image-text-to-text'
if reportroot.is_dir():
 for report in sorted(reportroot.iterdir()):
  if report.is_dir():result['actual_report_folders'].append(dict(path=str(report),files=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in report.iterdir() if p.is_file()],run_log=file(report/'run.log'),eval_report=file(report/'eval_report.json',65536)))
q=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);result['npu_smi']=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
result['canonical_SHA256']={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx']}
print(json.dumps(result,ensure_ascii=False))

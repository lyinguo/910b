"""Read actual resource recovery watcher and current committed native forwards."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'immediate_disk_recovery_cleanup_watch_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files={}
for n in ['state.json','launch.json','first_actual_noninterrupting_check.json','actual_failure.json','watch_stdout.log','watch_stderr.log']:
 p=H/n
 if p.is_file():
  raw=p.read_bytes();files[n]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw) if p.suffix=='.json' else raw.decode(errors='replace'))
launch=files['launch.json']['value'];proc=Path('/proc')/str(launch['PID']);assert proc.exists();argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x];assert argv==launch['argv'];watch=dict(PID=launch['PID'],argv=argv,cwd=os.readlink(proc/'cwd'),stat=(proc/'stat').read_text())
for n,h in launch['exact_source_SHA256'].items():assert sha(n)==h
p=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/read_exact_matrix_physical_storage_application_and_GLM41_serial_20261120_remote.py')],capture_output=True,text=True);assert p.returncode==0,p.stderr;actual=json.loads(p.stdout);native=[]
for cap in actual['GLM41_captures']:
 home=Path(cap['home']);current=any(home.name.endswith('_'+str(x['PID'])) and any(Path(s).name=='run.py' for s in x['argv'][1:]) for x in actual['actual_processes']);trace=home/'actual_sample_trace.jsonl'
 if not current or not trace.is_file():continue
 raw=trace.read_bytes();complete=[];pending=0
 for line in raw.splitlines(keepends=True):
  if line.endswith(b'\n'):complete.append(json.loads(line))
  else:pending=len(line)
 assert [x['index'] for x in complete]==list(range(len(complete)))
 native.append(dict(home=str(home),completed_samples=len(complete),committed_native_forwards=sum(x['actual_native_forwards'] for x in complete),last_completed_index=complete[-1]['index'] if complete else None,pending_nonterminated_tail_bytes=pending,source_trace_SHA256=hashlib.sha256(raw).hexdigest(),current_model_PID_matches_home=True))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),watcher_actual=watch,watcher_files=files,current_committed_NPU=native,formal_actual=actual,no_model_cache_math_source_job_deadline_or_canonical_change=True),ensure_ascii=False))

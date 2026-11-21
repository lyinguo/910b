"""Current own Meta original serial run plus completed GLM full200 evidence."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
META=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'
GLM=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
MH=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120'
GH=B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def file(p):
 p=Path(p)
 if not p.is_file():return dict(exists=False)
 b=p.read_bytes();v=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_text=b.decode('utf-8',errors='replace'))
 try:v['complete_JSON']=json.loads(b)
 except ValueError:pass
 return v
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if not args or not Path(args[0]).name.startswith('python'):continue
  if not any(any(s in a.lower() for s in ['glm41','glm-4.1v','meta-llama--llama-3.2','meta_mllama','metamllama','a100_alignment_evidence_runner','watch_batch_910b_immediate']) for a in args[1:]):continue
  tail=(p/'stat').read_text().rsplit(')',1)[1].split()
  if tail[0]=='Z':continue
  procs.append(dict(PID=int(p.name),PPid=int(tail[1]),startticks=int(tail[19]),argv=args,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text()))
 except FileNotFoundError:continue
files={}
for home in [META,GLM,MH,GH,D/'immediate_disk_recovery_cleanup_watch_20261120']:
 for n in ['job.json','state.json','completion.json','actual_completion.json','own_first200_measurement_approval.json']:
  p=home/n
  if p.is_file():files[str(p)]=file(p)
for home in [META,GLM]:
 for stage in ['fresh3','formal200']:
  for n in ['console.log','verification_stdout.json','verification_stderr.log','accepted.json']:
   p=home/stage/n
   if p.is_file():files[str(p)]=file(p)
captures=[]
for home in sorted((R/'workspace/diagnostics/model_runs').glob('MetaMllama_own_original_SDPA2048_*')):
 rows=[];tail_bytes=0;trace=home/'sample_execution.jsonl'
 if trace.is_file():
  raw=trace.read_bytes();end=raw.rfind(b'\n');body=raw[:end+1] if end>=0 else b'';tail_bytes=len(raw)-len(body);rows=[json.loads(x) for x in body.splitlines() if x]
 progress=home/'generation_progress.jsonl';pr=[]
 if progress.is_file():
  b=progress.read_bytes();end=b.rfind(b'\n');pr=[json.loads(x) for x in b[:end+1].splitlines() if x] if end>=0 else []
 current=any(home.name.endswith('_'+str(p['PID'])) and any(Path(a).name=='run.py' for a in p['argv'][1:]) for p in procs)
 captures.append(dict(home=str(home),completed_samples=len(rows),committed_native_forwards=sum(x['generation_count'] for x in rows),pending_nonterminated_tail_bytes=tail_bytes,current_model_PID_matches_capture=current,last_original_progress=pr[-1] if pr else None))
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
out=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_processes=procs,original_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free,files=files,Meta_captures=captures,npu_smi=dict(actual_returncode=npu.returncode,stdout=npu.stdout,stderr=npu.stderr),canonical_SHA256={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]},no_NN_replay_source_job_deadline_signal_or_G_write=True)
print(json.dumps(out,ensure_ascii=False))

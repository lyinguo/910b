import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,actual_original_FLOCK=runner_lock_owners(),frozen_project_snapshot=project_processes(),actual_processes=[],files={},captures=[])
homes=[B/'MetaMllama_own_SDPA2048_serial_controller_v1_launch_20261120',B/'MetaMllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120',B/'MetaMllama_own_original_SDPA_BF16_2048_source_readiness_v1_20261120',B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120',B/'MetaMllama_own_completed3_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120',B/'MetaMllama_completed_genuine3_formal_NF2_exact_original_instruction_no_NN_replay_capacity_serial_20261120',B/'MetaMllama_formal_NF2_same_genuine3_no_NN_replay_source_launch_20261120',B/'MetaMllama_completed_genuine3_formal_NF2_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261120']
homes.extend([B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120',B/'MetaMllama_formal_NF3_same_genuine3_no_NN_replay_source_launch_20261120',B/'MetaMllama_completed_genuine3_formal_NF3_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261120'])
for parent in list(homes):
 if parent.is_dir():homes.extend(p for p in parent.iterdir() if p.is_dir() and p.name in ['fresh3','fixed200','three','200'])
for home in homes:
 for p in sorted(home.glob('*')) if home.is_dir() else []:
  if not p.is_file() or p.suffix not in ['.json','.log','.jsonl']:continue
  raw=p.read_bytes();row=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
  try:row['value']=json.loads(raw)
  except ValueError:row['complete_text']=raw.decode(errors='replace')
  out['files'][str(p)]=row
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  if not argv or not Path(argv[0]).name.startswith('python'):continue
  if not any('meta_mllama' in a.lower() or 'metamllama' in a.lower() or 'meta-llama--llama-3.2-11b-vision-instruct-own-original-bf16-sdpa' in a.lower() for a in argv) and int(p.name) not in {v['PID'] for v in out['actual_original_FLOCK']}:continue
  out['actual_processes'].append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),status=(p/'status').read_text(),io=(p/'io').read_text()))
 except FileNotFoundError:continue
for home in sorted((R/'workspace/diagnostics/model_runs').glob('MetaMllama_own_original_SDPA2048_*')):
 rows=[];trace=home/'sample_execution.jsonl';partial=0
 if trace.is_file():
  lines=trace.read_bytes().splitlines(keepends=True);rows=[json.loads(s) for s in lines if s.endswith(b'\n')];partial=sum(len(s) for s in lines if not s.endswith(b'\n'))
 progress=home/'generation_progress.jsonl';pr=[]
 if progress.is_file():pr=[json.loads(s) for s in progress.read_bytes().splitlines()]
 row=dict(home=str(home),completed_samples=len(rows),committed_native_forwards=sum(x['generation_count'] for x in rows),last_complete_sample=rows[-1] if rows else None,last_actual_forward_progress=pr[-1] if pr else None,pending_incomplete_trace_bytes=partial,current_live_model_PID_matches_capture=any(str(home).endswith('_'+str(v['PID'])) and any(Path(a).name=='run.py' for a in v['argv'][1:]) for v in out['actual_processes']))
 proofs=list(home.glob('independent_report_verification.json'))
 if proofs:row['fullNF_proof']=dict(path=str(proofs[0]),SHA256=sha(proofs[0]),value=json.loads(proofs[0].read_bytes()))
 vp=home/'own_independent_visual_values/progress.json'
 if vp.is_file():row['independent_visual_progress']=json.loads(vp.read_bytes())
 out['captures'].append(row)
sm=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15);out['npu_smi']=dict(returncode=sm.returncode,stdout=sm.stdout,stderr=sm.stderr)
print(json.dumps(out,ensure_ascii=False))

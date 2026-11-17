import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,subprocess,datetime,os,shutil,re
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'two_card_Merve_lossless_Prithiv_card2_parallel_v2_20261117';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p,full=True):
 p=Path(p)
 if not p.exists():return None
 raw=p.read_bytes();v=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
 if p.suffix=='.json' and full:v['value']=json.loads(raw)
 else:v['last_complete_text']=raw[-6500:].decode('utf-8',errors='replace')
 return v
result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),project_processes=project_processes(),real_original_global_FLOCK_owners=runner_lock_owners(),project_free_bytes=shutil.disk_usage(R).free,batch_files={n:read(H/n) for n in ['batch.json','actual_launch.json','coordinator_manifest.json','state.json','actual_completion.json','actual_failure.json','coordinator_failure_pending_natural_children.json','coordinator_stdout.log','coordinator_stderr.log','NOTIFY_capacity_error.json']})
result['jobs']={}
for row in json.loads((H/'batch.json').read_bytes())['jobs']:
 J=Path(row['home']);files={n:read(J/n) for n in ['state.json','completion.json','parallel_actual_launch.json','parallel_worker_stdout.log','parallel_worker_stderr.log','actual_NEWfresh3_complete_storage_capacity_review.json']}
 for count,phase in [(3,'fresh3'),(200,'formal200')]:
  for suffix in ['request','grant','NN_finished','released','refused']:files[f'parallel_stage_{count}_{suffix}.json']=read(J/f'parallel_stage_{count}_{suffix}.json')
  for name in ['console.log','verification_stderr.log','accepted.json']:files[phase+'/'+name]=read(J/phase/name)
 result['jobs'][row['model_id']]=dict(card=row['card'],home=str(J),files=files)
J=B/'Merve_lossless_v2_completed_approved_fresh3_resumed_unstarted200_after_retention_v3_real_FLOCK_20261117'
if J.exists():
 result['same_approved_fresh3_unstarted200_continuation']=dict(home=str(J),files={n:read(J/n) for n in ['job.json','actual_launch.json','state.json','completion.json','same_approved_fresh3_unstarted200_source_and_capacity_review.json','runner_stdout.log','runner_stderr.log','formal200/console.log','formal200/verification_stderr.log','formal200/accepted.json']})
card_locks={}
for card in [2,4]:
 f=D/'parallel_card_locks'/f'NPU_{card}.lock'
 if not f.exists():continue
 st=f.stat();card_locks[card]=[s for s in Path('/proc/locks').read_text().splitlines() if s.split()[1]=='FLOCK' and len(s.split()[5].split(':'))==3 and tuple([int(s.split()[5].split(':')[0],16),int(s.split()[5].split(':')[1],16),int(s.split()[5].split(':')[2])])==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)]
result['real_per_card_kernel_FLOCKs']=card_locks
result['actual_runtime_progress']=[]
launch_utc=json.loads((H/'actual_launch.json').read_bytes())['UTC'];after=datetime.datetime.fromisoformat(launch_utc).timestamp()
for home in (R/'workspace/diagnostics/model_runs').iterdir():
 if not home.is_dir() or not (home.name.startswith('merve_native_vllm_') or home.name.startswith('PrithivGemma4E4B_own_original_BF16_')) or home.stat().st_mtime<after:continue
 trace=home/('requests.jsonl' if home.name.startswith('merve_') else 'actual_sample_trace.jsonl')
 if not trace.exists():continue
 lines=trace.read_text().splitlines();rows=[json.loads(x) for x in lines];v=dict(runtime_home=str(home),trace=str(trace),committed_samples=len(rows),trace_SHA256=sha(trace),error_indices=[x.get('index') for x in rows if x.get('error')])
 if home.name.startswith('merve_'):
  v['committed_generated_token_count']=sum(len(x['generated_token_ids']) for x in rows);index=home/'full_numeric_group_index.json'
  if index.exists():
   groups=json.loads(index.read_bytes())['groups'];v['completed_numeric_groups']=len(groups);v['actual_native_forward_calls_in_completed_groups']=sum(json.loads(Path(x['path']).read_bytes())['actual_native_forward_calls'] for x in groups)
 else:v['committed_native_forward_calls']=sum(x['actual_native_forwards'] for x in rows)
 result['actual_runtime_progress'].append(v)
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);result['actual_npu_smi']=dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(result,ensure_ascii=False))

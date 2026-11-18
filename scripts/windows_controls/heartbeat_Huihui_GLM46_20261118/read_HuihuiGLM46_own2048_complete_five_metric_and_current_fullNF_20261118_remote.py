import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
J=B/'HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118';HC=B/'HuihuiGLM46Flash_own_original_BF16_2048_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261118';REPORT=R/'workspace/reports/huihui-ai_Huihui-GLM-4.6V-Flash-abliterated/image-text-to-text/20261118_203406'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={n:dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p)) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'] if (p:=REPORT/n).is_file()}
states={};static={}
for label,p in [('J/state.json',J/'state.json'),('J/completion.json',J/'completion.json'),('J/fresh3/verification_stdout.json',J/'fresh3/verification_stdout.json'),('J/fresh3/verification_stderr.log',J/'fresh3/verification_stderr.log'),('H/state.json',HC/'state.json'),('H/actual_completion.json',HC/'actual_completion.json'),('H/full_shape_complete_numeric_storage_capacity_review.json',HC/'full_shape_complete_numeric_storage_capacity_review.json')]:
 if not p.is_file():continue
 data=p.read_bytes();rec=dict(path=str(p),bytes=len(data),SHA256=hashlib.sha256(data).hexdigest())
 try:value=json.loads(data)
 except ValueError:value=dict(nonfinal_or_text=True,tail=data.decode('utf8',errors='replace')[-15000:])
 if label.endswith('verification_stdout.json') and isinstance(value,dict) and value.get('passed'):
  value={k:v for k,v in value.items() if k not in ['numeric_sample_checks','external_evidence_sha256','report_sha256']}
 rec['value']=value;states[label]=rec
 if 'state.json' not in label and len(data):static[label]={k:v for k,v in rec.items() if k!='value'}
actual=project_processes();proc={}
for v in actual:
 if v.get('role') not in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']:continue
 pid=v['PID'];p=Path('/proc')/str(pid)
 try:proc[pid]=dict(v,stat=(p/'stat').read_text(),actual_argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),io=(p/'io').read_text())
 except OSError as e:proc[pid]=dict(v,read_error=str(e))
report=json.loads((REPORT/'eval_report.json').read_bytes()) if len(files)==5 else None
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),five_original_files=files,report_metric=report.get('metrics') if report else None,report_statistics=report.get('statistics') if report else None,actual_states=states,static_original_metadata=static,actual_proc_details=proc,actual_original_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free,no_NN_or_numeric_replay_or_model_mutation=True),ensure_ascii=False))

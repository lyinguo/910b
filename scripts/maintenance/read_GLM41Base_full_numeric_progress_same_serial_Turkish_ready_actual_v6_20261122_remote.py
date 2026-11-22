"""Read full module-audit progress and closed five reports; never open active matrices."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_GLM41Base_same_serial_Turkish_own_fullNF_queue_actual_v5_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr;x=json.loads(r.stdout);progress=[];reports={}
for h in sorted((R/'workspace/diagnostics/model_runs').glob('GLM41Base_own_original_option_logprob_BF16_*')):
 p=h/'independent_full_numeric_checks.jsonl';rows=[];pending=False
 if p.is_file():
  b=p.read_bytes();pending=bool(b and not b.endswith(b'\n'));lines=b.splitlines();lines=lines[:-1] if pending else lines;rows=[json.loads(v) for v in lines if v.strip()]
  progress.append(dict(path=str(p),bytes=len(b),observed_SHA256=hashlib.sha256(b).hexdigest(),complete_output_tensor_checks=len(rows),total_compared_values=sum(v['all_values'] for v in rows),last_complete_check=rows[-1] if rows else None,pending_partial_line=pending,all_committed_check_summaries=rows))
J=B/'GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122';state=json.loads((J/'state.json').read_bytes())
if state.get('report_dir'):
 home=Path(state['report_dir'])
 for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','operator_lineage.json','run.log']:
  p=home/n
  if p.is_file():
   b=p.read_bytes()
   try:v=json.loads(b)
   except ValueError:v=b.decode('utf8',errors='replace')
   reports[str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_value=v)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),same_serial_full_actual=x,GLM41Base_current_full_numeric_module_progress=progress,GLM41Base_five_reports_closed_model_stage=reports,no_active_matrix_open_or_NN_NF_source_job_config_deadline_signal_mutation=True),ensure_ascii=False))

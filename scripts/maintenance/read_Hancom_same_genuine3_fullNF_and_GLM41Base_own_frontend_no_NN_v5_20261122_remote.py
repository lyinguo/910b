"""Read the sole formal verifier and next own unweighted frontend separately."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def record(p):
 raw=p.read_bytes()
 try:v=json.loads(raw)
 except ValueError:v=raw.decode('utf8',errors='replace')
 return dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),complete_value=v)
run=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_Hancom_private_original_registration_v3_actual_serial_v4_all_proc_cwd_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr;x=json.loads(run.stdout);records={}
for stem in ['GLM41Base_own_original_option_logprob_native_frontend200_meta_no_NN_v1_20261122','GLM41Base_own_original_option_frontend200_no_NN_actual_launcher_v1_20261122']:
 for name in ['progress.json','review.json','actual_completion.json','actual_failure.json','launch.json','stderr.log','stdout.json']:
  p=B/stem/name
  if p.is_file():records[str(p)]=record(p)
proofs={}
for p,r in x['Hancom_private_v3_current_complete_records'].items():
 if p.endswith('/completion.json') and isinstance(r['complete_value'],dict):
  for accepted in r['complete_value'].get('accepted',[]):
   f=Path(accepted['proof']);assert sha(f)==accepted['proof_sha256'];proofs[str(f)]=record(f)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),Hancom_same_serial_full_actual=x,Hancom_completed_formal_acceptance_proofs_if_any=proofs,GLM41Base_own_original_frontend_only_complete_records=records,GLM41Base_not_a_model_NN_or_fullNF=True,no_active_model_job_source_config_deadline_signal_mutation=True),ensure_ascii=False))

"""Read exact natural completion and original next-model serial evidence, no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except (ValueError,UnicodeDecodeError):return b.decode('utf8',errors='replace')
def complete(p):
 if not p.is_file():return None
 value=read(p);s=p.stat()
 if s.st_size>8*1024*1024 and isinstance(value,dict):
  def summary(v):
   if isinstance(v,(list,dict)):
    b=json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')
    return dict(type=type(v).__name__,count=len(v),all_values_canonical_JSON_SHA256=hashlib.sha256(b).hexdigest())
   return v
  return dict(path=str(p),bytes=s.st_size,SHA256=sha(p),complete_JSON_read=True,original_file_preserved_not_replaced=True,top_level={k:summary(v) for k,v in value.items()})
 return dict(path=str(p),bytes=s.st_size,SHA256=sha(p),complete_value=value)
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_verified185_Xtuner_LlavaMore_serial_ready_frozen_monitor_actual_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
x=json.loads(r.stdout)
J=B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122';H=B/'XtunerLlava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122'
xtuner={str(p):complete(p) for p in [J/'completion.json',J/'fresh3/verification_stdout.json',J/'fresh3/verification_stderr.log',H/'actual_completion.json',H/'own_passed3_complete_all200_shape_capacity.json']}
for key in ['job','readiness','controller']:
 for path,value in x['current_Xtuner_same_serial_actual'].get(key,{}).items():
  if path.endswith('readiness_stderr.log'):continue
  if path.endswith('state.json') or path.endswith('completion.json'):continue
moreJ=B/'LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122'
moreH=B/'LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122'
more200=B/'LlavaMore_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122'
paths=[moreH/'state.json',moreH/'actual_completion.json',moreH/'actual_failure.json',moreH/'own_passed3_complete_all200_shape_capacity.json',moreJ/'state.json',moreJ/'completion.json',moreJ/'fresh3/verification_stdout.json',moreJ/'fresh3/verification_stderr.log',moreJ/'fresh3/console.log',more200/'state.json',more200/'completion.json']
for a in ['fresh3','formal200']:
 for n in ['verification_stdout.json','verification_stderr.log','console.log']:
  p=more200/a/n
  if p.is_file():paths.append(p)
more={str(p):complete(p) for p in paths}
captures=[]
for home in sorted((R/'workspace/diagnostics/model_runs').glob('LlavaMore_own_original_FP16_letter_*')):
 files={}
 for name in ['probability_progress.jsonl','sample_execution.jsonl','model_loading_path.json','actual_weight_loading.json','lineage.json']:
  p=home/name
  if p.is_file():files[name]=complete(p)
 for p in home.glob('*loading*.json'):files[p.name]=complete(p)
 npz=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in home.glob('groups/*/*.npz')]
 captures.append(dict(home=str(home),files=files,NPZ=npz))
reports=[]
reportHome=R/'workspace/reports/aimagelab_LLaVA_MORE-llama_3_1-8B-finetuning/image-text-to-text'
if reportHome.exists():
 for home in sorted(reportHome.iterdir()):
  if not home.is_dir():continue
  files={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in home.iterdir() if p.is_file()}
  selected={n:complete(home/n) for n in ['eval_report.json','operator_lineage.json','run.log']}
  reports.append(dict(home=str(home),files=files,selected_complete=selected))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),frozen_monitor_and_actual_processes=x,Xtuner_completed3_full_proof_capacity=xtuner,More_current_attempt_complete=more,More_captures=captures,More_reports=reports),ensure_ascii=False))

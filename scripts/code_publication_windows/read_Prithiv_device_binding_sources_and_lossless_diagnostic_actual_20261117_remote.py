import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'PrithivGemma4E4B_own_original_BF16_true3_v2_own_config_real_FLOCK_20261116'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=js(J/'job.json');cfg=js(R/job['spec']['config']);o={'UTC':datetime.now(timezone.utc).isoformat(),'job':job,'config':cfg,'completion':js(J/'completion.json'),'sources':{},'diagnostic_files':{}}
for n in job['spec']['source_sha256']:
 if (n.startswith('universal_eval/model/adapters/') and any(t in n.lower() for t in ['prithiv','gemma4_e4b','gemma4e4b'])) or n==job['spec']['verifier']:
  f=R/n;o['sources'][n]={'SHA256':sha(f),'text':f.read_text()}
p=Path(cfg['model']['kwargs']['protocol_file']);o['protocol']={'path':str(p),'SHA256':sha(p),'value':js(p)}
for home in [B/'Merve_lossless_full_numeric3_diagnostic_source_and_actual_launch_v1_20261117',B/'Merve_completed_numeric3_lossless_exact_tensor_sharing_full_independent_diagnostic_v1_20261117']:
 for n in ['actual_completion.json','stdout.json','stderr.log','review.json','failure.json']:
  f=home/n
  if f.exists():
   raw=f.read_bytes()
   try:v=json.loads(raw)
   except Exception:v=raw[-16000:].decode(errors='replace')
   o['diagnostic_files'][str(f)]={'SHA256':sha(f),'value':v}
print(json.dumps(o,ensure_ascii=False))

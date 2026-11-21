import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120'
job=json.loads((J/'job.json').read_bytes());cfg=R/job['spec']['config'];p=Path(json.loads(cfg.read_bytes())['model']['kwargs']['protocol_file'])
paths=[J/'job.json',cfg,p,R/'scripts/continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py',R/'scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py']
out={}
for p in paths:
 raw=p.read_bytes();out[str(p)]={'SHA256':hashlib.sha256(raw).hexdigest(),'complete_text':raw.decode('utf-8')}
print(json.dumps({'files':out}))

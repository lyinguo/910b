import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'
js=lambda p:json.loads(Path(p).read_bytes())
import yaml
j=js(J/'job.json');c=yaml.safe_load((R/j['spec']['config']).read_bytes());p=js(c['model']['kwargs']['protocol_file']);oldcfg=js(R/'configs/queues/Huihui_real_dual181_then_own_GLM46_complete_component_capacity_serial_20261119.json')
print(json.dumps(dict(parent_config=c,parent_spec=j['spec'],protocol=p,old_serial_config=oldcfg,ledger_old_protection=j['protected_canonical_hashes'],original_verifier=js(J/'completion.json')['accepted'][0]),ensure_ascii=False))

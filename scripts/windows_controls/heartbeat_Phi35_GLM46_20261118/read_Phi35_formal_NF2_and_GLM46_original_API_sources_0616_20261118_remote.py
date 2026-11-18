import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
 p=Path(p)
 if not p.exists():return None
 raw=p.read_bytes();s=raw.decode('utf8');value=json.loads(s) if p.suffix=='.json' and s.strip().startswith(('{','[')) else s
 return dict(path=str(p),SHA256=sha(p),bytes=len(raw),value=value)
H=B/'Phi35_completed_genuine_true3_formal_NF2_original_BF16_RoPE_no_NN_replay_0616_20261118'
value=dict(UTC=datetime.now(timezone.utc).isoformat(),files={n:read(H/n) for n in ['state.json','completion.json','actual_completion.json','fresh3/verification_stderr.log']})
L=B/'Phi35_formal_NF2_no_NN_replay_actual_launch_0616_20261118'
value['launch']={n:read(L/n) for n in ['actual_launch.json','stderr.log']}
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
value['processes']=project_processes();value['locks']=runner_lock_owners()
value['sources']={}
for p in [R/'universal_eval/model/adapters/transformers_vlm.py',R/'universal_eval/algorithm/greedy.py']:
 if p.exists():value['sources'][str(p)]=read(p)
import transformers
base=Path(transformers.__file__).parent
for relative in ['models/glm46v/processing_glm46v.py','models/glm46v/image_processing_glm46v.py','models/glm46v/image_processing_glm46v_fast.py','models/glm4v/modeling_glm4v.py','models/glm4v/configuration_glm4v.py']:
 p=base/relative
 if p.exists():value['sources'][str(p)]=read(p)
print(json.dumps(value,ensure_ascii=False))

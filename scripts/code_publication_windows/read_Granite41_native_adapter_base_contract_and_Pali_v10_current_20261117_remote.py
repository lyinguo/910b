import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,inspect
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from universal_eval.model.coder_gemma4_e2b_own_20261116 import observer
from transformers.models.granite import modeling_granite
paths=[R/'universal_eval/model/adapters/transformers_vlm.py',Path(inspect.getfile(observer)),Path(inspect.getfile(modeling_granite))]
for t in observer.Gemma4LosslessObserver.__mro__[1:]:
 try:p=Path(inspect.getfile(t))
 except TypeError:continue
 if p.is_relative_to(R):paths.append(p)
v=dict(procs=project_processes(),FLOCK=runner_lock_owners(),sources={str(p):dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()) for p in paths})
J=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117'
v['Pali_actual']={str(p):p.read_text() if p.suffix!='.json' else json.loads(p.read_bytes()) for p in [J/'state.json',J/'completion.json',J/'fresh3/verification_stderr.log'] if p.exists()}
console=J/'fresh3/console.log'
if console.exists():v['Pali_console_last12']=console.read_text(errors='replace').splitlines()[-12:]
print(json.dumps(v,ensure_ascii=False))

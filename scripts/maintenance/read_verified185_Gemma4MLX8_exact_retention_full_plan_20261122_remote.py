import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'verified185_Gemma4MLX8_fullNF2_dual_audit_correct_nonexample137_exact_retention_20261122';I=B/'verified185_Gemma4MLX8_exact_retention_plan_launcher_20261122'
out={}
for p in [H/'progress.json',H/'plan.json',H/'actual_completion.json',I/'stdout.json',I/'stderr.log']:
 if not p.exists():continue
 try:out[str(p)]=json.loads(p.read_bytes())
 except Exception:out[str(p)]=p.read_text(errors='replace')
print(json.dumps(out,ensure_ascii=False))

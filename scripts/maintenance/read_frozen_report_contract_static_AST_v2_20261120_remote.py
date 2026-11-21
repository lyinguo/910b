import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json
R=Path('/root/projects/comperror');p=R/'scripts/A100_alignment_evidence_runner_20261103.py';s=p.read_text();t=ast.parse(s)
out=dict(imports=[ast.unparse(n) for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom))],matching_definitions={n.name:ast.get_source_segment(s,n) for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name in ['validate_report','validate_completed_exit']},complete_frozen_source=s)
q=R/'scripts/batch_910b_support.py';v=q.read_text();out['actual_support_definitions']={n.name:ast.get_source_segment(v,n) for n in ast.walk(ast.parse(v)) if isinstance(n,ast.FunctionDef) and n.name in ['validate_report','validate_completed_exit']}
print(json.dumps(out,ensure_ascii=False))

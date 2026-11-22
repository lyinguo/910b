"""Read the actual saved probability adapter; no model execution or source edits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';p=R/'universal_eval/model/adapters/glm41v_base_discriminative.py';raw=p.read_bytes();ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());pending=json.loads(Path(ptr['new_primary_pending']).read_bytes());paths=[r for r in pending['all_current_primary_source_and_true_consumer_configs'] if 'glm41' in r['relative'].lower() or 'glm46' in r['relative'].lower()]
print(json.dumps(dict(original_source_path=str(p),original_source_SHA256=hashlib.sha256(raw).hexdigest(),original_source_complete_text=raw.decode(),existing_registered_architecture_sources=paths,no_learned_CPU_or_NPU_execution=True),ensure_ascii=False))

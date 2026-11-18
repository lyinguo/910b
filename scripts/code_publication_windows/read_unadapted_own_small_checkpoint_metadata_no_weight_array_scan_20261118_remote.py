import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import json

R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
state = json.loads((D / 'state.json').read_bytes())['models']
out = []
for mid, row in state.items():
    if row.get('status') != 'needs_adaptation':
        continue
    roots = row.get('weight_paths') or []
    if not roots:
        continue
    root = Path(roots[0])
    try:
        files = list(root.glob('*.safetensors')) if root.is_dir() else []
        size = sum(p.stat().st_size for p in files)
        if not files or size > 2 * 1024**3:
            continue
        config = json.loads((root / 'config.json').read_bytes()) if (root / 'config.json').is_file() else {}
        arch = config.get('architectures', [])
        candidates = [R / 'configs' / a / (mid.replace('/', '--') + '.yaml') for a in arch]
        saved = {str(p): p.read_text() for p in candidates if p.is_file()}
        out.append(dict(model_id=mid, task=row.get('task'), status=row['status'], score=row.get('score'), weight_root=str(root), total_safetensors_stat_bytes=size, shard_count=len(files), own_config_architectures=arch, own_config_dtype=config.get('torch_dtype', config.get('dtype')), own_config_model_type=config.get('model_type'), saved_own_YAML=saved))
    except (OSError, ValueError) as exc:
        out.append(dict(model_id=mid, error_type=type(exc).__name__, temporary_resource_unreadability_not_confirmed_missing=True))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), only_unadapted_rows_small_own_checkpoint_metadata=True, no_weight_array_or_old_delivered_proof_read=True, candidates=sorted(out, key=lambda x: x.get('total_safetensors_stat_bytes', 10**20))), ensure_ascii=False))

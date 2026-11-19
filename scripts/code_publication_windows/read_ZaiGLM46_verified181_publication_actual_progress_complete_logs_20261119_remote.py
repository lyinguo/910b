import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
R = Path('/root/projects/comperror')
B = R / 'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0] = [str(R), str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
out = dict(UTC=datetime.now(timezone.utc).isoformat(), actual_processes=project_processes(), actual_FLOCK=runner_lock_owners(), files={})
homes = [B/'ZaiGLM46Flash_verified181_publication_review_inputs_v2_20261119', B/'ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119']
for h in homes:
    if not h.exists():
        out['files'][str(h)] = None
        continue
    out['files'][str(h)] = {'children': [p.name for p in sorted(h.iterdir())]}
    for p in sorted(h.iterdir()):
        if not p.is_file() or p.suffix not in ('.json', '.log'):
            continue
        raw = p.read_bytes()
        v = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
        if len(raw) <= 8*1024*1024:
            v['complete_text'] = raw.decode('utf8')
        else:
            v['requires_full_bounded_followup'] = True
        out['files'][str(p)] = v
print(json.dumps(out, ensure_ascii=False))

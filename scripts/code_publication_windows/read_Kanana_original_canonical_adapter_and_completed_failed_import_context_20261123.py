"""Read original canonical deployment and previous exact failure, no execution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
files=[R/'universal_eval/model/adapters/high_nine_kanana_canonical.py',
 R/'universal_eval/model/adapters/kanana_adapter.py',
 B/'Kanana_A100_BF16_SDPA_compat3_1322_20261110/kanana_A100_native_bf16_sdpa_20261110_adapter.py',
 B/'Kanana_A100_BF16_SDPA_compat3_1322_20261110/fresh3/model_stderr.log']
records={}
for p in files:
 if p.is_file():
  b=p.read_bytes();records[str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_text=b.decode('utf8',errors='replace'))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),complete_original_records=records,no_model_or_mutation=True),ensure_ascii=False))

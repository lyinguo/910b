import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path("/root/projects/comperror");B=R/"workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct";H=B/"GLM46Flash_completed_true3_complete_standard_Fortran_LZMA_storage_capacity_diagnostic_only_1714_20261118";L=B/"GLM46Flash_completed3_LZMA_diagnostic_actual_launch_1714_20261118"
rows={}
for f in [H/"review.json",H/"actual_completion.json",H/"actual_failure.json",L/"stderr.log"]:
 if f.exists():
  raw=f.read_bytes()
  try:v=json.loads(raw)
  except ValueError:v=raw.decode()
  if isinstance(v,dict) and "all_original_protected_SHA256" in v:v={k:q for k,q in v.items() if k!="all_original_protected_SHA256"}
  rows[str(f)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=v)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=rows,NPZ_stats=[dict(path=str(f),bytes=f.stat().st_size) for f in H.glob("*.npz")]),ensure_ascii=False))

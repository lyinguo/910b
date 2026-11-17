import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());H=Path(ptr['handoff_directory']);assert sha(H/'handoff.txt')=='76306766f495fef6777762f6b2ab9018582de15d7df1d826e1b02f1d1df58282'
assert sha(H/'facts.json')=='186789a3a946e2d8fc7c677a75d10ce9f3142057543a7d0fa3d0eda2072fa7af';facts=json.loads((H/'facts.json').read_bytes());exp=Path(facts['verified178_export']);assert sha(exp)==facts['verified178_export_SHA256'];o=dict(UTC=datetime.now(timezone.utc).isoformat(),pointer=ptr,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),export_path=str(exp),export_SHA256=sha(exp),export_base64=base64.b64encode(exp.read_bytes()).decode(),handoff_body_SHA256=sha(H/'handoff.txt'),facts_SHA256=sha(H/'facts.json'),closing_SHA256=sha(H/'closing_addendum.json'))
print(json.dumps(o,ensure_ascii=False))

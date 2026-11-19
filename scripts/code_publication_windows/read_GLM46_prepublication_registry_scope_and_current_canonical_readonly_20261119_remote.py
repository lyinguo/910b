import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,collections
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';MID='zai-org/GLM-4.6V-Flash';js=lambda p:json.loads(p.read_bytes())
reg=js(D/'registry.json');st=js(D/'state.json');paths=[D/'registry.json',D/'state.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),registry_count=len(reg),current_target_already_registered=MID in reg,current_target_registry=reg.get(MID),target_status=st['models'][MID]['status'],canonical_counts=dict(collections.Counter(v['status'] for v in st['models'].values())),canonical_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},no_NN_or_production_write=True),ensure_ascii=False))

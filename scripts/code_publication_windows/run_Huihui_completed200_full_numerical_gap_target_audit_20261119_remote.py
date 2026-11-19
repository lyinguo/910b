import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
p=R/'scripts/maintenance/audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='9b863f9e424dc84e5125f1748c6b0f174ec7f016ed57ce8546a3f186e1e01b6f'
from maintenance.audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119 import main
print(json.dumps(main('9167108780bb4ec4424680db43b7b9060b554cc242dc80efbae27797238a5d51'),ensure_ascii=False))

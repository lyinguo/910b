import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,importlib,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119')
try:result=m.apply('7e4f151c2dc6f2def1f63c20bcf71be1dc46697921e4479512e2eb065e4f31ed')
except Exception:
 (m.H/'apply_owner_reference_v2_failure_preserved.txt').write_text(traceback.format_exc(),encoding='utf8');raise
print(json.dumps(result,ensure_ascii=False))

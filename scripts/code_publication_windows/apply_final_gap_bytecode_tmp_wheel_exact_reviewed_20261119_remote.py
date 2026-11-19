import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib,json,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.cleanup_regenerable_bytecode_and_inactive_tmp_installers_exact_20261119')
try:result=m.apply('2b80d8dadcc3bb6570f18c034e77f7269b10526accbff5e7ceb9edbd7b013c95')
except Exception:
 (m.H/'actual_failure_preserved.txt').write_text(traceback.format_exc(),encoding='utf8');raise
print(json.dumps(result,ensure_ascii=False))

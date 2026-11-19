import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib,json,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
H=R/'workspace/diagnostics/batch_910b/user_optional_HTTP_cache_pause_exact_cleanup_1410_20261119'
m=importlib.import_module('maintenance.remove_inactive_auxiliary_HTTP_cache_exact_manifest_20261119')
try:
 result=m.run(R/'workspace/diagnostics/batch_910b/storage_attribution_detail_20261119_055009_078322/read_only_categories_complete.json','db2d45692c1f2fb082a04334303fbaf7c53fa484ab3794b5ab816e0c983a49ae',H,apply=True,reviewed_plan_sha256='d1350a1a3bb18eaec44824f81569e6eb31d15c60a365e14e7eb856a5f75750d1')
except Exception:
 (H/'actual_failure_preserved.txt').write_text(traceback.format_exc(),encoding='utf8');raise
print(json.dumps(result,ensure_ascii=False))

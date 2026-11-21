import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,unittest,tempfile
R=Path('/root/projects/comperror');H=R/'workspace/diagnostics/batch_910b/code_publication_source_layout_20261114/daily_20261122_actual_unit_checks_no_model'
H.mkdir(exist_ok=False);os.chdir(R);sys.path[:0]=[str(R),str(R/'scripts')]
assert hashlib.sha256((R/'scripts/publish_code_daily.py').read_bytes()).hexdigest()=='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
with tempfile.TemporaryDirectory(prefix='code_publication_day22_unit_') as temp:
 os.environ['TMPDIR']=temp;tempfile.tempdir=temp
 suite=unittest.defaultTestLoader.discover(str(R/'tests'))
 with (H/'unit_checks_R_actual.log').open('x',encoding='utf8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),exact_R_source=True,no_NN_or_canonical_or_active_job_change=True,log=str(H/'unit_checks_R_actual.log'),log_SHA256=hashlib.sha256((H/'unit_checks_R_actual.log').read_bytes()).hexdigest())
(H/'unit_checks_R_actual.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(value,ensure_ascii=False));sys.exit(not result.wasSuccessful())

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,unittest,os
from datetime import datetime,timezone
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
H=R/'workspace/diagnostics/batch_910b/code_publication_source_layout_20261114/daily_20261122_LlavaMini_private_cache_complete_value_observer_tests_no_learned_NN';H.mkdir(exist_ok=False)
suite=unittest.defaultTestLoader.discover(str(R/'tests'),pattern='test_llava_mini_own_original_private_cache_observer_20261122.py')
with (H/'unit_checks.log').open('x') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),only_CPU_unlearned_operands_and_original_AST_identity_no_learned_model_or_NPU=True,log=str(H/'unit_checks.log'),log_SHA256=hashlib.sha256((H/'unit_checks.log').read_bytes()).hexdigest())
(H/'unit_checks.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));print(json.dumps(value,ensure_ascii=False));sys.exit(not result.wasSuccessful())

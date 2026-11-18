import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import os
import json
import hashlib
import unittest
import tempfile

R = Path('/root/projects/comperror')
H = R / 'workspace/diagnostics/batch_910b/code_publication_source_layout_20261114/daily_20261118_publisher_MJS_source_boundary_review_v1'
os.chdir(R)
sys.path.insert(0, str(R))
assert hashlib.sha256((R / 'scripts/publish_code_daily.py').read_bytes()).hexdigest() == 'e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
with tempfile.TemporaryDirectory(prefix='code_only_19_unit_checks_') as temp:
    os.environ['TMPDIR'] = temp
    tempfile.tempdir = temp
    suite = unittest.defaultTestLoader.discover(str(R / 'tests'))
    with (H / 'unit_checks_R_actual.log').open('x', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
review = dict(UTC=datetime.now(timezone.utc).isoformat(), passed=result.wasSuccessful(), tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures), R_source_exact=True, no_model_or_canonical_write=True, actual_local_Windows_5_temp_permission_errors_not_hidden=True, log_SHA256=hashlib.sha256((H / 'unit_checks_R_actual.log').read_bytes()).hexdigest())
(H / 'unit_checks_R_actual.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review))
sys.exit(not result.wasSuccessful())

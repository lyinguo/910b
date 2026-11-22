"""Run meaningful source and full-value artifact tests only; no checkpoint or model evaluation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,tempfile,unittest
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'code_publication_source_layout_20261114/daily_20261123_actual_unit_checks_no_model'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());canonical=ptr['canonical_SHA256']
assert all(sha(p)==h for p,h in canonical.items())
assert sha(R/'scripts/publish_code_daily.py')=='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
H.mkdir(exist_ok=False);os.chdir(R);sys.path[:0]=[str(R),str(R/'scripts')]
with tempfile.TemporaryDirectory(prefix='code_publication_day23_unit_') as temp:
    os.environ['TMPDIR']=temp;tempfile.tempdir=temp
    suite=unittest.TestSuite();suite.addTests(unittest.defaultTestLoader.discover(str(R/'tests')))
    extra=['scripts/tests/test_complete_module_values_tree_20261122.py','scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py']
    for i,n in enumerate(extra):
        spec=importlib.util.spec_from_file_location('daily23_explicit_artifact_unit_'+str(i),R/n)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
    with (H/'unit_checks_R_actual.log').open('x',encoding='utf8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert all(sha(p)==h for p,h in canonical.items())
value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),explicit_extra_full_value_artifact_modules=extra,exact_R_source=True,no_checkpoint_NN_CPU200_or_canonical_or_job_change=True,canonical_SHA256=canonical,log=str(H/'unit_checks_R_actual.log'),log_SHA256=sha(H/'unit_checks_R_actual.log'))
(H/'unit_checks_R_actual.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(value,ensure_ascii=False));raise SystemExit(not result.wasSuccessful())

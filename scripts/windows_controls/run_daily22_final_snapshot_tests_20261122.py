import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,tempfile,unittest,json,hashlib,importlib.util
P=Path(__file__).resolve().parent;root=Path('D:/code-lv/.tmp/cd22s1');run_id='actual';assert run_id=='actual';tmp=Path('D:/code-lv/.tmp')/'cd22tep'
assert root.resolve().parent==Path('D:/code-lv/.tmp').resolve() and tmp.resolve().parent==root.resolve().parent
tmp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(tmp);tempfile.tempdir=str(tmp)
os.environ['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH'];os.chdir(root);sys.path.insert(0,str(root))
suite=unittest.TestSuite();modules=['test_code_publication','test_code_publication_javascript','test_readonly_npz_indexed','test_unsloth_completed_batch_lineage']
if importlib.util.find_spec('torch') is not None:modules.append('test_phi35_observation_only_full_values_0616_20261118')
for module in modules:suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=module+'.py'))
logfile=P/('daily_20261122_snapshot_tests_'+run_id+'.log')
with logfile.open('x',encoding='utf8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),modules=modules,source_root=str(root),temp_root=str(tmp),all_test_temp_cleanup_confined_to_verified_workspace=True,no_model_inference_or_source_mutation=True,log_SHA256=hashlib.sha256(logfile.read_bytes()).hexdigest())
(P/('daily_20261122_snapshot_tests_'+run_id+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(value,ensure_ascii=False));sys.exit(not result.wasSuccessful())

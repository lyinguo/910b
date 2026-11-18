import os,sys,pathlib,tempfile,unittest,json,hashlib
from datetime import datetime,timezone
P=pathlib.Path(__file__).parent;root=pathlib.Path('D:/code-lv/.tmp/cd18s');tmp=pathlib.Path('D:/code-lv/.tmp/cd18te')
tmp.mkdir(exist_ok=True);assert tmp.resolve().parent==pathlib.Path('D:/code-lv/.tmp').resolve() and root.resolve().parent==tmp.resolve().parent
os.environ['TEMP']=os.environ['TMP']=str(tmp);tempfile.tempdir=str(tmp);os.environ['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH'];os.chdir(root);sys.path.insert(0,str(root));suite=unittest.defaultTestLoader.discover('tests')
with (P/'daily_20261118_tests_actual.log').open('w',encoding='utf-8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),source_root=str(root),temp_root=str(tmp),all_recursive_test_temp_cleanup_confined_to_verified_workspace_directory=True,source_files_not_modified=True,log_SHA256=hashlib.sha256((P/'daily_20261118_tests_actual.log').read_bytes()).hexdigest())
(P/'daily_20261118_tests_actual.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(v));sys.exit(not result.wasSuccessful())

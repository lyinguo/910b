"""Stage useful exact-value sources and launch one guarded CPU-only storage review."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent
supervisor='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json,hashlib
R=Path('/root/projects/comperror');H=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM46Flash_complete_bitplane_storage_diagnostic_actual_launch_for_next_serial_20261119'
source=R/'scripts/diagnostic_glm46_completed3_full_bitplane_exact_sharing_next_serial_20261119.py'
start=datetime.now(timezone.utc).isoformat()
with (H/'diagnostic_stdout.json').open('xb') as out,(H/'diagnostic_stderr.log').open('xb') as err:
 result=subprocess.run(['python3','-B',str(source)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=result.returncode,diagnostic_only_no_NEW_model_or_full_NF=True)
(H/'actual_completion.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');raise SystemExit(result.returncode)
'''
ast.parse(supervisor);(P/'run_GLM46_complete_bitplane_storage_diagnostic_actual_20261119.py').write_text(supervisor,encoding='utf8',newline='\n')
names=[
 ('complete_bitplane_shared_npz_20261119.py','universal_eval/utils/complete_bitplane_shared_npz_20261119.py'),
 ('test_complete_bitplane_shared_npz_20261119.py','scripts/tests/test_complete_bitplane_shared_npz_20261119.py'),
 ('diagnostic_glm46_completed3_full_bitplane_exact_sharing_next_serial_20261119.py','scripts/diagnostic_glm46_completed3_full_bitplane_exact_sharing_next_serial_20261119.py'),
 ('run_GLM46_complete_bitplane_storage_diagnostic_actual_20261119.py','scripts/run_GLM46_complete_bitplane_storage_diagnostic_actual_20261119.py'),
 ('publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py','scripts/publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py'),
 ('prepare_Huihui_verified181_exact_original_writer_serial_publication_source_20261119.py','scripts/code_publication_windows/prepare_Huihui_verified181_exact_original_writer_serial_publication_source_20261119.py'),
 (Path(__file__).name,'scripts/code_publication_windows/'+Path(__file__).name),
]
rows=[]
for name,relative in names:
 raw=(P/name).read_bytes();ast.parse(raw);rows.append(dict(relative=relative,SHA256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
remote='FILES='+repr(rows)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');H=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM46Flash_complete_bitplane_storage_diagnostic_actual_launch_for_next_serial_20261119';H.mkdir(exist_ok=False)
for row in FILES:
 p=R/row['relative'];raw=base64.b64decode(row['data'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R);ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
test=R/'scripts/tests/test_complete_bitplane_shared_npz_20261119.py'
result=subprocess.run(['python3','-B',str(test)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CODEC_SCOPE_TEST_HOME=str(H)))
(H/'codec_scope_checks_stdout.json').write_text(result.stdout,encoding='utf8');(H/'codec_scope_checks_stderr.log').write_text(result.stderr,encoding='utf8');assert result.returncode==0,result.stderr
checks=json.loads(result.stdout);assert checks['passed'] and checks['complete_logical_arrays_roundtripped']==9
source=R/'scripts/run_GLM46_complete_bitplane_storage_diagnostic_actual_20261119.py'
with (H/'supervisor_stdout.log').open('xb') as out,(H/'supervisor_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(source)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,diagnostic_only_no_model_NN_or_full_NF=True,all_sources={x['relative']:x['SHA256'] for x in FILES},codec_checks=checks,source=str(source),live_Huihui_job_not_changed_and_no_original_runner_lock_taken=True)
(H/'actual_launch.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(value,ensure_ascii=False))
'''
ast.parse(remote);(P/'deploy_launch_GLM46_complete_bitplane_storage_diagnostic_next_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps({x['relative']:x['SHA256'] for x in rows},ensure_ascii=False))

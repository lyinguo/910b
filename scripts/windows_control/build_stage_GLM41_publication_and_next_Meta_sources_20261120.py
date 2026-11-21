from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
source='continue_MetaMllama_completed_genuine3_after_GLM41_verified184_dual_release_shaped_storage_v4_serial_20261120.py'
config='MetaMllama_completed_genuine3_after_GLM41_verified184_dual_release_shaped_storage_v4_serial_20261120.json'
names={
 'publish_GLM41Thinking_verified184_same_reserved_original_lock_dual_transaction_v1_20261120.py':'scripts/',
 'test_GLM41Thinking_reviewed_publication_contract_20261120.py':'scripts/tests/',
 source:'scripts/',config:'configs/queues/',
 'prepare_GLM41Thinking_verified184_publication_sources_20261120.py':'scripts/windows_control/',
 'prepare_GLM41Thinking_verified184_actual_completed200_plan_20261120.py':'scripts/windows_control/',
 'apply_GLM41Thinking_verified184_exact_server_bytes_local_CAS_20261120.py':'scripts/windows_control/',
 'build_GLM41Thinking_verified184_exact_local_receipt_upload_20261120.py':'scripts/windows_control/',
 'render_GLM41Thinking_verified184_actual_published_20261120.mjs':'scripts/windows_control/',
 'glm41thinking_workbook_cell_verified184_20261120.mjs':'scripts/windows_control/',
 'read_GLM41Thinking_verified184_server_book_for_local_CAS_20261120_remote.py':'scripts/maintenance/',
 'read_GLM41_verified183_publication_contract_20261120_remote.py':'scripts/maintenance/',
 'read_MetaMllama_prior183_approved_protocol_for_next_serial_20261120_remote.py':'scripts/maintenance/',
 'build_MetaMllama_after_GLM41_verified184_serial_v4_20261120.py':'scripts/windows_control/'}
assets=[]
for name,prefix in names.items():
 b=(V/name).read_bytes()
 if name.endswith('.py'):ast.parse(b)
 elif name.endswith('.json'):json.loads(b)
 assets.append(dict(relative=prefix+name,SHA256=sha(b),base64=base64.b64encode(b).decode()))
code='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for a in ASSETS:
 p=R/a['relative'];raw=base64.b64decode(a['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==a['SHA256']
 if p.suffix=='.py':ast.parse(raw)
 elif p.suffix=='.json':json.loads(raw)
 if p.exists():assert p.read_bytes()==raw,(str(p),'Existing source must not be overwritten')
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
home=B/'GLM41_prepared_reviewed_publication_and_Meta_next_serial_sources_20261120';home.mkdir(exist_ok=False)
with (home/'contract_tests_stdout.json').open('xb') as out,(home/'contract_tests_stderr.log').open('xb') as err:
 r=subprocess.run(['python3','-B',str(R/'scripts/tests/test_GLM41Thinking_reviewed_publication_contract_20261120.py')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
assert r.returncode==0,(home/'contract_tests_stderr.log').read_text()
tests=json.loads((home/'contract_tests_stdout.json').read_bytes());assert tests['passed']
source=R/SOURCE
with (home/'next_Meta_waiter_stdout.log').open('xb') as out,(home/'next_Meta_waiter_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(source)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
time.sleep(.4);assert p.poll() is None,(home/'next_Meta_waiter_stderr.log').read_text()
v=dict(UTC=datetime.now(timezone.utc).isoformat(),source_paths=[dict(relative=a['relative'],SHA256=a['SHA256']) for a in ASSETS],tests=tests,next_Meta_no_NN_waiter_PID=p.pid,next_Meta_source=str(source),only_after_same_GLM41_full200_qualification_dual_publication_and_lock_release=True,no_current_NN_source_job_or_parameter_change=True,no_G_or_canonical_write=True)
(home/'source_stage_and_noNN_launch_receipt.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''
payload='ASSETS='+repr(assets)+'\nSOURCE='+repr('scripts/'+source)+'\n'+code
ast.parse(payload);(V/'stage_GLM41_publication_and_next_Meta_sources_20261120_remote.py').write_text(payload,encoding='utf-8',newline='\n')
(V/'GLM41_publication_and_next_Meta_declared_primary_sources_20261120.json').write_text(json.dumps([dict(relative=a['relative'],SHA256=a['SHA256']) for a in assets],indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(assets=len(assets),no_execution_yet=True)))

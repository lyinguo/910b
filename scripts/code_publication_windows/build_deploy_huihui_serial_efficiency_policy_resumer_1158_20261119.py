"""Install reviewed serial controls and start only a capacity waiter, no NN yet."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent
paths={
 'serial_efficiency_user_priority_20261119.json':'configs/queues/serial_efficiency_user_priority_20261119.json',
 'serial_efficiency_dispatch_policy_20261119.py':'scripts/serial_efficiency_dispatch_policy_20261119.py',
 'test_serial_efficiency_dispatch_policy_20261119.py':'scripts/tests/test_serial_efficiency_dispatch_policy_20261119.py',
 'continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py':'scripts/continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py',
}
sources=[]
for name,relative in paths.items():
 b=(P/name).read_bytes()
 if name.endswith('.py'):ast.parse(b)
 else:json.loads(b)
 sources.append(dict(relative=relative,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
data=dict(sources=sources)
script='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,contextlib,hashlib,importlib.util,json,os,subprocess,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());assert {p:sha(p) for p in pointer['canonical_SHA256']}==pointer['canonical_SHA256']
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
for item in DATA['sources']:
 p=R/item['relative'];assert p.resolve().is_relative_to(R);b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['SHA256']
 if p.suffix=='.py':ast.parse(b)
 else:json.loads(b)
 assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
L=B/'Huihui_serial_efficiency_user_decision_policy_and_waiter_launch_1158_20261119';L.mkdir(exist_ok=False)
test=R/'scripts/tests/test_serial_efficiency_dispatch_policy_20261119.py';spec=importlib.util.spec_from_file_location('serial_efficiency_test',test);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with (L/'isolated_serial_scope_tests_stdout.log').open('x') as log,contextlib.redirect_stdout(log):m.run(R/'scripts/serial_efficiency_dispatch_policy_20261119.py',R/'configs/queues/serial_efficiency_user_priority_20261119.json')
source=R/'scripts/continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py';H=B/'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119';assert not H.exists()
with (L/'controller_stdout.log').open('xb') as out,(L/'controller_stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',str(source)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
launch=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=str(source),source_SHA256=sha(source),serial_only=True,max_active_models=1,only_capacity_waiter_started=True,actual_model_forward_calls=0,completed_true3_not_replayed=True,legacy_waiter_not_signalled_or_hot_edited=True,primary_sources=[{k:v for k,v in x.items() if k!='base64'} for x in DATA['sources']])
(L/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
for _ in range(30):
 if (H/'state.json').exists() or (H/'actual_completion.json').exists() or child.poll() is not None:break
 time.sleep(.2)
launch['current_state']=json.loads((H/'state.json').read_bytes()) if (H/'state.json').exists() else None
launch['actual_completion']=json.loads((H/'actual_completion.json').read_bytes()) if (H/'actual_completion.json').exists() else None
launch['process_returncode_if_exited']=child.poll();print(json.dumps(launch,ensure_ascii=False))
'''
ast.parse(script);(P/'deploy_launch_huihui_serial_efficiency_waiter_1158_20261119_remote.py').write_text(script,encoding='utf8',newline='\n')
(P/'Huihui_serial_efficiency_primary_sources_payload_1158_20261119.json').write_text(json.dumps(data),encoding='utf8')
print(json.dumps(dict(source_count=len(sources),reviewed_source_SHA256={x['relative']:x['SHA256'] for x in sources})))

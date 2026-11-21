from pathlib import Path
import ast, base64, hashlib, json
V=Path(__file__).resolve().parent
R='/root/projects/comperror'
B=R+'/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
source='reserve_GLM41_completed200_reviewed_dual_publication_original_lock_v2_actual_runner_directory_20261120.py'
parent=(V/'reserve_GLM41_completed200_reviewed_dual_publication_original_lock_v1_20261120.py').read_text(encoding='utf-8')
assert "assert runner and str(J/'job.json') in runner['argv']" in parent
updated=parent.replace("assert runner and str(J/'job.json') in runner['argv']", "assert runner and runner['startticks'] == 456181163 and runner['PPid'] == q['initial_controller_PID'] and runner['argv'] == ['python3', '-B', str(R/'scripts/A100_alignment_evidence_runner_20261103.py'), str(J)]")
raw=updated.encode();ast.parse(raw);(V/source).write_bytes(raw)
sha=lambda b:hashlib.sha256(b).hexdigest()
cfg=dict(model_id='zai-org/GLM-4.1V-9B-Thinking',
 job=B+'/GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120',
 job_SHA256='756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163',
 controller=B+'/GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120',
 controller_source_name='continue_GLM41Thinking_completed_true3_after_verified183_exact271_retention_wait_real_capacity_serial_v2_20261120.py',
 reservation_home=B+'/GLM41Thinking_verified184_reviewed_publication_window_v2_actual_runner_directory_after_same_formal200_original_EX_20261120',
 initial_runner_PID=1622920, initial_controller_PID=1618883, reservation_source_SHA256=sha(raw),
 no_NN_replay_no_signals_no_hot_source_or_job_change=True)
cfgname='GLM41Thinking_same_formal200_reviewed_dual_publication_original_lock_v2_actual_runner_directory_20261120.json'
cb=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();(V/cfgname).write_bytes(cb)
assets=[dict(relative='scripts/'+source,base64=base64.b64encode(raw).decode(),SHA256=sha(raw)),dict(relative='configs/publication/'+cfgname,base64=base64.b64encode(cb).decode(),SHA256=sha(cb))]
body='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import ast,base64,copy,hashlib,json,os,subprocess,tempfile,time
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for a in ASSETS:
 p=R/a['relative'];raw=base64.b64decode(a['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==a['SHA256']
 if p.suffix=='.py':ast.parse(raw)
 else:json.loads(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(raw)
source=R/ASSETS[0]['relative'];config=R/ASSETS[1]['relative']
tree=ast.parse(source.read_bytes());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='qualified');g={'Decimal':Decimal};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<qualification-admission>','exec'),g)
good=dict(passed=True,sample_count=200,actual_NPU_inference=True,historical_record_reuse=False,formal_alignment_within_5pct=True,original_protocol_final_qualification_complete=True,relative_error_to_current_A100_D=dict(server='.05',local='.05',preserved_original='.05'))
assert g['qualified'](good)
checks=['exact_5pct_full200_qualification_accepted']
for key in ['server','local','preserved_original']:
 bad=copy.deepcopy(good);bad['relative_error_to_current_A100_D'][key]='.0500000001';assert not g['qualified'](bad);checks.append(key+'_outside_rejected')
for key,value in [('sample_count',3),('passed',False),('original_protocol_final_qualification_complete',False),('historical_record_reuse',True)]:
 bad=copy.deepcopy(good);bad[key]=value;assert not g['qualified'](bad);checks.append(key+'_invalid_rejected')
home=B/'GLM41Thinking_reviewed_publication_window_v2_actual_runner_directory_admission_launch_20261120';home.mkdir(exist_ok=False)
# Real flock contention: the waiting process cannot enter until the owner releases.
import fcntl
lp=home/'isolated_test.lock';marker=home/'isolated_test_acquired.json'
with lp.open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 child_code="import fcntl,json,os,sys;from pathlib import Path;p=Path(sys.argv[1]);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX);Path(sys.argv[2]).write_text(json.dumps({'PID':os.getpid(),'acquired':True}))"
 c=subprocess.Popen(['python3','-B','-c',child_code,str(lp),str(marker)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 time.sleep(.4);assert c.poll() is None and not marker.exists()
 out,err=None,None
fcntl.flock if False else None
out,err=c.communicate(timeout=10);assert c.returncode==0 and json.loads(marker.read_bytes())['acquired']
checks.append('real_blocking_flock_serial_release_handoff')
(home/'isolated_admission_tests.json').write_text(json.dumps(dict(passed=True,test_count=len(checks),checks=checks,no_production_lock_or_model_used_in_tests=True),indent=2)+'\\n')
with (home/'reservation_stdout.log').open('xb') as out,(home/'reservation_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(source),str(config),ASSETS[1]['SHA256']],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
time.sleep(.5)
assert p.poll() is None,(home/'reservation_stderr.log').read_text()
result=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=str(source),source_SHA256=ASSETS[0]['SHA256'],config=str(config),config_SHA256=ASSETS[1]['SHA256'],isolated_tests_passed=len(checks),queued_for_same_original_EX_only_not_NN=True)
(home/'background_launch.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
'''
payload='ASSETS='+repr(assets)+'\n'+body
ast.parse(payload)
(V/'stage_and_launch_GLM41_publication_reservation_20261120_remote.py').write_text(payload,encoding='utf-8',newline='\n')
print(json.dumps(dict(source_SHA256=sha(raw),config_SHA256=sha(cb),assets=len(assets))))

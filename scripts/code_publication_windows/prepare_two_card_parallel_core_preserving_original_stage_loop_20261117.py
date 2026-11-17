import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64,copy
P=Path(__file__).parent;packet=json.loads((P/'R_actual_multi_NPU_topology_runner_storage_constraints_20261117.json').read_bytes());old=packet['sources']['scripts/A100_alignment_evidence_runner_20261103.py']['text'];assert hashlib.sha256(old.encode()).hexdigest()=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
helpers='''
_PARALLEL=None
def _parallel_bind(home,manifest):
 global _PARALLEL
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 manifest=pathlib.Path(manifest).resolve();m=json.loads(manifest.read_bytes());parent=m['coordinator_PID'];assert os.getppid()==parent and parent>1
 stat=(pathlib.Path('/proc')/str(parent)/'stat').read_text().split(') ',1)[1].split();assert int(stat[19])==m['coordinator_start_ticks']
 argv=(pathlib.Path('/proc')/str(parent)/'cmdline').read_bytes().split(bytes([0]));assert str(m['coordinator_source']).encode() in argv
 assert home.as_posix() in m['jobs'];row=m['jobs'][home.as_posix()];assert sha(home/'job.json')==row['job_SHA256'];assert sha(__file__)==m['worker_source_SHA256'];assert sha(m['coordinator_source'])==m['coordinator_source_SHA256']
 assert row['card'] in range(1,16);assert json.loads((home/'job.json').read_bytes())['spec']['npu_devices']==[row['card']]
 kernel_lock_owned(parent,D/'runner.lock');lockdir=D/'parallel_card_locks';lockdir.mkdir(exist_ok=True);p=lockdir/('NPU_'+str(row['card'])+'.lock');assert not p.is_symlink() and p.resolve().is_relative_to(D.resolve());handle=p.open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),p)
 _PARALLEL=dict(manifest=m,parent=parent,home=home,row=row,handle=handle,card_lock=p);return handle
def _parallel_assert_locks():
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 assert _PARALLEL;kernel_lock_owned(_PARALLEL['parent'],D/'runner.lock');kernel_lock_owned(os.getpid(),_PARALLEL['card_lock']);assert sha(_PARALLEL['home']/'job.json')==_PARALLEL['row']['job_SHA256']
def _parallel_admit(count,state,home,spec):
 _parallel_assert_locks();need=int(spec['parallel_storage_budget']['small3_bytes']);review=None
 if count==200:
  import zipfile
  a=state['accepted'][-1];assert a['sample_count']==3 and sha(a['proof'])==a['proof_sha256'];proof=json.loads(pathlib.Path(a['proof']).read_bytes());assert proof['passed'] and proof['sample_count']==3
  line=json.loads((pathlib.Path(a['report_dir'])/'operator_lineage.json').read_bytes())['actual_execution'];idx=json.loads(pathlib.Path(line['full_numeric_group_index']).read_bytes());parts=[];rawbytes=metabytes=0
  for x in idx['groups']:
   g=json.loads(pathlib.Path(x['path']).read_bytes());assert sha(x['path'])==x['SHA256'] and sha(g['archive'])==g['archive_SHA256'];storage=g['lossless_shared_tensor_storage'];assert storage['no_tensor_or_occurrence_or_numeric_check_dropped'] and storage['every_duplicate_compared_bitwise_to_actual_current_NPU_value']
   with zipfile.ZipFile(g['archive']) as z:members=[dict(name=i.filename,uncompressed=i.file_size,compressed=i.compress_size) for i in z.infolist()]
   assert len(members)==storage['physical_unique_arrays'];rawbytes+=sum(i['uncompressed'] for i in members);metabytes+=pathlib.Path(x['path']).stat().st_size;parts.append(dict(manifest=x,all_physical_unique_NPY_members=members,all_logical_occurrences=storage['all_actual_tensor_occurrences']))
  metabytes+=pathlib.Path(line['full_numeric_group_index']).stat().st_size;need=int((rawbytes+metabytes)*200/3*4+6*1024**3)
  review=dict(UTC=now(),real_fresh3_proof=a['proof'],real_fresh3_proof_SHA256=a['proof_sha256'],all_complete_NPY_parts=parts,all200_own_fixed_image_shape=[224,224],all_semantic_occurrences_and_values_retained=True,metadata_overhead_included=True,empirical_estimate_bytes=need,factor=4,extra_budget_bytes=6*1024**3,empirical_not_worst_case_guarantee=True,original_hard_log_precheck_and_batch_guards_unchanged=True);atomic(home/'actual_NEWfresh3_complete_storage_capacity_review.json',review)
 request=dict(UTC=now(),worker_PID=os.getpid(),card=_PARALLEL['row']['card'],count=count,budget_bytes=need,job_SHA256=sha(home/'job.json'),capacity_review=review);p=home/('parallel_stage_'+str(count)+'_request.json');assert not p.exists();atomic(p,request)
 grant=home/('parallel_stage_'+str(count)+'_grant.json');refuse=home/('parallel_stage_'+str(count)+'_refused.json')
 while not grant.exists():
  _parallel_assert_locks()
  if refuse.exists():raise RuntimeError('Independent stage capacity admission refused: '+refuse.read_text())
  state.update(phase='waiting_for_shared_project_capacity_admission',pending_sample_count=count,pending_budget_bytes=need);atomic(home/'state.json',dict(state,updated_at=now()));time.sleep(1)
 g=json.loads(grant.read_bytes());assert g['request_SHA256']==sha(p) and g['budget_bytes']==need and g['coordinator_PID']==_PARALLEL['parent'];_parallel_assert_locks()
def _parallel_nn_finished(count,home):
 _parallel_assert_locks();atomic(home/('parallel_stage_'+str(count)+'_NN_finished.json'),dict(UTC=now(),worker_PID=os.getpid(),original_full_report_exit_validated=True,numeric_CPU_only_log_budget_bytes=1024**3))
def _parallel_released(count,home,accepted):
 _parallel_assert_locks();atomic(home/('parallel_stage_'+str(count)+'_released.json'),dict(UTC=now(),worker_PID=os.getpid(),accepted=accepted,original_full_numeric_verifier_exit0=True))
'''
new=old.replace('"""Sole real runner-lock A100 alignment evidence runner. No Excel/canonical state/registry writes, takeover, signals or automatic main restart."""','"""Card-isolated stage loop under one real original global coordinator FLOCK; no writers."""\nimport sys\nsys.dont_write_bytecode=True')
new=new.replace('def main(home):',helpers+'\ndef main(home,coordinator_manifest):')
needle="lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');owned=None;save()";assert new.count(needle)==1;new=new.replace(needle,'lock=_parallel_bind(home,coordinator_manifest);owned=None;save()')
new=new.replace(";atomic(D/'A100_ALIGNMENT_STATUS.json',dict(state,job=str(home/'job.json'),state=str(home/'state.json')))",'')
new=new.replace("'remaining_new_models_not_started':True","'only_declared_parallel_peer_jobs_may_be_active':True")
start=new.index("  if pv['owner']=='explicit_user_stop_after_latest_test_20261103T0510':");end=new.index('  pause_sha=sha(pause);',start);new=new[:start]+"  assert pv['owner']==owner,'Parallel workers never rename or replace PAUSE'\n"+new[end:]
new=new.replace("kernel_lock_owned(os.getpid(),D/'runner.lock')",'_parallel_assert_locks()')
new=new.replace('  for count in stages:\n   guard()','  for count in stages:\n   _parallel_admit(count,state,home,spec)\n   guard()')
new=new.replace('validate_completed_exit(rc,report);guard()','validate_completed_exit(rc,report);guard();_parallel_nn_finished(count,home)')
new=new.replace("state['accepted'].append(accepted);save()","state['accepted'].append(accepted);save();_parallel_released(count,home,accepted)")
new=new.replace("else:assert len(sys.argv)==2;main(sys.argv[1])","else:assert len(sys.argv)==3;main(sys.argv[1],sys.argv[2])")
ast.parse(new);(P/'A100_alignment_parallel_stage_worker_20261117.py').write_bytes(new.encode())
# The executable model/verifier argv, waits, report validation and all numeric
# acceptance assertions stay byte-identical inside the original for-loop.
for literal in ["cmd=['python3','run.py','--config',str(config.relative_to(R))]","cmd=['python3',spec['verifier'],str(report),str(count)]","assert vrc==0,'Independent verification failed; retain original reports and all failure evidence'","proof['historical_record_reuse'] is False","raise TimeoutError('Owned process finished after approved deadline; do not accept automatically')"]:assert literal in new
control='''"""Pure parallel admission rules plus real process/kernel-lock isolation checks."""
from pathlib import Path
import os,json,hashlib,fcntl,time
GiB=1024**3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def can_admit(free,active,budget,max_parallel):
 assert isinstance(free,int) and free>=0 and isinstance(budget,int) and budget>0
 assert all(isinstance(v,int) and v>0 for v in active.values())
 return len(active)<max_parallel and sum(active.values())+budget<=free
def validate_cards(rows):
 cards=[x['card'] for x in rows];assert cards and len(cards)==len(set(cards)) and all(isinstance(x,int) and 1<=x<=15 for x in cards);return cards
def proc_start(pid):return int((Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19])
def real_exclusive_owner(pid,path):
 st=Path(path).stat();found=[]
 for fd in (Path('/proc')/str(pid)/'fd').iterdir():
  try:
   s=fd.stat();text=(fd.parent.parent/'fdinfo'/fd.name).read_text()
  except OSError:continue
  if (s.st_dev,s.st_ino)==(st.st_dev,st.st_ino) and 'FLOCK' in text and 'WRITE' in text:found.append(str(fd))
 assert found;return found
'''
master='''"""One original global FLOCK, disjoint card workers, shared-disk stage reservations."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,fcntl,subprocess,time,traceback,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R/'scripts'),str(R)]
from A100_alignment_parallel_control_20261117 import sha,can_admit,validate_cards,proc_start,real_exclusive_owner
from A100_alignment_evidence_runner_20261103 import scope,unchanged
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(tmp,p)
def main(home):
 home=Path(home).resolve();assert home.is_relative_to(D/'A100_relative5pct_alignment_20261103');q=js(home/'batch.json');rows=q['jobs'];validate_cards(rows);assert q['max_parallel_stages']==2 and len(rows)<=2
 assert sha(R/'scripts/A100_alignment_evidence_runner_20261103.py')=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 decision=js(q['latest_user_parallel_request']);assert sha(q['latest_user_parallel_request'])==q['latest_user_parallel_request_SHA256'] and decision['origin']=='actual_user_chat_instruction' and '多卡' in decision['exact_user_request'] and '并行' in decision['exact_user_request']
 unchanged(q['protected_canonical_hashes']);assert not project_processes() and not runner_lock_owners();assert js(D/'PAUSE')['owner']=='A100_relative5pct_alignment_20261103_explicit_user_resume'
 import batch_910b as batch
 for row in rows:
  job=js(Path(row['home'])/'job.json');assert sha(Path(row['home'])/'job.json')==row['job_SHA256'] and job['spec']['npu_devices']==[row['card']];assert job['protected_canonical_hashes']==q['protected_canonical_hashes'];scope(job['spec'],job['stages'],False);batch.check_sources(job['spec']);assert batch.device_idle(job['spec'])
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);real_exclusive_owner(os.getpid(),D/'runner.lock');lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush();os.fsync(lock.fileno());children={};active={};completed={};grants={};handled=set()
 manifest=dict(coordinator_PID=os.getpid(),coordinator_start_ticks=proc_start(os.getpid()),coordinator_source=__file__,coordinator_source_SHA256=sha(__file__),worker_source_SHA256=sha(q['worker_source']),jobs={row['home']:row for row in rows},one_original_global_exclusive_FLOCK=True,publishers_must_wait_for_all_workers_and_global_lock_release=True);mf=home/'coordinator_manifest.json';assert not mf.exists();put(mf,manifest)
 try:
  unchanged(q['protected_canonical_hashes'])
  for row in rows:
   path=Path(row['home']);out=(path/'parallel_worker_stdout.log').open('xb');err=(path/'parallel_worker_stderr.log').open('xb');p=subprocess.Popen(['python3','-B',q['worker_source'],row['home'],str(mf)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'));children[row['home']]=(p,out,err);put(path/'parallel_actual_launch.json',dict(UTC=now(),PID=p.pid,PPid=os.getpid(),card=row['card'],job_SHA256=row['job_SHA256'],coordinator_manifest=str(mf),coordinator_manifest_SHA256=sha(mf)))
  while len(completed)<len(rows):
   unchanged(q['protected_canonical_hashes']);real_exclusive_owner(os.getpid(),D/'runner.lock')
   for row in rows:
    name=row['home'];p,out,err=children[name];path=Path(name)
    if p.poll() is not None and name not in completed:
     completed[name]=dict(actual_returncode=p.wait(),UTC=now(),card=row['card']);active.pop(name,None);out.close();err.close()
    if name in active:
     count=grants[name]['count']
     if (path/('parallel_stage_'+str(count)+'_released.json')).exists():active.pop(name,None)
     elif (path/('parallel_stage_'+str(count)+'_NN_finished.json')).exists():active[name]=min(active[name],1024**3)
   for row in rows:
    name=row['home'];path=Path(name)
    if name in completed or name in active:continue
    for count in js(path/'job.json')['stages']:
     request=path/('parallel_stage_'+str(count)+'_request.json');key=(name,count)
     if not request.exists() or key in handled:continue
     req=js(request);assert req['job_SHA256']==row['job_SHA256'] and req['card']==row['card'] and req['worker_PID']==children[name][0].pid;free=shutil.disk_usage(D).free;need=req['budget_bytes']
     if can_admit(free,active,need,q['max_parallel_stages']):
      grant=dict(UTC=now(),coordinator_PID=os.getpid(),request_SHA256=sha(request),budget_bytes=need,count=count,card=row['card'],free_at_admission=free,other_active_reservations=dict(active));put(path/('parallel_stage_'+str(count)+'_grant.json'),grant);active[name]=need;grants[name]=grant;handled.add(key)
     elif not active and need>free:
      refusal=dict(UTC=now(),budget_bytes=need,actual_free_bytes=free,count=count,other_jobs_may_continue=True,no_model_inference_started_for_refused_stage=True);put(path/('parallel_stage_'+str(count)+'_refused.json'),refusal);put(home/'NOTIFY_capacity_error.json',dict(refusal,model_id=js(path/'job.json')['model_id'],chat_notification_not_claimed=True));handled.add(key)
     break
   current=dict(UTC=now(),PID=os.getpid(),phase='coordinating_real_disjoint_card_stages',active_reservations=active,grants=grants,completed=completed,jobs=rows,one_original_global_FLOCK_held=True,no_G_state_registry_or_workbook_write=True);put(home/'state.json',current);put(D/'A100_ALIGNMENT_PARALLEL_STATUS.json',dict(current,batch=str(home),manifest=str(mf)));time.sleep(.5)
  unchanged(q['protected_canonical_hashes']);rc=0 if all(x['actual_returncode']==0 for x in completed.values()) else 1;put(home/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,jobs=completed,no_G_or_canonical_write=True,all_actually_owned_workers_naturally_exited=True));return rc
 except BaseException:
  # No cancellation/signals: keep the real original lock while any owned stage
  # worker is still alive. A source/canonical failure remains an actual failure.
  for p,out,err in children.values():
   p.wait();out.close();err.close()
  put(home/'actual_failure.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),owned_children_all_naturally_exited=True));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
'''
tests='''"""Real temporary-kernel FLOCK/concurrency and reservation tests; never NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import tempfile,fcntl,subprocess,time,json,os,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from A100_alignment_parallel_control_20261117 import can_admit,validate_cards,real_exclusive_owner
def main():
 assert can_admit(100,{},60,2) and can_admit(100,{'a':60},40,2) and not can_admit(100,{'a':60},41,2) and not can_admit(100,{'a':10,'b':10},10,2)
 for rows in [[dict(card=0)],[dict(card=2),dict(card=2)],[dict(card=16)]]:
  try:validate_cards(rows)
  except AssertionError:pass
  else:raise AssertionError('Unsafe cards admitted')
 assert validate_cards([dict(card=2),dict(card=4)])==[2,4]
 with tempfile.TemporaryDirectory(prefix='parallel_real_lock_test_') as root:
  root=Path(root);globalfile=root/'global.lock';held=globalfile.open('a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);real_exclusive_owner(os.getpid(),globalfile)
  child=root/'child.py';child.write_text("import sys,fcntl,time,json,os\\nfrom pathlib import Path\\nr=Path(sys.argv[1]);card=sys.argv[2];f=(r/(card+'.lock')).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\\nw=(r/'global.lock').open('a+')\\ntry:fcntl.flock(w,fcntl.LOCK_EX|fcntl.LOCK_NB)\\nexcept BlockingIOError:pass\\nelse:raise AssertionError('Publisher lock incorrectly released')\\n(r/(card+'.ready')).write_text(str(os.getpid()))\\nwhile not (r/'go').exists():time.sleep(.01)\\na=time.monotonic();time.sleep(.5);(r/(card+'.json')).write_text(json.dumps([a,time.monotonic()]))\\n".replace('\\n',chr(10)))
  p1=subprocess.Popen([sys.executable,'-B',str(child),str(root),'2']);p2=subprocess.Popen([sys.executable,'-B',str(child),str(root),'4']);deadline=time.monotonic()+10
  while not ((root/'2.ready').exists() and (root/'4.ready').exists()):assert time.monotonic()<deadline;time.sleep(.01)
  duplicate=(root/'2.lock').open('a+')
  try:fcntl.flock(duplicate,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Same card double reservation admitted')
  (root/'go').touch();assert p1.wait()==p2.wait()==0;a=json.loads((root/'2.json').read_text());b=json.loads((root/'4.json').read_text());assert min(a[1],b[1])>max(a[0],b[0]);overlap=min(a[1],b[1])-max(a[0],b[0]);fcntl.flock(held,fcntl.LOCK_UN);held.close();fcntl.flock(duplicate,fcntl.LOCK_EX|fcntl.LOCK_NB);duplicate.close()
 print(json.dumps(dict(passed=True,test_count=12,real_disjoint_card_lock_holders_overlapped_seconds=overlap,global_publisher_exclusion_verified=True,same_card_duplicate_refused=True,conservative_combined_storage_and_max_parallel_gates_verified=True,production_locks_and_NPU_and_canonical_files_never_touched=True)))
if __name__=='__main__':main()
'''
payload={}
for n,s in [('scripts/A100_alignment_parallel_control_20261117.py',control),('scripts/A100_alignment_parallel_stage_worker_20261117.py',new),('scripts/A100_alignment_parallel_coordinator_20261117.py',master),('scripts/tests/test_A100_alignment_parallel_control_20261117.py',tests)]:ast.parse(s);(P/Path(n).name).write_bytes(s.encode());payload[n]=dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest())
review=dict(original_runner_SHA256=hashlib.sha256(old.encode()).hexdigest(),original_runner_file_not_modified=True,original_model_and_verifier_launch_wait_and_report_numeric_acceptance_code_preserved=True,orchestration_changes_only=['one real parent owns original exclusive lock','worker real per-card lock','immutable parent/job/source/PID binding','per-job status instead of global status races','unchanged original PAUSE owner only','shared project storage phase admission','single-model TP/math/dtype/inputs untouched'],source_SHA256={n:v['SHA256'] for n,v in payload.items()});(P/'two_card_parallel_core_source_review_20261117.json').write_bytes(json.dumps(review,indent=2).encode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,ast,subprocess,os
R=Path('/root/projects/comperror');FILES=__FILES__
for n,v in FILES.items():
 p=R/n;raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=subprocess.run(['python3','-B','scripts/tests/test_A100_alignment_parallel_control_20261117.py'],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_test_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_SHA256={n:v['SHA256'] for n,v in FILES.items()},no_production_NPU_or_lock_used=True),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__FILES__',repr(payload));ast.parse(remote);(P/'deploy_test_two_card_parallel_core_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(review,indent=2))

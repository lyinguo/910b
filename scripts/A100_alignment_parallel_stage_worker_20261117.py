"""Card-isolated stage loop under one real original global coordinator FLOCK; no writers."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,os,sys,fcntl,datetime,hashlib,subprocess,time,copy,traceback,tempfile
R=pathlib.Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
sys.path.insert(0,str(R/'scripts'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,x):
 p=pathlib.Path(p);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(q,p)
def scope(spec,stages,write):
 assert write is False,'This controller never writes an existing or blank G'
 assert stages==[3] or stages==[3,200],'Every formal200 must be preceded by this controller fresh3'
 if stages==[3,200]:assert spec.get('approved') is True and spec.get('alignment_approval_review') and spec.get('alignment_approval_review_sha256')
 for n in spec['source_sha256']:
  p=pathlib.PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
 for n in [spec['config'],spec['verifier'],'run.py','universal_eval/orchestrator/pipeline.py']:assert n in spec['source_sha256']
def unchanged(files):
 for p,h in files.items():assert sha(p)==h,'Protected canonical or source changed: '+p
def selftest():
 spec={'approved':False,'config':'configs/model.yaml','verifier':'scripts/verify.py','source_sha256':{n:'x' for n in ['configs/model.yaml','scripts/verify.py','run.py','universal_eval/orchestrator/pipeline.py']}}
 scope(spec,[3],False);count=1
 for stages,write in [([200],False),([3,199],False),([3],True),([3,200],False)]:
  try:scope(spec,stages,write)
  except AssertionError:count+=1
  else:raise AssertionError('Unsafe scope accepted')
 for n in ['/mnt/afs/readonly.py','../escape.py']:
  x=copy.deepcopy(spec);x['source_sha256'][n]='x'
  try:scope(x,[3],False)
  except AssertionError:count+=1
  else:raise AssertionError('Unsafe snapshot path accepted')
 with tempfile.TemporaryDirectory(prefix='A100_alignment_safety_') as root:
  p=pathlib.Path(root)/'lock';p.touch()
  with p.open('a+') as a,p.open('a+') as b:
   fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
   try:fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:count+=1
   else:raise AssertionError('Duplicate controller lock accepted')
  p.write_text('original');h=sha(p);unchanged({str(p):h});p.write_text('changed')
  try:unchanged({str(p):h})
  except AssertionError:count+=1
  else:raise AssertionError('Protected change accepted')
 print(json.dumps({'passed':True,'test_count':count,'tests':'Unapproved3 only, fresh3-before200, no writer, relative snapshot path, duplicate FLOCK refusal, exact protected hash refusal','no_production_lock_or_file_changed':True}))

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

def main(home,coordinator_manifest):
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_completed_exit
 from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned,children
 home=pathlib.Path(home).resolve();assert home.is_relative_to(D/'A100_relative5pct_alignment_20261103')
 job=json.loads((home/'job.json').read_text());spec=job['spec'];stages=job['stages'];scope(spec,stages,job['write_workbook']);assert job['operation']=='A100_D_column_alignment_evidence_only' and job['no_takeover_no_signals_no_main_restart'] is True
 request=pathlib.Path(job['user_request_receipt']);assert sha(request)==job['user_request_receipt_sha256'];decision=json.loads(request.read_text());assert decision['origin']=='actual_user_chat_instruction' and 'A100' in decision['exact_user_request'] and '重新适配' in decision['exact_user_request']
 if 200 in stages:
  p=pathlib.Path(spec['alignment_approval_review']);assert sha(p)==spec['alignment_approval_review_sha256'];review=json.loads(p.read_text());assert review['approved'] and review['real_NPU3_independent_proof_verified'] and review['numerical_review_verified']
 baseline=job['protected_canonical_hashes'];core_unchanged();unchanged(baseline);batch.check_sources(spec);unchanged({str(R/n):h for n,h in job['runner_source_sha256'].items()})
 state={'PID':os.getpid(),'operation':job['operation'],'phase':'preflight','current':job['model_id'],'child_PID':None,'accepted':[],'started_at':now(),'canonical_state_registry_Excel_unchanged':True,'only_declared_parallel_peer_jobs_may_be_active':True}
 def save():
  state['updated_at']=now();atomic(home/'state.json',state)
 lock=_parallel_bind(home,coordinator_manifest);owned=None;save()
 try:
  canonical=json.loads((D/'state.json').read_text());assert canonical['current'] is None
  assert not children(os.getpid())
  pause=D/'PAUSE';pv=json.loads(pause.read_text());owner='A100_relative5pct_alignment_20261103_explicit_user_resume'
  assert pv['owner']==owner,'Parallel workers never rename or replace PAUSE'
  pause_sha=sha(pause);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush();os.fsync(lock.fileno())
  entry=canonical['models'][job['model_id']]
  def guard():
   core_unchanged();unchanged(baseline);batch.check_sources(spec);assert sha(pause)==pause_sha;_parallel_assert_locks()
  def wait(child,limit):
   start=time.monotonic();warning=False
   while child.poll() is None:
    state['child_PID']=child.pid
    if time.monotonic()-start>limit:
     state['timeout_review_required_no_signal']=True;warning=True
    save();time.sleep(3)
   state['child_PID']=None;save()
   if warning:raise TimeoutError('Owned process finished after approved deadline; do not accept automatically')
   return child.wait()
  for count in stages:
   _parallel_admit(count,state,home,spec)
   guard()
   while not batch.device_idle(spec):state['phase']='waiting_for_original_device_idle';save();time.sleep(10);guard()
   config,effective=batch.config_for(spec,copy.deepcopy(entry),count);assert effective['sample_range']=='0:'+str(count-1)
   folder=R/'workspace/reports'/job['model_id'].replace('/','_')/entry['task'];folder.mkdir(parents=True,exist_ok=True);before=set(folder.iterdir());attempt=home/('fresh3' if count==3 else 'formal200');attempt.mkdir()
   (attempt/'effective_config.yaml').write_bytes(config.read_bytes());start=time.time();env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',MODELSCOPE_OFFLINE='1',EVAL_OFFLINE_ONLY='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
   state.update(phase='running_fresh3' if count==3 else 'running_formal200',sample_count=count,attempt=str(attempt));save()
   with (attempt/'console.log').open('wb') as f:
    cmd=['python3','run.py','--config',str(config.relative_to(R))];child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);owned=child;state['model_child_argv']=cmd;rc=wait(child,spec['smoke_timeout_seconds'] if count==3 else spec['formal_timeout_seconds']);owned=None
   new=[p for p in folder.iterdir() if p not in before and p.is_dir()];assert len(new)==1;report=new[0];state.update(report_dir=str(report),model_returncode=rc,phase='independent_CPU_verification');save();validate_completed_exit(rc,report);guard();_parallel_nn_finished(count,home)
   with (attempt/'verification_stdout.json').open('wb') as out,(attempt/'verification_stderr.log').open('wb') as err:
    cmd=['python3',spec['verifier'],str(report),str(count)];child=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=err,start_new_session=True);owned=child;state['verifier_child_argv']=cmd;vrc=wait(child,spec['verify_timeout_seconds']);owned=None
   state['verifier_returncode']=vrc;save();assert vrc==0,'Independent verification failed; retain original reports and all failure evidence'
   proof=json.loads((attempt/'verification_stdout.json').read_text());score=validate_report(R,report,job['model_id'],entry['task'],count,proof,fresh_after=start);assert proof['passed'] and proof['historical_record_reuse'] is False;guard()
   accepted={'sample_count':count,'score':score,'report_dir':str(report),'proof':str(attempt/'verification_stdout.json'),'proof_sha256':sha(attempt/'verification_stdout.json'),'new_real_NPU_inference':True,'no_G_or_canonical_state_registry_write':True,'relative_error_to_A100_table_D':proof['relative_error_to_A100_table_D'],'formal_alignment_within_5pct':proof['formal_alignment_within_5pct']};atomic(attempt/'accepted.json',accepted);state['accepted'].append(accepted);save();_parallel_released(count,home,accepted)
  guard();state.update(phase='completed_evidence_only',completed_at=now(),next_requires_numerical_review_and_approved_fresh3_then200=stages==[3]);save();atomic(home/'completion.json',dict(state,actual_returncode=0,protected_canonical_hashes_verified=True))
 except BaseException:
  # Never release the sole lock while any actually owned model/verifier is live.
  if owned is not None:
   while owned.poll() is None:time.sleep(3)
   state['child_PID']=None
  state.update(phase='failed_evidence_preserved',error=traceback.format_exc(),completed_at=now());save();atomic(home/'completion.json',dict(state,actual_returncode=1));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='--self-test':selftest()
 else:assert len(sys.argv)==3;main(sys.argv[1],sys.argv[2])

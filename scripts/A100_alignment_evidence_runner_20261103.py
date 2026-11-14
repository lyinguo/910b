"""Sole real runner-lock A100 alignment evidence runner. No Excel/canonical state/registry writes, takeover, signals or automatic main restart."""
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
def main(home):
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_completed_exit
 from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned,children
 home=pathlib.Path(home).resolve();assert home.is_relative_to(D/'A100_relative5pct_alignment_20261103')
 job=json.loads((home/'job.json').read_text());spec=job['spec'];stages=job['stages'];scope(spec,stages,job['write_workbook']);assert job['operation']=='A100_D_column_alignment_evidence_only' and job['no_takeover_no_signals_no_main_restart'] is True
 request=pathlib.Path(job['user_request_receipt']);assert sha(request)==job['user_request_receipt_sha256'];decision=json.loads(request.read_text());assert decision['origin']=='actual_user_chat_instruction' and 'A100' in decision['exact_user_request'] and '重新适配' in decision['exact_user_request']
 if 200 in stages:
  p=pathlib.Path(spec['alignment_approval_review']);assert sha(p)==spec['alignment_approval_review_sha256'];review=json.loads(p.read_text());assert review['approved'] and review['real_NPU3_independent_proof_verified'] and review['numerical_review_verified']
 baseline=job['protected_canonical_hashes'];core_unchanged();unchanged(baseline);batch.check_sources(spec);unchanged({str(R/n):h for n,h in job['runner_source_sha256'].items()})
 state={'PID':os.getpid(),'operation':job['operation'],'phase':'preflight','current':job['model_id'],'child_PID':None,'accepted':[],'started_at':now(),'canonical_state_registry_Excel_unchanged':True,'remaining_new_models_not_started':True}
 def save():
  state['updated_at']=now();atomic(home/'state.json',state);atomic(D/'A100_ALIGNMENT_STATUS.json',dict(state,job=str(home/'job.json'),state=str(home/'state.json')))
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');owned=None;save()
 try:
  canonical=json.loads((D/'state.json').read_text());assert canonical['current'] is None
  assert not children(os.getpid())
  pause=D/'PAUSE';pv=json.loads(pause.read_text());owner='A100_relative5pct_alignment_20261103_explicit_user_resume'
  if pv['owner']=='explicit_user_stop_after_latest_test_20261103T0510':
   assert sha(pause)==job['initial_user_stop_PAUSE_sha256'];dest=home/'previous_user_stop_PAUSE.json';assert not dest.exists();os.rename(pause,dest)
   fd=os.open(pause,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
   with os.fdopen(fd,'w') as f:json.dump({'owner':owner,'created_at_UTC':now(),'reason':'User resumed A100 relative-error alignment first; do not start remaining models until <=5% gate and protocol reviews pass','request':str(request),'previous_user_stop_receipt_retained':str(dest),'existing_G_overwrite_authorized':False},f,ensure_ascii=False,indent=2)
  else:assert pv['owner']==owner,'Never change another PAUSE owner'
  pause_sha=sha(pause);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush();os.fsync(lock.fileno())
  entry=canonical['models'][job['model_id']]
  def guard():
   core_unchanged();unchanged(baseline);batch.check_sources(spec);assert sha(pause)==pause_sha;kernel_lock_owned(os.getpid(),D/'runner.lock')
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
   guard()
   while not batch.device_idle(spec):state['phase']='waiting_for_original_device_idle';save();time.sleep(10);guard()
   config,effective=batch.config_for(spec,copy.deepcopy(entry),count);assert effective['sample_range']=='0:'+str(count-1)
   folder=R/'workspace/reports'/job['model_id'].replace('/','_')/entry['task'];folder.mkdir(parents=True,exist_ok=True);before=set(folder.iterdir());attempt=home/('fresh3' if count==3 else 'formal200');attempt.mkdir()
   (attempt/'effective_config.yaml').write_bytes(config.read_bytes());start=time.time();env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',MODELSCOPE_OFFLINE='1',EVAL_OFFLINE_ONLY='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
   state.update(phase='running_fresh3' if count==3 else 'running_formal200',sample_count=count,attempt=str(attempt));save()
   with (attempt/'console.log').open('wb') as f:
    cmd=['python3','run.py','--config',str(config.relative_to(R))];child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);owned=child;state['model_child_argv']=cmd;rc=wait(child,spec['smoke_timeout_seconds'] if count==3 else spec['formal_timeout_seconds']);owned=None
   new=[p for p in folder.iterdir() if p not in before and p.is_dir()];assert len(new)==1;report=new[0];state.update(report_dir=str(report),model_returncode=rc,phase='independent_CPU_verification');save();validate_completed_exit(rc,report);guard()
   with (attempt/'verification_stdout.json').open('wb') as out,(attempt/'verification_stderr.log').open('wb') as err:
    cmd=['python3',spec['verifier'],str(report),str(count)];child=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=err,start_new_session=True);owned=child;state['verifier_child_argv']=cmd;vrc=wait(child,spec['verify_timeout_seconds']);owned=None
   state['verifier_returncode']=vrc;save();assert vrc==0,'Independent verification failed; retain original reports and all failure evidence'
   proof=json.loads((attempt/'verification_stdout.json').read_text());score=validate_report(R,report,job['model_id'],entry['task'],count,proof,fresh_after=start);assert proof['passed'] and proof['historical_record_reuse'] is False;guard()
   accepted={'sample_count':count,'score':score,'report_dir':str(report),'proof':str(attempt/'verification_stdout.json'),'proof_sha256':sha(attempt/'verification_stdout.json'),'new_real_NPU_inference':True,'no_G_or_canonical_state_registry_write':True,'relative_error_to_A100_table_D':proof['relative_error_to_A100_table_D'],'formal_alignment_within_5pct':proof['formal_alignment_within_5pct']};atomic(attempt/'accepted.json',accepted);state['accepted'].append(accepted);save()
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
 else:assert len(sys.argv)==2;main(sys.argv[1])

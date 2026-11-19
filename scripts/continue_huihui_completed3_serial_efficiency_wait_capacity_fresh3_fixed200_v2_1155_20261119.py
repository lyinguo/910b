"""User-prioritized serial continuation of already accepted own Huihui NPU3.
No true3 replay, live-source edits, legacy signals, parallel NN, or canonical writes.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,tempfile,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated';READY=B/'HuihuiGLM46Flash_own_original_BF16_2048_native_source_readiness_v1_20261118'
H=B/'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119'
J3=B/'HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118'
J200=B/'HuihuiGLM46_completed_true3_single_model_serial_approved_NEWfresh3_fixed200_v2_real_FLOCK_1155_20261119'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';SELF='scripts/continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py'
CFG='configs/Glm4vForConditionalGeneration/huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-serial-approved-first200-v2-1155-20261119.yaml'
PROTO='configs/protocols/huihui_glm46flash_own_original_BF16_2048_serial_approved_first200_v2_1155_20261119.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
SERIAL_CONFIG=R/'configs/queues/serial_efficiency_user_priority_20261119.json'
SERIAL_MODULE=R/'scripts/serial_efficiency_dispatch_policy_20261119.py'
LEGACY_SOURCE=R/'scripts/continue_glm46flash_completed_true3_wait_actual_capacity_resume_fresh3_fixed200_v3_1723_20261118.py'
LEGACY_HOME=B/'GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118'
LEGACY_JOB=B/'GLM46Flash_completed_true3_actual_capacity_restored_approved_fresh3_fixed200_v3_real_FLOCK_1723_20261118'
LEGACY_BUDGET=235720910370

def existing_legacy_dispatch():
 assert sha(LEGACY_SOURCE)=='96dfdd09b7b14a4845e142cbcc6dcc4384130a362ff1bea777f9fcf2a059637b'
 state=js(LEGACY_HOME/'state.json');pid=state['PID'];proc=Path('/proc')/str(pid);alive=False;detail=dict(PID=pid)
 if proc.is_dir():
  argv=(proc/'cmdline').read_bytes().split(b'\0');cwd=os.readlink(proc/'cwd')
  alive=str(LEGACY_SOURCE).encode() in argv and cwd=='/root'
  detail.update(actual_argv=[v.decode(errors='replace') for v in argv if v],actual_cwd=cwd,actual_stat=(proc/'stat').read_text())
 pending=bool(alive and (LEGACY_JOB/'job.json').exists() and not (LEGACY_JOB/'completion.json').exists())
 detail.update(bound_alive=alive,actual_job_pending=pending,legacy_actual_completion=js(LEGACY_HOME/'actual_completion.json') if (LEGACY_HOME/'actual_completion.json').exists() else None)
 return alive,pending,detail

def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from batch_910b_support import validate_report
 import batch_910b as batch
 from serial_efficiency_dispatch_policy_20261119 import validate_policy,choose_action
 scheduling=js(SERIAL_CONFIG);validate_policy(scheduling);serial_hashes={str(SERIAL_CONFIG):sha(SERIAL_CONFIG),str(SERIAL_MODULE):sha(SERIAL_MODULE)}
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 def active():return [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 def run(job,stage):
  value=js(job/'job.json');batch.check_sources(value['spec']);unchanged(value['protected_canonical_hashes']);assert not active() and not runner_lock_owners() and batch.device_idle(value['spec'])
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_MODULES_CACHE=tempfile.mkdtemp(prefix='GLM46Flash_own_'+stage+'_20261118_'))
  with (H/(stage+'_runner_stdout.log')).open('xb') as out,(H/(stage+'_runner_stderr.log')).open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd='/root',stdout=out,stderr=err,start_new_session=True,env=env)
   put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=stage+'_original_runner_running',runner_PID=child.pid,actual_job=str(job),job_SHA256=sha(job/'job.json')));code=child.wait()
  assert (job/'completion.json').exists();done=js(job/'completion.json');put(H/(stage+'_original_completion.json'),done);assert code==done['actual_returncode']==0,(stage,done)
  while active() or runner_lock_owners():time.sleep(3)
  return done
 try:
  assert sha(J3/'job.json')=='e5e7eff6d8e501f2c60737e83985482ab53ea03c6a556f18b979d01e3280c57b'
  small=js(J3/'completion.json');assert small['actual_returncode']==0 and small['model_returncode'] in [0,1] and small['verifier_returncode']==0
  assert not (J200/'job.json').exists()
  # This source is a new resource continuation of completed genuine NPU3,
  # never a restart or replay of the old true3 or capacity-blocked controller.
  assert len(small['accepted'])==1;accepted=small['accepted'][0];proof_path=Path(accepted['proof']);proof=js(proof_path)
  assert sha(proof_path)==accepted['proof_sha256'] and proof['passed'] and proof['sample_count']==3 and proof['NEW_real_NPU_inference'] and proof['actual_forward_count']>0
  parent=js(J3/'job.json');spec=parent['spec'];import yaml
  cfg=yaml.safe_load((R/spec['config']).read_bytes());pfile=Path(cfg['model']['kwargs']['protocol_file']);p=js(pfile);assert sha(pfile)==cfg['model']['kwargs']['protocol_sha256']
  validate_report(R,Path(accepted['report_dir']),MID,'image-text-to-text',3,proof);front=js(p['own_frontend200']);trace=[json.loads(s) for s in Path(js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution']['sample_trace']).read_text().splitlines()]
  archives=[];total=0;members=[]
  for row in trace:
   record=js(row['actual_execution_file']);ap=Path(record['archive']);assert sha(ap)==record['archive_SHA256'];total+=ap.stat().st_size+Path(row['actual_execution_file']).stat().st_size;archives.append(dict(path=str(ap),bytes=ap.stat().st_size,SHA256=sha(ap)))
   with zipfile.ZipFile(ap) as z:
    for m in z.infolist():assert m.filename.endswith('.npy');members.append(dict(archive=str(ap),name=m.filename,bytes=m.file_size,compressed_bytes=m.compress_size,CRC=m.CRC))
  prompt=[len(r['input_ids']) for r in front];pixels=[math.prod(r['tensor_metadata']['pixel_values']['shape']) for r in front]
  ratios=dict(samples=200/3,all200_own_prompt_tokens=sum(prompt)/sum(prompt[:3]),all200_own_pixel_elements=sum(pixels)/sum(pixels[:3]));estimate=max(math.ceil(total*max(ratios.values())*4)+6*1024**3,int(4.5*1024**3)+20*1024**3);free=shutil.disk_usage(R).free
  cap=dict(UTC=now(),passed=free>=estimate,empirical_estimate_bytes=estimate,actual_free_bytes=free,actual_small3_complete_archives=archives,all_complete_ZIP_members=members,all200_original_prompt_lengths=prompt,all200_original_pixel_elements=pixels,ratios=ratios,original_factor4_plus6GiB_and_hard3_log1_precheckHalf_sample2GiB_preserved=True,empirical_not_worst_case_guarantee=True)
  assert estimate==164003454340,'Original own complete storage estimate changed; preserve and review, never weaken guard'
  put(H/'full_shape_complete_numeric_storage_capacity_review.json',cap)
  put(H/'initial_original_storage_capacity.json',cap)
  put(H/'NOTIFY_resource_wait_only_not_NN.json',dict(UTC=now(),reason='Original complete200 evidence budget exceeds actual project disk; waiting without original runner lock, no expansion/deletion/other disk',capacity_required_bytes=estimate,actual_free_bytes=free,notification_file_not_chat=True))
  while True:
   batch.check_sources(spec);unchanged(parent['protected_canonical_hashes']);assert serial_hashes=={n:sha(n) for n in serial_hashes}
   nn=active();owners=runner_lock_owners();current_free=shutil.disk_usage(R).free
   legacy_alive,legacy_pending,legacy_detail=existing_legacy_dispatch()
   action=choose_action(scheduling,free_bytes=current_free,candidate_budget_bytes=estimate,active_NN_NF=nn,lock_owners=owners,bound_legacy_resumer_alive=legacy_alive,legacy_job_pending=legacy_pending,legacy_budget_bytes=LEGACY_BUDGET)
   put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=action,serial_only=True,max_active_models=1,original_true3_complete_no_replay=True,actual_NEW_model_forwards=0,actual_free_bytes=current_free,required_original_full200_bytes=estimate,capacity_passed=current_free>=estimate,actual_NN_NF=nn,actual_original_FLOCK=owners,existing_unmodified_legacy=legacy_detail,continuation_holds_original_runner_lock=False,no_disk_expansion_or_other_disk_or_matrix_deletion=True))
   if action=='start_candidate_NEWfresh3_fixed200_original_runner':
    cap.update(UTC=now(),actual_free_bytes=current_free,passed=True);put(H/'full_shape_complete_numeric_storage_capacity_review.json',cap);break
   time.sleep(20)
  approval=dict(UTC=now(),approved=True,first200_measurement_only_not_preclaimed_qualification=True,true3_proof=str(proof_path),true3_proof_SHA256=sha(proof_path),true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),full_own_source_and_all_numeric_verified=True,small_sample_score_and_full_D_comparison_disclosed=proof['metrics'],not_an_additional_5pct_small_sample_accuracy_gate=True,capacity_review=str(H/'full_shape_complete_numeric_storage_capacity_review.json'),capacity_review_SHA256=sha(H/'full_shape_complete_numeric_storage_capacity_review.json'),final200_full_NF_original_qualification_threeD5pct_required=True)
  put(H/'first200_measurement_approval.json',approval);p2=copy.deepcopy(p);p2['implementation_source_SHA256'].update({SELF:sha(R/SELF),str(SERIAL_MODULE.relative_to(R)):sha(SERIAL_MODULE),str(SERIAL_CONFIG.relative_to(R)):sha(SERIAL_CONFIG)});p2['static_evidence_SHA256'].update({str(proof_path):sha(proof_path),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')});assert not (R/PROTO).exists() and not (R/CFG).exists();put(R/PROTO,p2)
  c2=copy.deepcopy(cfg);c2['name']=Path(CFG).stem;c2['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/CFG,c2)
  s2=copy.deepcopy(spec);sources={k:v for k,v in spec['source_sha256'].items() if k not in [spec['config'],str(pfile.relative_to(R))]};s2.update(approved=True,config=CFG,source_sha256={**sources,SELF:sha(R/SELF),str(SERIAL_MODULE.relative_to(R)):sha(SERIAL_MODULE),str(SERIAL_CONFIG.relative_to(R)):sha(SERIAL_CONFIG),CFG:sha(R/CFG),PROTO:sha(R/PROTO)},alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
  job=copy.deepcopy(parent);job.update(spec=s2,stages=[3,200],created_at=now(),private_stage_type='Own Huihui original2048 single-model serial first200 approval NEWfresh3 fixed0_199');J200.mkdir(exist_ok=False);put(J200/'job.json',job);scope(s2,[3,200],False);batch.validate_config(c2,MID);batch.check_sources(s2)
  assert cfg['algorithm']==c2['algorithm'] and cfg['data']==c2['data'] and cfg['output']==c2['output'] and cfg['task_type']==c2['task_type']
  assert {k:v for k,v in cfg['model']['kwargs'].items() if k not in ['protocol_file','protocol_sha256']}=={k:v for k,v in c2['model']['kwargs'].items() if k not in ['protocol_file','protocol_sha256']}
  legacy_alive,legacy_pending,_=existing_legacy_dispatch()
  assert choose_action(scheduling,free_bytes=shutil.disk_usage(R).free,candidate_budget_bytes=estimate,active_NN_NF=active(),lock_owners=runner_lock_owners(),bound_legacy_resumer_alive=legacy_alive,legacy_job_pending=legacy_pending,legacy_budget_bytes=LEGACY_BUDGET)=='start_candidate_NEWfresh3_fixed200_original_runner','Changed dispatch admission; preserve new private job without starting a second model'
  done=run(J200,'approved_fresh3_fixed200');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=0,phase='approved_fresh3_fixed200_original_runner_completed',original_completion=done,no_G_or_canonical_write=True));return 0
 except BaseException:
  result=dict(UTC=now(),actual_returncode=1,phase='failed_evidence_preserved',error=traceback.format_exc(),no_G_or_canonical_write=True);put(H/'actual_completion.json',result);put(H/'NOTIFY_error.json',dict(actual=result,notification_file_not_chat=True));raise
if __name__=='__main__':sys.exit(main())

"""Own GLM46 private real3 then measured first200, without waiting for a heartbeat."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,tempfile,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.6V-Flash';READY=B/'GLM46Flash_own_original_BF16_thinking4096_native_source_readiness_v1_1645_20261118'
H=B/'GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_1650_20261118'
J3=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v1_real_FLOCK_1645_20261118'
J200=B/'GLM46Flash_own_original_BF16_thinking4096_approved_fresh3_fixed200_v1_real_FLOCK_1650_20261118'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';SELF='scripts/continue_glm46flash_own_ready_true3_capacity_fresh3_fixed200_v1_1650_20261118.py'
CFG='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.6V-Flash-own-original-BF16-thinking4096-approved-first200-v1-1650-20261118.yaml'
PROTO='configs/protocols/glm46flash_own_original_BF16_thinking4096_approved_first200_v1_1650_20261118.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from batch_910b_support import validate_report
 import batch_910b as batch
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
  while True:
   if (READY/'actual_failure.json').exists():raise RuntimeError('Own readiness failed: '+(READY/'actual_failure.json').read_text())
   ready=(READY/'readiness.json').exists();nn=active();owners=runner_lock_owners()
   put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_own_readiness_and_original_NN_NF_FLOCK_release',own_readiness_exists=ready,actual_NN_NF=nn,original_FLOCK=owners,continuation_holds_runner_lock=False))
   if ready and not nn and not owners:break
   time.sleep(5)
  rd=js(READY/'readiness.json');assert rd['passed'] and rd['ready_for_unique_original_runner_true3'] and rd['actual_NEW_model_forwards']==0 and sha(J3/'job.json')==rd['job_SHA256']
  assert not (J3/'state.json').exists() and not (J3/'completion.json').exists()
  small=run(J3,'true3');assert len(small['accepted'])==1;accepted=small['accepted'][0];proof_path=Path(accepted['proof']);proof=js(proof_path)
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
  put(H/'full_shape_complete_numeric_storage_capacity_review.json',cap)
  if not cap['passed']:
   value=dict(UTC=now(),actual_returncode=1,phase='capacity_blocked_after_genuine_true3_full_NF',capacity=cap,NEWfresh3_or_fixed200_started=False,no_G_or_canonical_write=True);put(H/'actual_completion.json',value);put(H/'NOTIFY_capacity_blocked.json',dict(actual=value,notification_file_not_chat=True));return 1
  approval=dict(UTC=now(),approved=True,first200_measurement_only_not_preclaimed_qualification=True,true3_proof=str(proof_path),true3_proof_SHA256=sha(proof_path),true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),full_own_source_and_all_numeric_verified=True,small_sample_score_and_full_D_comparison_disclosed=proof['metrics'],not_an_additional_5pct_small_sample_accuracy_gate=True,capacity_review=str(H/'full_shape_complete_numeric_storage_capacity_review.json'),capacity_review_SHA256=sha(H/'full_shape_complete_numeric_storage_capacity_review.json'),final200_full_NF_original_qualification_threeD5pct_required=True)
  put(H/'first200_measurement_approval.json',approval);p2=copy.deepcopy(p);p2['implementation_source_SHA256'][SELF]=sha(R/SELF);p2['static_evidence_SHA256'].update({str(proof_path):sha(proof_path),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')});assert not (R/PROTO).exists() and not (R/CFG).exists();put(R/PROTO,p2)
  c2=copy.deepcopy(cfg);c2['name']=Path(CFG).stem;c2['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/CFG,c2)
  s2=copy.deepcopy(spec);sources={k:v for k,v in spec['source_sha256'].items() if k not in [spec['config'],str(pfile.relative_to(R))]};s2.update(approved=True,config=CFG,source_sha256={**sources,SELF:sha(R/SELF),CFG:sha(R/CFG),PROTO:sha(R/PROTO)},alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
  job=copy.deepcopy(parent);job.update(spec=s2,stages=[3,200],created_at=now(),private_stage_type='Own GLM46 measured first200 approval NEWfresh3 fixed0_199');J200.mkdir(exist_ok=False);put(J200/'job.json',job);scope(s2,[3,200],False);batch.validate_config(c2,MID);batch.check_sources(s2)
  done=run(J200,'approved_fresh3_fixed200');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=0,phase='approved_fresh3_fixed200_original_runner_completed',original_completion=done,no_G_or_canonical_write=True));return 0
 except BaseException:
  result=dict(UTC=now(),actual_returncode=1,phase='failed_evidence_preserved',error=traceback.format_exc(),no_G_or_canonical_write=True);put(H/'actual_completion.json',result);put(H/'NOTIFY_error.json',dict(actual=result,notification_file_not_chat=True));raise
if __name__=='__main__':sys.exit(main())

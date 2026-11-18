"""After genuine Phi35 real3/NF, measure full storage and continue without a timer."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,fcntl,hashlib,json,math,os,shutil,subprocess,tempfile,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='microsoft/Phi-3.5-vision-instruct'
J3=B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118'
H=B/'Phi35_formal_NF3_completed_true3_full_shape_capacity_fresh3_fixed200_continuation_v3_1642_20261118'
ACCEPT=B/'Phi35_completed_genuine_true3_formal_NF3_actual_profiler_core_schema_no_NN_replay_1642_20261118'
NF2='scripts/verify_phi35_own_original_native_BF16_4096_v3_actual_profiler_core_schema_1642_20261118.py'
J200=B/'Phi35_own_original_native_BF16_4096_approved_fresh3_fixed200_v3_actual_kernel_schema_real_FLOCK_1642_20261118'
CFG='configs/Phi3VForCausalLM/microsoft--Phi-3.5-vision-instruct-own-original-BF16-native-4096-approved-first200-v3-actual-kernel-schema-1642-20261118.yaml'
PROTO='configs/protocols/phi35_own_original_native_BF16_4096_approved_first200_v3_actual_kernel_schema_1642_20261118.json'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
SELF='scripts/continue_phi35_formal_NF3_completed_true3_capacity_fresh3_fixed200_v3_1642_20261118.py'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(path):return json.loads(Path(path).read_bytes())
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()));tmp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode());os.replace(tmp,path)

def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope
 from batch_910b_support import validate_report
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (H/'actual_completion.json').exists() and not (J200/'job.json').exists()
 assert sha(J3/'job.json')=='a433f2d64b56da90d215f8532913424528f4a6d7acbe05a6a3d6bde72870ccb6'
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 try:
  while True:
   active=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];owners=runner_lock_owners()
   put(H/'state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='waiting_for_genuine_own_true3_full_NF_and_original_runner_release',true3_completion_exists=(ACCEPT/'completion.json').exists(),actual_project_NN_NF=active,actual_original_FLOCK=owners,continuation_holds_original_runner_lock=False))
   if (ACCEPT/'completion.json').exists() and not active and not owners:break
   time.sleep(10)
  failed=js(J3/'completion.json');assert failed['actual_returncode']==1 and failed['model_returncode']==1 and failed['verifier_returncode']==1
  completed=js(ACCEPT/'completion.json');put(H/'formal_NF2_original_actual_completion.json',completed)
  assert completed['actual_returncode']==0 and completed['phase']=='completed_formal_independent_acceptance_of_original_NPU3'
  accept_job=js(ACCEPT/'job.json');assert accept_job['source_original_NPU3_job_SHA256']==sha(J3/'job.json') and accept_job['source_original_NPU3_completion_SHA256']==sha(J3/'completion.json')
  assert sha(R/NF2)=='56d7e6f41f4465bb9a30b585a574a76be2520aa556b3be9198dc9a50d25a1a99'
  assert len(completed['accepted'])==1;accepted=completed['accepted'][0]
  assert accepted['sample_count']==3 and accepted['new_real_NPU_inference']
  proof_path=Path(accepted['proof']);assert sha(proof_path)==accepted['proof_sha256'];proof=js(proof_path)
  assert proof['passed'] and proof['actual_NPU_inference'] and not proof['NEW_real_NPU_inference'] and proof['formal_independent_acceptance_phase_not_diagnostic_only'] and proof['actual_NEW_model_forwards_in_this_acceptance']==0 and proof['sample_count']==3 and proof['actual_forward_count']>0
  parent=js(J3/'job.json');spec=parent['spec'];batch.check_sources(spec)
  for path,h in parent['protected_canonical_hashes'].items():assert sha(path)==h,path
  import yaml
  cfg=yaml.safe_load((R/spec['config']).read_bytes());pfile=Path(cfg['model']['kwargs']['protocol_file']);p=js(pfile)
  assert sha(pfile)==cfg['model']['kwargs']['protocol_sha256']
  folder=Path(accepted['report_dir']);validate_report(R,folder,MID,'image-text-to-text',3,proof)
  line=js(folder/'operator_lineage.json')['actual_execution'];idx=js(line['full_numeric_group_index'])
  assert idx['actual_completed_samples']==3 and len(idx['groups'])==1
  group=js(idx['groups'][0]['path']);assert sha(group['archive'])==group['archive_SHA256'] and sha(idx['groups'][0]['path'])==idx['groups'][0]['SHA256']
  front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];assert [r['index'] for r in front]==list(range(200))
  lengths=[len(r['actual_native_prompt_token_ids']) for r in front];visual=[r['original_HD_image_token_count'] for r in front]
  pixel_elements=[math.prod(r['actual_native_processor_pixel_tensor']['shape']) for r in front]
  ratios=dict(samples=200/3,all200_own_prompt_tokens=sum(lengths)/sum(lengths[:3]),all200_own_HD_visual_tokens=sum(visual)/sum(visual[:3]),all200_own_actual_pixel_elements=sum(pixel_elements)/sum(pixel_elements[:3]))
  members=[]
  with zipfile.ZipFile(group['archive']) as z:
   for member in z.infolist():
    assert member.filename.endswith('.npy');members.append(dict(name=member.filename,bytes=member.file_size,compressed_bytes=member.compress_size,CRC=member.CRC))
  assert len(members)==group['lossless_shared_tensor_storage']['physical_unique_arrays']
  control_scale=max(ratios.values());small_bytes=Path(group['archive']).stat().st_size+Path(idx['groups'][0]['path']).stat().st_size
  estimate=math.ceil(small_bytes*control_scale*4)+6*1024**3
  # Keep the original complete outer10 admission and full archive scratch guard,
  # even if short observed generations would suggest a smaller estimate.
  full_outer10_guard=int(4.5*1024**3)+10*2*1024**3
  estimate=max(estimate,full_outer10_guard)
  free=shutil.disk_usage(R).free
  capacity=dict(UTC=datetime.now(timezone.utc).isoformat(),empirical_estimate_bytes=estimate,actual_free_bytes=free,passed=free>=estimate,all200_original_prompt_lengths=lengths,all200_original_visual_tokens=visual,all200_actual_native_pixel_elements=pixel_elements,complete_true3_ZIP_members=members,ratios=ratios,maximum_full200_shape_ratio=control_scale,full_small_archive_bytes=small_bytes,original_factor4_plus6GiB_preserved=True,original_hard3_log1_precheck_half_sample2GiB_guards_preserved=True,empirical_not_worst_case_guarantee=True,no_generation_budget_or_model_or_original_batching_changed=True)
  put(H/'full_shape_complete_numeric_storage_capacity_review.json',capacity)
  if not capacity['passed']:
   result=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,phase='capacity_blocked_after_genuine_true3_and_full_NF_no_replay',true3_proof=str(proof_path),true3_proof_SHA256=sha(proof_path),capacity=capacity,NEWfresh3_or_fixed200_started=False,no_G_or_canonical_write=True)
   put(H/'actual_completion.json',result);put(H/'NOTIFY_capacity_blocked.json',dict(actual=result,notification_file_only_not_chat_delivery=True));return 1
  approval=dict(UTC=datetime.now(timezone.utc).isoformat(),approved=True,first_full200_is_measurement_not_preclaimed_qualification=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,real_NPU3_proof=str(proof_path),real_NPU3_proof_SHA256=sha(proof_path),true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),original_generation_budget4096_and_outer10_and_all_model_math_preserved=True,small_sample_score_and_difference_disclosed=proof['metrics'],small_scope_relative_errors_to_full200_D_are_not_a_hardware_accuracy_gate=proof['relative_error_exact_three_D'],capacity_review=str(H/'full_shape_complete_numeric_storage_capacity_review.json'),capacity_review_SHA256=sha(H/'full_shape_complete_numeric_storage_capacity_review.json'),all_own_source_actual_NPU_and_full_numeric_checks_passed=True,final200_full_NF_original_qualification_and_threeD_5pct_still_required=True)
  put(H/'first200_measurement_approval.json',approval)
  p2=copy.deepcopy(p);p2['implementation_source_SHA256'].update({SELF:sha(R/SELF),NF2:sha(R/NF2)});p2['static_evidence_SHA256'].update({str(proof_path):sha(proof_path),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(H/'full_shape_complete_numeric_storage_capacity_review.json'):sha(H/'full_shape_complete_numeric_storage_capacity_review.json')})
  assert not (R/PROTO).exists() and not (R/CFG).exists();put(R/PROTO,p2)
  c2=copy.deepcopy(cfg);c2['name']=Path(CFG).stem;c2['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/CFG,c2)
  s2=copy.deepcopy(spec);sources={k:v for k,v in spec['source_sha256'].items() if k not in [spec['config'],str(pfile.relative_to(R))]}
  s2.update(approved=True,config=CFG,verifier=NF2,source_sha256={**sources,SELF:sha(R/SELF),NF2:sha(R/NF2),CFG:sha(R/CFG),PROTO:sha(R/PROTO)},alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
  job=copy.deepcopy(parent);job.update(spec=s2,stages=[3,200],created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Approved first200 measurement NEWfresh3 then fixed0_199')
  J200.mkdir(exist_ok=False);put(J200/'job.json',job);scope(s2,[3,200],False);batch.validate_config(c2,MID);batch.check_sources(s2)
  entry=js(D/'state.json')['models'][MID]
  for count in [3,200]:_,actual=batch.config_for(s2,entry,count);assert actual['sample_range']==f'0:{count-1}'
  assert not runner_lock_owners() and batch.device_idle(s2)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_MODULES_CACHE=tempfile.mkdtemp(prefix='Phi35_own_approved_cache_0616_20261118_'),ASCEND_RT_VISIBLE_DEVICES='4')
  with (H/'approved_runner_stdout.log').open('xb') as out,(H/'approved_runner_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(RUNNER),str(J200)],cwd='/root',env=env,stdout=out,stderr=err,start_new_session=True)
   put(H/'state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='own_approved_NEWfresh3_then_fixed200_original_runner_running',runner_PID=child.pid,formal_home=str(J200),job_SHA256=sha(J200/'job.json')))
   code=child.wait()
  assert (J200/'completion.json').exists();result=js(J200/'completion.json');put(H/'approved_runner_actual_completion.original.json',result)
  put(H/'actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=code,phase='approved_original_runner_exited',formal_home=str(J200),original_actual_completion=result,no_G_or_canonical_write=True))
  if code:put(H/'NOTIFY_approved_runner_error.json',dict(actual=result,notification_file_only_not_chat_delivery=True))
  return code
 except BaseException:
  value=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,phase='failed_evidence_preserved',error=traceback.format_exc(),no_G_or_canonical_write=True)
  put(H/'actual_completion.json',value);put(H/'NOTIFY_error.json',dict(actual=value,notification_file_only_not_chat_delivery=True));raise

if __name__=='__main__':sys.exit(main())

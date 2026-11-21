"""Accept existing genuine Ovis200 values after Meta fullNF, correcting only original raw fallback metadata."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,fcntl,os,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
sys.path[:0]=[str(R),str(R/'scripts')]
CONFIG='configs/queues/OvisU1_completed200_original_raw_fallback_NF2_after_Meta_fullNF_serial_20261121.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
 q=js(R/CONFIG);H=Path(q['home']);assert not H.exists();H.mkdir();guard=(H/'acceptance.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in q['source_sha256'].items():assert sha(R/p)==h,p
 old=(R/q['original_verifier']).read_text();new=(R/q['new_verifier']).read_text();assert new.replace("outs[i]['prediction']==details[i]['prediction']==(choice or raw)","outs[i]['prediction']==details[i]['prediction']==choice")==old
 J=Path(q['original_job']);assert sha(J/'job.json')==q['original_job_SHA256'];failed=js(J/'completion.json');assert failed['actual_returncode']==failed['verifier_returncode']==1 and failed['model_returncode']==0 and failed['sample_count']==200 and [x['sample_count'] for x in failed['accepted']]==[3] and failed['accepted'][0]['proof_sha256']==q['own_approved_NEWfresh3_proof_SHA256']
 report=Path(q['original_report']);before={str(p):sha(p) for p in report.iterdir()};assert len(before)==5
 previous=Path(q['previous_Meta_acceptance_home'])
 while not (previous/'actual_completion.json').is_file() or not quiet():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_current_Meta_fullNF_actual_completion_and_original_EX_release_no_model_NN',original_sample_count=200,original_native_forwards=102400,actual_NEW_model_forwards=0));time.sleep(20)
 assert js(previous/'actual_completion.json')['actual_returncode'] in [0,1]
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
  for p,h in q['source_sha256'].items():assert sha(R/p)==h,p
  assert sha(J/'job.json')==q['original_job_SHA256'] and all(sha(p)==h for p,h in before.items())
  protected={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]}
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='formal_independent_NF2_all_original_Ovis200_values_original_raw_fallback_schema_no_NN_replay',original_sample_count=200,original_native_forwards=102400,actual_NEW_model_forwards=0))
  with (H/'verification_stdout.json').open('xb') as out,(H/'verification_stderr.log').open('xb') as err:
   run=subprocess.run(['python3','-B',str(R/q['new_verifier']),str(report),'200'],cwd=R,stdout=out,stderr=err,timeout=21600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  assert run.returncode==0,'New complete NF failed; preserve original/current reports and every numerical array'
  proof=js(H/'verification_stdout.json');assert proof['passed'] and proof['sample_count']==200 and proof['actual_forward_count']==102400 and proof['source_tensor_count']==1060 and proof['active_physical_source_tensor_count']==487 and proof['original_unexecuted_generator_source_tensor_count']==573 and proof['metrics']['Accuracy']==.895
  assert proof['all_own_source_actual_NPU_SHA256_verified'] and proof['all_native_KV_values_losslessly_reconstructed'] and proof['all_full_inputs_tokens_raw_references_errors_preserved'] and len(proof['numeric_sample_checks'])==200 and all(v['all28_KV_and_all_masks_positions_preserved'] for v in proof['numeric_sample_checks'])
  assert proof['error_indices']==proof['empty_indices']==[] and proof['invalid_indices']==[198,199] and proof['truncated_indices']==list(range(200)) and proof['independent_verifier_SHA256']==sha(R/q['new_verifier'])
  assert all(sha(p)==h for p,h in before.items()) and all(sha(p)==h for p,h in protected.items())
  put(H/'formal_acceptance.json',dict(passed=True,UTC=now(),actual_NEW_model_forwards=0,original_genuine_NEW_NPU200_job=str(J),original_job_SHA256=sha(J/'job.json'),original_job_remains_failed=True,original_model_returncode=0,original_verifier_returncode=1,current_verifier_returncode=0,original_sample_count=200,original_native_forwards=102400,own_approved_NEWfresh3_proof_SHA256=q['own_approved_NEWfresh3_proof_SHA256'],proof=str(H/'verification_stdout.json'),proof_SHA256=sha(H/'verification_stdout.json'),original_full_five_reports_unchanged=True,all_own1060_source_and_all200_full_numeric_head_KV_inputs_raw_ref_error_denominators_verified=True,only_original_raw_prediction_fallback_expected_metadata_corrected=True,score=.895,three_D_relative_errors=proof['relative_error_to_current_A100_D'],original_protocol_qualified=proof['original_protocol_final_qualification_complete'],no_G_or_canonical_write=True))
 put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_NEW_model_forwards=0,formal_acceptance=str(H/'formal_acceptance.json'),formal_acceptance_SHA256=sha(H/'formal_acceptance.json'),no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  try:
   q=js(R/CONFIG);H=Path(q['home'])
   if H.exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0,all_original_reports_sources_values_logs_preserved=True,no_G_or_canonical_write=True))
  finally:raise

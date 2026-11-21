"""Revalidate the existing Meta200 under the original EX after the current Ovis serial work."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,fcntl,hashlib,json,os,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
CONFIG='configs/queues/MetaMllama_completed200_original_invalid_empty_NF4_v2_actual_file_SHA_after_Ovis_serial_20261121.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
 q=js(R/CONFIG)
 original_failed=Path(q['original_preNN_source_pin_failed_completion']);assert sha(original_failed)==q['original_preNN_source_pin_failed_completion_SHA256'];failed_preNN=js(original_failed);assert failed_preNN['actual_returncode']==1 and failed_preNN['actual_NEW_model_forwards']==0
 H=Path(q['home']);assert not H.exists();H.mkdir();guard=(H/'acceptance.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for n,h in q['source_sha256'].items():assert sha(R/n)==h
 old=(R/q['old_verifier']).read_text();new=(R/q['new_verifier']).read_text()
 assert old.count("prediction=raw if choice is None else choice")==1 and new.replace("prediction='' if choice is None else choice","prediction=raw if choice is None else choice")==old
 previous=Path(q['previous_Ovis_job'])
 while not (previous/'completion.json').is_file() or not quiet():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_same_Ovis_real_NN_NF_and_original_EX_release_no_NEW_NN',actual_NEW_model_forwards=0));time.sleep(20)
 done=js(previous/'completion.json')
 if done.get('actual_returncode')==0 and done.get('accepted') and done['accepted'][-1]['sample_count']==200:
  last=js(done['accepted'][-1]['proof'])
  if last['original_protocol_final_qualification_complete']:
   while True:
    state=js(D/'state.json');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');entry=state['models']['AIDC-AI/Ovis-U1-3B']
    if entry.get('original_protocol_final_qualification_complete',False) and binding['local_workbook_sha256']==sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx') and binding['real_record_count']==sum(v['status']=='done' for v in state['models'].values()):break
    put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_qualified_Ovis_dual_publication_priority_no_NEW_NN',actual_NEW_model_forwards=0));time.sleep(20)
 while not quiet():time.sleep(20)
 J=Path(q['original_Meta_job']);assert sha(J/'job.json')==q['original_Meta_job_SHA256'];failed=js(J/'completion.json')
 assert failed['actual_returncode']==failed['verifier_returncode']==1 and failed['model_returncode']==1 and [a['sample_count'] for a in failed['accepted']]==[3]
 report=Path(q['original_report']);before={str(p):sha(p) for p in report.iterdir()};assert len(before)==5
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
  for n,h in q['source_sha256'].items():assert sha(R/n)==h
  protected={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]}
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='formal_independent_NF4_existing_genuine_Meta200_original_empty_invalid_schema_no_NN_replay',original_NPU_sample_count=200,original_NPU_forward_count=2823,actual_NEW_model_forwards=0))
  with (H/'verification_stdout.json').open('xb') as out,(H/'verification_stderr.log').open('xb') as err:
   result=subprocess.run(['python3','-B',str(R/q['new_verifier']),str(report),'200'],cwd=R,stdout=out,stderr=err,timeout=21600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  assert result.returncode==0,'New independent verifier failed; preserve all original and current failure evidence'
  proof=js(H/'verification_stdout.json');assert proof['passed'] and proof['sample_count']==200 and proof['actual_native_forwards']==2823 and proof['Accuracy']==.705 and len(proof['invalid_indices'])==6 and proof['source_tensor_count']==906
  assert proof['all906_own_source_cast_actual_NPU_SHA_verified'] and proof['all511_own_visual_source_SHA_verified'] and proof['all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward'] and proof['all_full128256_native_greedy_argmax_replayed']==2823
  assert all(sha(p)==h for p,h in before.items()) and all(sha(p)==h for p,h in protected.items()) and sha(J/'job.json')==q['original_Meta_job_SHA256']
  put(H/'formal_acceptance.json',dict(passed=True,UTC=now(),actual_NEW_model_forwards=0,original_genuine_NPU200_job=str(J),original_job_remains_failed=True,original_NPU_forward_count=2823,proof=str(H/'verification_stdout.json'),proof_SHA256=sha(H/'verification_stdout.json'),original_full_five_reports_unchanged=True,all906_source_all200_full_numeric_and_original_inputs_tokens_raw_ref_error_denominators_verified=True,metadata_expectation_only_original_invalid_predictions_are_empty_strings=True,score=.705,three_D_relative_errors=proof['relative_error_to_current_A100_D'],original_protocol_qualified=proof['original_protocol_final_qualification_complete'],no_G_or_canonical_write=True))
 put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_NEW_model_forwards=0,formal_acceptance=str(H/'formal_acceptance.json'),formal_acceptance_SHA256=sha(H/'formal_acceptance.json'),no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  try:
   q=js(R/CONFIG);H=Path(q['home'])
   if H.exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0,all_original_reports_values_sources_logs_unchanged=True,no_G_or_canonical_write=True))
  finally:raise

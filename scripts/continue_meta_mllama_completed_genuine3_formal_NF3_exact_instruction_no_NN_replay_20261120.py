"""Exact metadata-only formal acceptance of same genuine3, no model replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,fcntl,hashlib,importlib,json,os,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120';OLD=B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120';OLDH=B/'MetaMllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120'
CFG='configs/queues/MetaMllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_capacity_serial_20261120.json';NF='scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py';SELF='scripts/continue_meta_mllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_20261120.py';MODEL='meta-llama/Llama-3.2-11B-Vision-Instruct'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def next200(job,accepted,pf,p,cap,control):
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 assert control.quiet()
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(OLD),true3_job_SHA256=sha(OLD/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_all200_complete_shape_capacity_review.json'),capacity_review_SHA256=sha(H/'own_passed3_all200_complete_shape_capacity_review.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True,formal_same_genuine_NN3_acceptance=str(H/'formal_acceptance.json'),formal_same_genuine_NN3_acceptance_SHA256=sha(H/'formal_acceptance.json'))
 put(H/'own_first200_measurement_approval.json',approval)
 p2=copy.deepcopy(p);visual=Path(pf['own_original_visual_numerical_controls']['archive']).parent/'review.json';p2['own_completed_true3_visual_reference']=dict(review=str(visual),SHA256=sha(visual));p2['runtime_and_original_source_sha256'][str(R/NF)]=sha(R/NF);p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(H/'formal_acceptance.json'):sha(H/'formal_acceptance.json'),str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')})
 original=js(R/job['spec']['config']);oldproto=str(Path(original['model']['kwargs']['protocol_file']).relative_to(R));c2=copy.deepcopy(original)
 np='configs/protocols/meta_mllama_own_original_SDPA2048_approved_first200_NF3_serial_v2_20261120.json';nc='configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct-own-original-BF16-SDPA2048-approved-first200-NF3-serial-v2-20261120.yaml';J=B/'MetaMllama_completed_genuine3_formal_NF3_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261120'
 assert not (R/np).exists() and not (R/nc).exists() and not J.exists();put(R/np,p2);c2['name']=Path(nc).stem;c2['model']['kwargs'].update(protocol_file=str(R/np),protocol_sha256=sha(R/np));put(R/nc,c2)
 spec=copy.deepcopy(job['spec']);spec['source_sha256']={n:h for n,h in spec['source_sha256'].items() if n not in [job['spec']['config'],oldproto]};spec.update(approved=True,config=nc,verifier=NF,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));spec['source_sha256'].update({np:sha(R/np),nc:sha(R/nc),NF:sha(R/NF),SELF:sha(R/SELF),CFG:sha(R/CFG)})
 scope(spec,[3,200],False);batch.validate_config(c2,MODEL)
 for n in [3,200]:
  path,v=batch.config_for(spec,js(D/'state.json')['models'][MODEL],n);assert v['sample_range']==f'0:{n-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert control.quiet() and batch.device_idle(spec)
 jj=copy.deepcopy(job);jj.update(spec=spec,stages=[3,200],created_at=now(),private_stage_type='same genuine true3 formally accepted after exact metadata guard fix, approved NEWfresh3 fixed200, no original true3 replay');J.mkdir();put(J/'job.json',jj)
 with (H/'approved_NEWfresh3_fixed200_stdout.log').open('xb') as out,(H/'approved_NEWfresh3_fixed200_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(R/'scripts/A100_alignment_evidence_runner_20261103.py'),str(J)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put(H/'state.json',dict(UTC=now(),phase='approved_NEWfresh3_fixed200_original_runner_active',runner_PID=child.pid,job=str(J),job_SHA256=sha(J/'job.json'),max_concurrent_models=1));rc=child.wait()
 done=js(J/'completion.json');assert rc==done['actual_returncode'];return dict(UTC=now(),actual_returncode=rc,phase='approved_NEWfresh3_fixed200_actual_completed',job=str(J),completion=done,no_G_or_canonical_write=True)
def main():
 from A100_alignment_evidence_runner_20261103 import unchanged
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 control=importlib.import_module('continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120');config=js(R/CFG)
 assert not H.exists();H.mkdir();guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(R/NF)==config['new_verifier_SHA256']=='90bd632c82c339b0b060194342250bde5a868ec5795119db849b9cd5efa8b7c2';assert sha(OLD/'job.json')=='ac1384546dbab375c0d02e2fad31e4d0dea3c8f7f2a194e48b72ea9a24086567'
 while not (OLD/'completion.json').is_file() or not (OLDH/'actual_completion.json').is_file() or not control.quiet():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_original_actual_NN_NF_controller_exit_no_lock_no_model_replay',max_concurrent_models=1));time.sleep(5)
 prior=B/'MetaMllama_completed_genuine3_formal_NF2_exact_original_instruction_no_NN_replay_capacity_serial_20261120'
 assert js(prior/'actual_completion.json')['actual_returncode']==1 and "r['scale']==1./math.sqrt(r['query_shape'][-1])" in (prior/'formal_fullNF2_stderr.log').read_text()
 done=js(OLD/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==1,done
 logs={str(q):q.read_text(errors='replace') for q in (OLD/'fresh3').glob('*') if q.is_file() and q.suffix in ['.log','.txt']}
 joined='\n'.join(logs.values());assert 'verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v1_20261120.py' in joined and "cfg['algorithm']['kwargs']=={'extract_answer':True}" in joined,logs.keys()
 report=R/'workspace/reports/meta-llama_Llama-3.2-11B-Vision-Instruct/image-text-to-text/20261120_121105';assert {p.name for p in report.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
 rep=js(report/'eval_report.json');assert rep['pipeline_config']['model']['name']==MODEL and rep['statistics']['total_samples']==3 and rep['statistics']['error_count']==0 and len(rep['detailed_results'])==3
 job=js(OLD/'job.json');unchanged(job['protected_canonical_hashes']);before={p.name:sha(p) for p in report.iterdir()};sources={n:sha(R/n) for n in [NF,SELF,CFG]};original_done_sha=sha(OLD/'completion.json');lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock')
 try:
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='formal_fullNF3_same_genuine_NPU3_no_NN_replay',original_job=str(OLD),actual_NEW_model_forwards=0,actual_original_FLOCK_owned=True))
  with (H/'formal_fullNF3_stdout.json').open('xb') as out,(H/'formal_fullNF3_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(R/NF),str(report),'3'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put(H/'NF3_child.json',dict(UTC=now(),PID=child.pid,argv=['python3','-B',str(R/NF),str(report),'3']));rc=child.wait()
  assert before=={p.name:sha(p) for p in report.iterdir()} and sha(OLD/'completion.json')==original_done_sha;unchanged(job['protected_canonical_hashes']);assert {n:sha(R/n) for n in sources}==sources
  if rc!=0:return dict(UTC=now(),actual_returncode=rc,phase='formal_NF3_failed_original_values_preserved',actual_NEW_model_forwards=0,old_job_original_failure_preserved=True,no_G_or_canonical_write=True)
  pf=js(H/'formal_fullNF3_stdout.json');assert pf['passed'] and pf['sample_count']==3 and pf['actual_native_forwards']>0 and pf['all906_own_source_cast_actual_NPU_SHA_verified'] and pf['all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward']
  proof=Path(rep['operator_lineage']['actual_execution']['weight_loading_evidence']).parent/'independent_report_verification.json';assert js(proof)==pf
  accepted=dict(sample_count=3,report_dir=str(report),proof=str(proof),proof_sha256=sha(proof));formal=dict(schema='completed_original_genuine_NPU3_formal_independent_acceptance_exact_original_power_reference_v3',UTC=now(),passed=True,actual_returncode=0,actual_NEW_model_forwards_in_this_acceptance=0,original_genuine_NPU3_job=str(OLD),original_job_SHA256=sha(OLD/'job.json'),original_completion_SHA256=original_done_sha,original_NF1_returncode=1,new_NF3_returncode=0,original_instruction_metadata_and_original_power_expression_reference_restored=True,all_numeric_math_AST_unchanged=True,all_original_five_report_SHA256=before,all_original_NN_source_and_values_preserved=True,source_SHA256=sources,accepted=[accepted]);put(H/'formal_acceptance.json',formal)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 assert control.quiet();control.H=H # Private in-process diagnostic target only; the old source/job/state are not edited.
 cap,p,_=control.capacity(job,accepted)
 if not cap['capacity_passed']:return dict(UTC=now(),actual_returncode=1,phase='same_genuine3_formal_NF3_passed_first200_capacity_refused',formal_acceptance=str(H/'formal_acceptance.json'),formal_acceptance_SHA256=sha(H/'formal_acceptance.json'),proof=str(proof),proof_SHA256=sha(proof),actual_NEW_model_forwards_in_this_acceptance=0,no_approval_NEWfresh3_200_or_G=True,all_original_five_sources_configs_logs_values_retained=True)
 return next200(job,accepted,pf,p,cap,control)
if __name__=='__main__':
 try:r=main();put(H/'actual_completion.json',r);sys.exit(r['actual_returncode'])
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0,no_G_or_canonical_write=True,all_original_values_reports_sources_configs_logs_retained=True))
  raise

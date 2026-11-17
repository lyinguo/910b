"""Prepare immutable own-model device-isolated stages, validate, then launch once."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,copy,subprocess,ast,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
import yaml
from A100_alignment_evidence_runner_20261103 import unchanged,scope
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
import batch_910b as batch
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def main(queuepath,mode):
 assert mode in ['prepare','launch'];q=js(queuepath);assert q['max_parallel_stages']==2 and q['no_G_or_canonical_writes'] and q['no_deletion'];unchanged(q['expected_canonical']);assert not project_processes() and not runner_lock_owners();assert sha(R/'scripts/A100_alignment_evidence_runner_20261103.py')=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 for n,h in q['declared_sources'].items():assert sha(R/n)==h
 home=B/'two_card_Merve_lossless_Prithiv_card2_parallel_v2_20261117';mhome=B/'Merve_own603_lossless_complete_values_approved_NEWfresh3_fixed200_parallel_card4_v2_20261117';phome=B/'PrithivGemma4E4B_existing_model_first_independent_card2_true3_parallel_v1_20261117'
 if mode=='prepare':
  assert not home.exists() and not mhome.exists() and not phome.exists();home.mkdir();mhome.mkdir();phome.mkdir()
  d=home/'latest_actual_user_parallel_request.json';put(d,dict(origin='actual_user_chat_instruction',exact_user_request=q['exact_user_request'],recorded_UTC=now(),implementation_scope='Two independent own-checkpoint models on separate physical cards, preserving each original single-card mathematical protocol and all original report/numeric gates. No claim of new tensor parallel performance.',existing_user_evaluation_and_publication_rules_still_apply=True));put(home/'source_delta_review.json',q['source_delta_review'])
  canon=q['expected_canonical'];book,=[Path(p) for p in canon if p.endswith('.xlsx')];snapshot=home/'reference_workbook_verified178_before_parallel.xlsx';snapshot.write_bytes(book.read_bytes());assert sha(snapshot)==canon[str(book)]
  mj=B/q['Merve_original_completed_full_numeric3_home'];oldm=js(mj/'job.json');mc=js(mj/'completion.json');assert mc['actual_returncode']==0 and len(mc['accepted'])==1 and mc['accepted'][0]['sample_count']==3;accepted=mc['accepted'][0];assert sha(accepted['proof'])==accepted['proof_sha256']=='d8b7425d08dfbc19077f41c83b79dd296671802f9f56f84bc8555e2f8481cdd3';proof=js(accepted['proof']);assert proof['passed'] and proof['numeric_review_passed']
  diag=B/q['Merve_lossless_diagnostic_home']/'review.json';assert sha(diag)==q['Merve_lossless_diagnostic_review_SHA256'];diagv=js(diag);assert diagv['diagnostic_only'] and diagv['numeric_review_passed'];oldmfolder=Path(accepted['report_dir']);assert all(sha(oldmfolder/n)==h for n,h in proof['report_sha256'].items())
  review=home/'Merve_prospective_first200_measurement_approval.json';put(review,dict(approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,UTC=now(),actual_completed_original_true3_proof=accepted['proof'],actual_completed_original_true3_proof_SHA256=accepted['proof_sha256'],full_original_checks_against_exact_shared_value_storage=str(diag),full_original_checks_against_exact_shared_value_storage_SHA256=sha(diag),source_delta_review=str(home/'source_delta_review.json'),source_delta_review_SHA256=sha(home/'source_delta_review.json'),new_real_approved_fresh3_required_before200=True,first200_is_measurement_not_predetermined_5pct_qualification=True,small3_vs_whole200_D_not_used_as_accuracy_gate=True,complete_original_values_occurrences_and_all_numeric_checks_retained=True,all_original_model_weights_inputs_math_dtype_budget_cache_sampling_unchanged=True,capacity_still_requires_actual_NEWfresh3_all_NPY_and_metadata_factor4_plus6GiB=True,no_G_or_canonical_write=True))
  settings=[(mhome,oldm,'Merve',4,[3,200]),(phome,js(B/q['Prithiv_original_completed_card4_home']/'job.json'),'Prithiv',2,[3])];rows=[]
  common={n:sha(R/n) for n in ['scripts/A100_alignment_evidence_runner_20261103.py','scripts/A100_alignment_parallel_control_20261117.py',q['core_worker'],q['core_master'],str(Path(__file__).relative_to(R)),str(Path(queuepath).relative_to(R))]}
  canonical_state=js(D/'state.json')
  for jobhome,old,prefix,card,stages in settings:
   cfg=yaml.safe_load((R/old['spec']['config']).read_bytes());pold=Path(cfg['model']['kwargs']['protocol_file']);p=js(pold);adapter=q[prefix+'_adapter'];verifier=q[prefix+'_verifier']
   ppath=R/'configs/protocols'/('Merve_own603_lossless_full_values_parallel_card4_v2_20261117.json' if prefix=='Merve' else 'Prithiv_own_original_BF16_first_card2_affinity_v1_20261117.json');cpath=R/'configs'/('PaliGemmaForConditionalGeneration' if prefix=='Merve' else 'Gemma4ForConditionalGeneration')/('merve--paligemma_vqav2-original-BF16-2048-lossless-approved-card4-parallel-v2-20261117.yaml' if prefix=='Merve' else 'prithivMLmods--gemma-4-E4B-it-Uncensored-MAX-original-BF16-2048-first-card2-parallel-v1-20261117.yaml')
   for n in [adapter,verifier,*common]:assert (R/n).is_file()
   if prefix=='Merve':
    p['full_numeric_evidence_job']=str(jobhome/'job.json');p['observation_only_worker_extension_cls']=q['Merve_extension'];p['lossless_storage_full_value_preservation_review']=str(diag);p['lossless_storage_full_value_preservation_review_SHA256']=sha(diag);p['first200_measurement_approval']=str(review);p['first200_measurement_approval_SHA256']=sha(review);p['full_numeric_implementation_sources_SHA256'].update(common)
    for n in [adapter,verifier,'universal_eval/utils/lossless_npz_aliases.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_named_lossless_RPC_v2.py']:p['full_numeric_implementation_sources_SHA256'][n]=sha(R/n)
    for n in [review,diag,home/'source_delta_review.json']:p['static_evidence_sha256'][str(n)]=sha(n)
    impl=p['full_numeric_implementation_sources_SHA256'];cfg['model']['adapter']=adapter+':MervePaliOwn603LosslessValues'
   else:
    completed=js(B/q['Prithiv_original_completed_card4_home']/'completion.json');assert completed['actual_returncode']==0 and completed['accepted'][0]['proof_sha256']=='fd3bdcaf169547fe4e90f3fb6e1ea269ccb7d2490a085090d07d9d942a42c9b8';assert sha(completed['accepted'][0]['proof'])==completed['accepted'][0]['proof_sha256'];p['reference_workbook']=str(snapshot);p['reference_workbook_SHA256']=sha(snapshot);p['implementation_source_SHA256'].update(common);p['implementation_source_SHA256'].update({adapter:sha(R/adapter),verifier:sha(R/verifier)});p['physical_card_deployment_validation']=dict(original_completed_card4_proof=completed['accepted'][0]['proof'],original_completed_card4_proof_SHA256=completed['accepted'][0]['proof_sha256'],this_stage_is_first_independent_card2_affinity_validation_not_repetition_of_card4_qualification=True,model_forward_and_observer_and_dtype_math_unchanged=True);p['static_evidence_SHA256'][str(home/'source_delta_review.json')]=sha(home/'source_delta_review.json');impl=p['implementation_source_SHA256'];cfg['model']['adapter']=adapter+':PrithivGemma4E4BOwnBF16Card2'
   cfg['name']=cpath.stem;cfg['model']['kwargs']['npu_devices']=[card];cfg['model']['kwargs']['protocol_file']=str(ppath);put(ppath,p);cfg['model']['kwargs']['protocol_sha256']=sha(ppath);put(cpath,cfg)
   spec=copy.deepcopy(old['spec']);spec.update(config=str(cpath.relative_to(R)),verifier=verifier,npu_devices=[card],approved=prefix=='Merve',review_revision='parallel_card_isolation_only_full_numeric_and_original_math_preserved',parallel_storage_budget=dict(small3_bytes=q['small3_budget_bytes_each'],formal200_requires_actual_accepted_NEWfresh3_capacity=True),protocol_file=str(ppath),protocol_sha256=sha(ppath))
   if prefix=='Merve':spec.update(alignment_approval_review=str(review),alignment_approval_review_sha256=sha(review))
   spec['source_sha256'].update(impl);spec['source_sha256'].update(common);spec['source_sha256'][str(ppath.relative_to(R))]=sha(ppath);spec['source_sha256'][str(cpath.relative_to(R))]=sha(cpath)
   entry=copy.deepcopy(canonical_state['models'][old['model_id']]);entry['model_id']=old['model_id'];batch.validate_config(cfg,old['model_id']);assert batch.device_idle(spec);batch.check_sources(spec)
   for count in [3,200]:
    effective,ecfg=batch.config_for(spec,copy.deepcopy(entry),count);assert ecfg['sample_range']==f'0:{count-1}' and ecfg['model']['kwargs']['npu_devices']==[card];spec['source_sha256'][str(effective.relative_to(R))]=sha(effective)
   job=copy.deepcopy(old);job.update(spec=spec,stages=stages,protected_canonical_hashes=canon,created_at=now(),parallel_stage_type='first independent card2 deployment validation' if card==2 else 'approved NEWfresh3 then first complete fixed200 with full lossless value storage',runner_source_sha256=common);scope(spec,stages,False);batch.check_sources(spec);put(jobhome/'job.json',job);rows.append(dict(home=str(jobhome),job_SHA256=sha(jobhome/'job.json'),card=card,model_id=old['model_id']))
   source_snapshot=jobhome/'source_snapshot_before_start'
   for n,h in spec['source_sha256'].items():
    raw=(R/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==h;out=source_snapshot/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
  batchfile=home/'batch.json';put(batchfile,dict(max_parallel_stages=2,jobs=rows,worker_source=str(R/q['core_worker']),latest_user_parallel_request=str(d),latest_user_parallel_request_SHA256=sha(d),protected_canonical_hashes=canon,one_real_original_global_FLOCK_and_real_individual_card_locks=True,no_G_or_state_or_registry_or_workbook_writes=True,full_model_and_full_numeric_loop_preserved=True))
  unchanged(canon);put(home/'readiness_review.json',dict(UTC=now(),passed=True,all_primary_source_SHA256={n:sha(R/n) for n in common},jobs=rows,no_NPU_started=True,no_original_source_or_completed_phase_or_canonical_modified=True,source_snapshot_complete=True,each_small3_budget_bytes=q['small3_budget_bytes_each'],combined_initial_small3_budget_bytes=q['small3_budget_bytes_each']*2,actual_project_free=shutil.disk_usage(R).free));print(json.dumps(dict(phase='prepared_not_started',home=str(home),readiness_SHA256=sha(home/'readiness_review.json'),jobs=rows),ensure_ascii=False));return
  
 assert mode=='launch' and js(home/'readiness_review.json')['passed'] and not (home/'actual_launch.json').exists();bf=js(home/'batch.json');assert len(bf['jobs'])==2 and {x['card'] for x in bf['jobs']}=={2,4};assert shutil.disk_usage(R).free>q['small3_budget_bytes_each']*2
 for row in bf['jobs']:
  job=js(Path(row['home'])/'job.json');assert sha(Path(row['home'])/'job.json')==row['job_SHA256'];batch.check_sources(job['spec']);scope(job['spec'],job['stages'],False);assert batch.device_idle(job['spec'])
 unchanged(q['expected_canonical']);out=(home/'coordinator_stdout.log').open('xb');err=(home/'coordinator_stderr.log').open('xb');child=subprocess.Popen(['python3','-B',str(R/q['core_master']),str(home)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'));out.close();err.close();put(home/'actual_launch.json',dict(UTC=now(),PID=child.pid,argv=['python3','-B',str(R/q['core_master']),str(home)],batch_SHA256=sha(home/'batch.json'),coordinator_source_SHA256=sha(R/q['core_master']),no_G_or_canonical_write=True));print(json.dumps(dict(phase='launched_real_coordinator',home=str(home),PID=child.pid,launch_SHA256=sha(home/'actual_launch.json')),ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]).resolve(),sys.argv[2])

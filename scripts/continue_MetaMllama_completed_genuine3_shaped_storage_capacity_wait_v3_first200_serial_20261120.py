"""Complete lossless storage only; reuse accepted genuine3, then fresh3/200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,fcntl,hashlib,json,os,shutil,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='meta-llama/Llama-3.2-11B-Vision-Instruct'
SELF='scripts/continue_MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.py'
CONFIG='configs/queues/MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.json'
MODEL='universal_eval/model/adapters/meta_mllama_own_original_BF16_SDPA_2048_native_lossless_shaped_LZMA_v3_20261120.py'
OLDMODEL='universal_eval/model/adapters/meta_mllama_own_original_BF16_SDPA_2048_native_v1_20261120.py'
UTILITY='universal_eval/utils/complete_native_raw_bytes_Fortran_LZMA_20261120.py'
NF='scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py'
NEWPROTO='configs/protocols/meta_mllama_completed_genuine3_shaped_storage_LZMA_v3_approved_first200_serial_20261120.json'
NEWCFG='configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct-own-original-SDPA-BF16-2048-complete-shaped-LZMA-v3-approved-first200-serial-20261120.yaml'
H=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120'
DIAG=B/'MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_storage_capacity_diagnostic_only_20261120'
FORMAL=B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120'
J3=B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120'
J=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 cfg=js(R/CONFIG);assert cfg['model_id']==MID and cfg['max_concurrent_models']==1
 assert not H.exists() and not J.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 unchanged({str(R/n):h for n,h in cfg['exact_source_SHA256'].items()})
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 while not (DIAG/'actual_completion.json').is_file():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_complete_storage_diagnostic_not_NN',actual_NEW_model_forwards=0));time.sleep(20)
 done=js(DIAG/'actual_completion.json');assert done['actual_returncode']==0 and done['actual_NEW_model_forwards']==0
 review=js(DIAG/'review.json');cap=js(DIAG/'all200_complete_storage_capacity_review.json');assert review['passed'] and review['all_original_keys_full_native_tensor_values_dtype_shape_raw_SHA_and_original_decoder_unchanged'] and sha(DIAG/'all200_complete_storage_capacity_review.json')==review['capacity_review_SHA256']
 assert cap['original_factor']==4 and cap['original_extra_reserve_bytes']==6*2**30 and cap['unknown_axes_original_maximum_retained'] and cap['all200_own_original_shapes_verified']
 # Existing GLM41 owns priority when its larger budget actually fits.
 prior= B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
 polls=0
 while True:
  free=shutil.disk_usage(R).free;higher_priority=(free>=cfg['existing_GLM41_priority_required_bytes'] and not (prior/'actual_completion.json').is_file())
  all_quiet=quiet();polls+=1
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_actual_original_capacity_not_NN',polls=polls,actual_free_bytes=free,required_bytes=cap['empirical_estimate_bytes'],existing_GLM41_priority=higher_priority,all_project_NN_NF_and_original_EX_released=all_quiet,actual_NEW_model_forwards=0,max_concurrent_models=1,no_expansion_other_volume_deletion_guard_change_or_passed3_replay=True))
  if free>=cap['empirical_estimate_bytes'] and all_quiet and not higher_priority:break
  time.sleep(20)
 assert sha(FORMAL/'formal_acceptance.json')=='e55ce8c8080c1a8696bb58725631c6b8635fb1eb01352fbafebec2f87c8c361c'
 accepted,=js(FORMAL/'formal_acceptance.json')['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256']=='eb85a1130a28cc99059da75a17b6110d0a3e7fa669dbe8dc51a694c1b6826d48';pf=js(accepted['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['actual_native_forwards']==11 and pf['all906_own_source_cast_actual_NPU_SHA_verified']
 for rec in review['all_three_complete_archive_reviews']:
  assert sha(rec['source_archive'])==rec['source_archive_SHA256'] and sha(rec['new_complete_archive'])==rec['new_archive_SHA256'] and rec['ordinary_original_independent_decoder_accepts_unchanged']
 assert len(review['all_three_complete_archive_reviews'])==3
 job=js(J3/'job.json');unchanged(job['protected_canonical_hashes']);assert quiet()
 # Exact AST comparison permits just this storage writer and its import.
 old=(R/OLDMODEL).read_text();new=(R/MODEL).read_text();oldcall="np.savez_compressed(q,**{k:v.reshape(-1).view(torch.uint8).numpy() for k,v in self._arrays.items()})";newcall='write_complete_native_raw_npz(q,self._arrays)';newimport='from universal_eval.utils.complete_native_raw_bytes_Fortran_LZMA_20261120 import write_complete_native_raw_npz\n'
 assert old.count(oldcall)==1 and new.count(newcall)==1 and new.count(newimport)==1
 assert ast.dump(ast.parse(old.replace(oldcall,newcall)),include_attributes=False)==ast.dump(ast.parse(new.replace(newimport,'')),include_attributes=False)
 oldcfg=R/job['spec']['config'];c2=js(oldcfg);oldproto=Path(c2['model']['kwargs']['protocol_file']);p=js(oldproto);p2=copy.deepcopy(p)
 exact={**cfg['exact_source_SHA256'],CONFIG:sha(R/CONFIG)}
 p2['implementation_source_SHA256'].update(exact);p2['implementation_source_SHA256'][NF]=sha(R/NF)
 p2['runtime_and_original_source_sha256'].update({str(R/n):h for n,h in exact.items()});p2['runtime_and_original_source_sha256'][str(R/NF)]=sha(R/NF)
 visual=Path(pf['own_original_visual_numerical_controls']['archive']).parent/'review.json';p2['own_completed_true3_visual_reference']=dict(review=str(visual),SHA256=sha(visual))
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],formal_NF3_acceptance=str(FORMAL/'formal_acceptance.json'),formal_NF3_acceptance_SHA256=sha(FORMAL/'formal_acceptance.json'),same_genuine_true3_no_model_replay=True,storage_review=str(DIAG/'review.json'),storage_review_SHA256=sha(DIAG/'review.json'),capacity_review=str(DIAG/'all200_complete_storage_capacity_review.json'),capacity_review_SHA256=sha(DIAG/'all200_complete_storage_capacity_review.json'),all_original_values_original_decoder_and_nonstorage_model_AST_identical=True,own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True)
 assert quiet();put(H/'own_first200_measurement_approval.json',approval)
 p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(DIAG/'review.json'):sha(DIAG/'review.json'),str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')})
 p2['complete_lossless_storage_layout']=dict(format='standard NPY uint8[original_shape...,itemsize] Fortran serialization in ZIP-LZMA; all native bytes and old independent decoder unchanged',source=UTILITY,SHA256=sha(R/UTILITY),old_source_original_genuine3_SHA256=sha(R/OLDMODEL),unchanged_nonstorage_AST=True)
 assert not (R/NEWPROTO).exists() and not (R/NEWCFG).exists();put(R/NEWPROTO,p2);c2['name']=Path(NEWCFG).stem;c2['model']['adapter']=MODEL+':MetaMllamaOwnOriginal2048';c2['model']['kwargs'].update(protocol_file=str(R/NEWPROTO),protocol_sha256=sha(R/NEWPROTO));put(R/NEWCFG,c2)
 spec=copy.deepcopy(job['spec']);spec['source_sha256'].update(exact);spec['source_sha256'].pop(job['spec']['config']);spec['source_sha256'].pop(str(oldproto.relative_to(R)))
 spec.update(approved=True,config=NEWCFG,verifier=NF,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));spec['source_sha256'].update({NF:sha(R/NF),NEWCFG:sha(R/NEWCFG),NEWPROTO:sha(R/NEWPROTO)})
 scope(spec,[3,200],False);batch.validate_config(c2,MID)
 for count in [3,200]:
  path,c=batch.config_for(spec,js(D/'state.json')['models'][MID],count);assert c['sample_range']==f'0:{count-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(spec) and shutil.disk_usage(R).free>=cap['empirical_estimate_bytes']
 nextjob=copy.deepcopy(job);nextjob.update(spec=spec,stages=[3,200],created_at=now(),private_stage_type='Accepted genuine3 plus complete lossless standard byteplane storage; approved measurement NEWfresh3 then fixed200 serial max1');J.mkdir();put(J/'job.json',nextjob)
 with (H/'runner_stdout.log').open('xb') as out,(H/'runner_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='approved_NEWfresh3_fixed200_original_runner_active',job=str(J),job_SHA256=sha(J/'job.json'),runner_PID=child.pid,max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
 done=js(J/'completion.json');assert done['actual_returncode']==rc;put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='NEWfresh3_fixed200_natural_actual_completion',job=str(J),completion=done,no_G_or_canonical_write=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_values_reports_sources_configs_logs_retained=True,no_G_or_canonical_write=True))
  raise

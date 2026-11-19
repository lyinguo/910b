"""Read-only serial queue gate, then own NEW AutoGLM NPU3 after actual GLM outcome.

The waiting phase owns no runner lock and performs no learned inference. A
qualifying GLM200 must be committed to both workbooks before dispatch. A fully
audited baseline gap preserves G and may release the next own-weight model.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,os,shutil,subprocess,time,traceback,zipfile
from decimal import Decimal
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
SELF='scripts/continue_AutoGLMPhone9B_single_own_inputs_true3_capacity_fresh3_fixed200_after182_serial_20261120.py'
CFG='configs/queues/after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120.json'
H=B/'after_AutoGLM182_own_single_Phone9B_inputs_true3_fixed200_serial_20261120'
MID='zai-org/AutoGLM-Phone-9B'
JGLM=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119'
CGLM=B/'GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119'
PUB=B/'ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119'
PUBRUN=B/'ZaiGLM46Flash_verified181_publication_review_inputs_v3_exact_verifier_20261119'
READY=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261120'
J3=B/'AutoGLMPhone9BSingle_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
ROLES={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def quiet():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 if runner_lock_owners() or [v for v in project_processes() if v['PID']!=os.getpid() and v['role'] in ROLES]:return False
 # Frozen monitor intentionally remains unchanged. Independently include the
 # actual absolute R runner/model/verifier commands even if cwd is /root.
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except FileNotFoundError:continue
  if not args or not Path(args[0]).name.startswith('python'):continue
  if any(str(x).startswith(str(R)+'/') and (Path(x).name in ['run.py','A100_alignment_evidence_runner_20261103.py'] or Path(x).name.startswith(('verify_','audit_'))) for x in args[1:]):return False
 return True
def actual_full200_qualification(pf):
 assert pf['passed'] is True and pf['sample_count']==200 and pf['actual_NPU_inference'] is True
 assert pf['all_own_source_actual_NPU_SHA256_verified'] is True and pf['all_native_KV_values_losslessly_reconstructed'] is True and pf['all_full_inputs_tokens_raw_references_errors_preserved'] is True
 assert pf['all_original_active_and_unused_source_bytes_verified'] is True and pf['source_tensor_count']==704 and pf['actual_forward_count']>0
 errors=pf['relative_error_to_current_A100_D'];assert set(errors)=={'server','local','preserved_original'}
 numerical=all(Decimal(str(x))>=0 and Decimal(str(x))<=Decimal('.05') for x in errors.values())
 assert pf['formal_alignment_within_5pct'] is numerical and pf['original_protocol_final_qualification_complete'] is numerical
 return numerical
def continue_own_passed3_full_shape_capacity_first200(cfg,job,accepted):
 """Complete ZIP axes plus all own 200 inputs; original factor4 and reserves."""
 import math,numpy as np,yaml
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from glm46_complete_shape_component_capacity_v3_actual_one_based_20261119 import estimate_complete
 assert quiet() and job['stages']==[3] and sha(accepted['proof'])==accepted['proof_sha256']
 pf=js(accepted['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['actual_NPU_inference'] and pf['actual_forward_count']>0 and pf['source_tensor_count']==704
 assert pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed'] and pf['all_full_inputs_tokens_raw_references_errors_preserved'] and pf['all_original_active_and_unused_source_bytes_verified']
 original=yaml.safe_load((R/job['spec']['config']).read_bytes());protocol=Path(original['model']['kwargs']['protocol_file']);assert sha(protocol)==original['model']['kwargs']['protocol_sha256'];p=js(protocol)
 assert original['algorithm']['batch_size']==10 and not original['data']['filter_empty'] and p['model_id']==MID
 front=js(p['own_frontend200']);assert sha(p['own_frontend200'])==p['own_frontend200_SHA256'] and [x['index'] for x in front]==list(range(200))
 tokens=[len(x['input_ids']) for x in front];pixels=[math.prod(x['tensor_metadata']['pixel_values']['shape']) for x in front]
 report=js(Path(accepted['report_dir'])/'eval_report.json');line=report['operator_lineage']['actual_execution'];trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert [x['index'] for x in trace]==[0,1,2]
 parts=[];forwards=[];metadata=0;bound={accepted['proof']:accepted['proof_sha256']}
 for i,x in enumerate(trace):
  rec=js(x['actual_execution_file']);assert sha(x['actual_execution_file'])==x['actual_execution_SHA256'] and rec['index']==i and rec['error'] is None and rec['input_ids']==front[i]['input_ids'];ap=Path(rec['archive']);assert sha(ap)==rec['archive_SHA256'];bound[str(ap)]=rec['archive_SHA256'];bound[x['actual_execution_file']]=x['actual_execution_SHA256'];forwards.append(rec['actual_native_forwards']);compressed=0
  with zipfile.ZipFile(ap) as z:
   assert len(z.namelist())==len(set(z.namelist()))
   for zi in z.infolist():
    assert zi.filename.endswith('.npy') and zi.compress_type==zipfile.ZIP_LZMA
    with z.open(zi) as f:
     version=np.lib.format.read_magic(f);assert version in [(1,0),(2,0)]
     shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f));assert not dtype.hasobject
    compressed+=zi.compress_size;parts.append(dict(index=i,name=zi.filename,shape=list(shape),dtype=dtype.str,fortran_order=fortran,compressed_bytes=zi.compress_size,uncompressed_complete_NPY_bytes=zi.file_size,original_complete_archive=str(ap),original_complete_archive_SHA256=rec['archive_SHA256']))
  metadata+=ap.stat().st_size-compressed+Path(x['actual_execution_file']).stat().st_size
 cap=estimate_complete(parts,tokens,pixels,forwards,metadata);cap.update(UTC=now(),passed=True,model_id=MID,all_original200_prompt_lengths=tokens,all_original200_pixel_elements=pixels,all_complete_original_values_still_available_and_SHA_bound=bound,actual_free_bytes=shutil.disk_usage(R).free,original_true3_proof_SHA256=accepted['proof_sha256'],all200_inputs_are_this_models_own_original_frontend=True,no_NN_or_compression_replay_in_capacity_review=True)
 cap['capacity_passed']=cap['actual_free_bytes']>=cap['empirical_estimate_bytes'];put(H/'own_passed3_complete_shape_capacity_review.json',cap)
 if not cap['capacity_passed']:
  put(H/'NOTIFY_own_AutoGLM_passed3_formal200_capacity_refused_not_chat_sent.json',dict(UTC=now(),own_true3_fullNF_passed=True,own_true3_proof_SHA256=accepted['proof_sha256'],actual_needed_bytes=cap['empirical_estimate_bytes'],actual_free_bytes=cap['actual_free_bytes'],no_approval_or_NEWfresh3_or200_or_G=True,all_guards_and_complete_values_retained=True,chat_notification_not_claimed=True))
  return dict(UTC=now(),actual_returncode=1,phase='own_AutoGLM_true3_fullNF_passed_formal200_capacity_refused',own_true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_shape_capacity_review.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_shape_capacity_review.json'),no_G_or_canonical_write_by_queue=True,all_original_values_reports_sources_configs_logs_retained=True)
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert quiet()
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_shape_capacity_review.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_shape_capacity_review.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True)
 put(H/'own_first200_measurement_approval.json',approval)
 newcfg='configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B-single-own-original-BF16-2048-LZMA-approved-first200-serial-v1-20261120.yaml';newproto='configs/protocols/autoglm_phone9b_single_own_original_BF16_2048_LZMA_approved_first200_serial_v1_20261120.json';j200=B/'AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120'
 assert not (R/newcfg).exists() and not (R/newproto).exists() and not j200.exists()
 impl=dict(p['implementation_source_SHA256']);impl.update({n:sha(R/n) for n in [SELF,CFG,'scripts/glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py']})
 p2=copy.deepcopy(p);p2['implementation_source_SHA256']=impl;p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json'),str(H/'own_passed3_complete_shape_capacity_review.json'):sha(H/'own_passed3_complete_shape_capacity_review.json')});put(R/newproto,p2)
 c2=copy.deepcopy(original);c2['name']=Path(newcfg).stem;c2['model']['kwargs'].update(protocol_file=str(R/newproto),protocol_sha256=sha(R/newproto));put(R/newcfg,c2)
 sources={n:h for n,h in job['spec']['source_sha256'].items() if n not in [job['spec']['config'],str(protocol.relative_to(R))]};sources.update(impl);sources.update({newcfg:sha(R/newcfg),newproto:sha(R/newproto)})
 spec=copy.deepcopy(job['spec']);spec.update(approved=True,config=newcfg,source_sha256=sources,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));scope(spec,[3,200],False);batch.validate_config(c2,MID)
 for count in [3,200]:
  path,actual=batch.config_for(spec,js(D/'state.json')['models'][MID],count);assert actual['sample_range']==f'0:{count-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(spec)
 nextjob=copy.deepcopy(job);nextjob.update(spec=spec,stages=[3,200],created_at=now(),private_stage_type='Own AutoGLM genuinely completed3 then approved NEWfresh3 fixed200 serial, original graph/input/budget and complete independent NF')
 j200.mkdir();put(j200/'job.json',nextjob)
 with (H/'own_NEWfresh3_fixed200_runner_stdout.log').open('xb') as out,(H/'own_NEWfresh3_fixed200_runner_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(j200)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own_AutoGLM_approved_NEWfresh3_fixed200_same_original_runner_active_not_claimed_passed',runner_PID=child.pid,job=str(j200),job_SHA256=sha(j200/'job.json'),no_G_or_canonical_write_by_queue=True));rc=child.wait()
 return dict(UTC=now(),actual_returncode=rc,phase='own_AutoGLM_NEWfresh3_fixed200_actual_runner_completed',job=str(j200),original_job_completion=js(j200/'completion.json'),no_G_or_canonical_write_by_queue=True,all_original_values_reports_sources_configs_logs_retained=True)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import unchanged
 cfg=js(R/CFG);assert cfg['max_concurrent_models']==1 and cfg['own_input200_and_own_source_all_required'];unchanged({str(R/n):h for n,h in cfg['own_next_source_SHA256'].items()});assert not H.exists() and not READY.exists() and not J3.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=B/'AutoGLMPhone9B_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119';pub=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120'
 recovery=js(pub/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json');audit=js(pub/'full_target_dual_publication_audit.json');dual=js(pub/'dual_publication_completed.json');assert recovery['actual_returncode']==0 and audit['passed'] and dual['completed'] and audit['qualified_delivered']==68
 while not quiet():
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_all_previous_NN_NF_original_EX_release_no_NN_here'));time.sleep(3)
 done=js(previous/'completion.json');formal,=[v for v in done['accepted'] if v['sample_count']==200];assert done['actual_returncode']==0 and sha(formal['proof'])==formal['proof_sha256']=='498c5f03c923bdcd12f8ea40022f03a4c8bae44ea6ca0eabde00687b8566c95d';pf=js(formal['proof']);passed=actual_full200_qualification(pf);assert passed
 for stage,producer in [('own_input200','scripts/prepare_autoglm_phone9b_single_own_original_BF16_2048_frontend200_20261120.py'),('own_fixed200_alias','scripts/prepare_autoglm_phone9b_single_own_AFS_R_identical_fixed200_no_filter_input_alias_20261120.py')]:
  assert quiet() and sha(R/producer)==cfg['own_next_source_SHA256'][producer]
  with (H/(stage+'_stdout.json')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:rc=subprocess.run(['python3','-B',str(R/producer)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')).returncode
  assert rc==0,(stage,'Own preNN input/source preparation refused; preserve full stderr')
 alias=B/'AutoGLMPhone9BSingle_own_original_AFS_R_identical_fixed200_no_filter_input_alias_no_NN_20261120/own_fixed200_no_filter_input_alias_proof.json';v=js(alias);assert v['passed'] and v['model_id']==MID
 validate_report(R,formal['report_dir'],'zai-org/AutoGLM-Phone-9B-Multilingual','image-text-to-text',200,pf)
 unchanged({str(R/n):h for n,h in cfg['own_next_source_SHA256'].items()});assert quiet() and sha(RUNNER)==cfg['frozen_runner_SHA256']
 put(H/'preceding_GLM_actual_outcome_gate.json',dict(UTC=now(),passed=True,prior_full200=str(formal['report_dir']),prior_fullNF=str(formal['proof']),prior_fullNF_SHA256=formal['proof_sha256'],prior_qualified=passed,prior_qualified_requires_dual_publication_complete=passed,prior_baselinegap_has_no_G_write=not passed,all_real_NN_NF_and_original_EX_released=True,next_stage_is_genuine_NEW_own_AutoGLM3_no_prior_own_model_replay=True))
 prep=R/cfg['own_next_readiness_preparer'];assert not READY.exists() and not J3.exists()
 with (H/'own_source_readiness_stdout.json').open('xb') as out,(H/'own_source_readiness_stderr.log').open('xb') as err:
  rc=subprocess.run(['python3','-B',str(prep)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')).returncode
 assert rc==0,('Own next readiness failed before NN; preserve full stderr',str(H/'own_source_readiness_stderr.log'))
 ready=js(READY/'readiness.json');assert ready['ready_for_unique_original_runner_true3'] and ready['actual_NEW_model_forwards']==0 and ready['first200_not_approved']
 job=js(J3/'job.json');assert job['model_id']==MID and job['stages']==[3] and not job['spec']['approved'];batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(job['spec'])
 with (H/'own_true3_runner_stdout.log').open('xb') as out,(H/'own_true3_runner_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J3)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own_NEW_AutoGLM3_original_real_runner_started_not_claimed_completed',runner_PID=child.pid,job=str(J3),job_SHA256=sha(J3/'job.json'),next_first200_not_approved=True,no_G_or_canonical_write_by_queue=True));rc=child.wait()
 result=dict(UTC=now(),actual_returncode=rc,original_job_completion=js(J3/'completion.json'),own_next_model=MID,no_G_or_canonical_write_by_queue=True,next_first200_capacity_and_measurement_approval_still_required=True)
 put(H/'own_true3_actual_completion.json',result);assert rc==result['original_job_completion']['actual_returncode']==0
 accepted,=result['original_job_completion']['accepted'];assert accepted['sample_count']==3
 outcome=continue_own_passed3_full_shape_capacity_first200(cfg,job,accepted);put(H/'actual_completion.json',outcome);assert outcome['actual_returncode']==0,outcome
 return 0
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write_by_queue=True,chat_notification_not_claimed=True))
  raise

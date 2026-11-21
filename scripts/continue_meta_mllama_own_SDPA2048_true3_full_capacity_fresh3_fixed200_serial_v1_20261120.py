"""Serial max1 own Meta true3/fullNF, complete-shape capacity and approved200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='meta-llama/Llama-3.2-11B-Vision-Instruct';H=B/'MetaMllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120';READY=B/'MetaMllama_own_original_SDPA_BF16_2048_source_readiness_v1_20261120';J3=B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';PREPARE='scripts/prepare_meta_mllama_own_original_SDPA_2048_true3_v1_20261120.py';SELF='scripts/continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py';CONFIG='configs/queues/MetaMllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.json'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def quiet():
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 if runner_lock_owners():return False
 roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
 if [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in roles]:return False
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:a=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
  except FileNotFoundError:continue
  if not a or not Path(a[0]).name.startswith('python'):continue
  if any(s.startswith(str(R)+'/') and (Path(s).name in ['run.py',RUNNER.name] or Path(s).name.startswith(('verify_','audit_'))) for s in a[1:]):return False
 return True
def launch_runner(job,phase):
 with (H/(phase+'_stdout.log')).open('xb') as out,(H/(phase+'_stderr.log')).open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,job=str(job),job_SHA256=sha(job/'job.json'),runner_PID=child.pid,max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
 done=js(job/'completion.json');assert done['actual_returncode']==rc
 return rc,done
def capacity(job,accepted):
 import numpy as np
 report=js(Path(accepted['report_dir'])/'eval_report.json');line=report['operator_lineage']['actual_execution'];p=js(line['protocol_file']);trace=[json.loads(x) for x in Path(line['per_sample_evidence']).read_text().splitlines()];assert [x['index'] for x in trace]==[0,1,2]
 parts=[];bytes3=0;metadata=0
 for x in trace:
  q=Path(x['group_execution']);g=js(q);assert g['indices']==[x['index']] and sha(q)==x['group_execution_sha256'];ap=Path(g['archive']);ap_sha=sha(ap);assert ap_sha==g['archive_sha256'];bytes3+=ap.stat().st_size;metadata+=q.stat().st_size;compressed=0
  with zipfile.ZipFile(ap) as z:
   assert len(z.namelist())==len(set(z.namelist()))==len(g['arrays'])
   for zi in z.infolist():
    with z.open(zi) as f:
     ver=np.lib.format.read_magic(f);assert ver in [(1,0),(2,0)];shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if ver==(1,0) else np.lib.format.read_array_header_2_0(f));assert not dtype.hasobject
    parts.append(dict(index=x['index'],name=zi.filename,shape=list(shape),dtype=dtype.str,fortran_order=fortran,compressed_bytes=zi.compress_size,complete_uncompressed_NPY_bytes=zi.file_size,archive=str(ap),archive_SHA256=ap_sha));compressed+=zi.compress_size
  metadata+=ap.stat().st_size-compressed
 tokens=[len(r['unpad_ids']) for r in p['input_records']];pixels=[math.prod(r['actual_tensor_metadata']['pixel_values']['shape']) for r in p['input_records']]
 ratio=max(1.,max(tokens)/min(tokens[:3]),max(pixels)/min(pixels[:3]));empirical=math.ceil((bytes3+metadata)*200/3*ratio*4+6*2**30)
 cap=dict(UTC=now(),model_id=MID,all_complete_ZIP_parts=parts,all200_original_prompt_token_lengths=tokens,all200_original_pixel_elements=pixels,all_unknown_axes_keep_original_global_maximum_ratio=ratio,original_factor=4,original_reserve_bytes=6*2**30,hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_reserve_bytes=2*2**30,outer_actual_guard_batch=10,complete_original_three_archive_bytes=bytes3,complete_metadata_bytes=metadata,empirical_estimate_bytes=empirical,actual_free_bytes=shutil.disk_usage(R).free,not_a_2048_worst_case_guarantee=True,no_original_values_or_precision_or_budget_or_guard_reduced=True)
 cap['capacity_passed']=cap['actual_free_bytes']>=empirical;put(H/'own_passed3_all200_complete_shape_capacity_review.json',cap);return cap,p,report
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 cfg=js(R/CONFIG);assert cfg['model_id']==MID and cfg['max_concurrent_models']==1 and not H.exists() and not J3.exists() and not READY.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 unchanged({str(R/n):h for n,h in cfg['exact_source_SHA256'].items()});assert quiet()
 with (H/'readiness_stdout.log').open('xb') as out,(H/'readiness_stderr.log').open('xb') as err:
  proc=subprocess.Popen(['python3','-B',str(R/PREPARE)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own906_source_YAML_readiness_no_NN',readiness_PID=proc.pid));rc=proc.wait()
 assert rc==0,dict(readiness_returncode=rc,readiness_failure=js(READY/'actual_failure.json') if (READY/'actual_failure.json').is_file() else None)
 ready=js(READY/'readiness.json');job=js(J3/'job.json');assert ready['ready_for_unique_original_runner_true3'] and sha(J3/'job.json')==ready['job_SHA256'];unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert quiet() and batch.device_idle(job['spec'])
 rc,done=launch_runner(J3,'own_genuine_NEWtrue3_fullNF_original_runner_active');assert rc==done['verifier_returncode']==0,done
 accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256'];pf=js(accepted['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['all906_own_source_cast_actual_NPU_SHA_verified'] and pf['actual_native_forwards']>0;assert quiet()
 cap,p,report=capacity(job,accepted)
 if not cap['capacity_passed']:
  put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='own_Meta_true3_fullNF_passed_formal200_capacity_refused',true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_all200_complete_shape_capacity_review.json'),no_approval_NEWfresh3_or200_or_G=True,all_original_values_reports_sources_configs_logs_retained=True,chat_notification_not_claimed=True));return 1
 # First200 is measurement only; small3 does not establish its qualification.
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_all200_complete_shape_capacity_review.json'),capacity_review_SHA256=sha(H/'own_passed3_all200_complete_shape_capacity_review.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);put(H/'own_first200_measurement_approval.json',approval)
 oldcfg=R/job['spec']['config'];c2=js(oldcfg);p2=copy.deepcopy(p);visual=Path(pf['own_original_visual_numerical_controls']['archive']).parent/'review.json';p2['own_completed_true3_visual_reference']=dict(review=str(visual),SHA256=sha(visual));p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')})
 newproto='configs/protocols/meta_mllama_own_original_SDPA2048_approved_first200_serial_v1_20261120.json';newcfg='configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct-own-original-BF16-SDPA2048-approved-first200-serial-v1-20261120.yaml';J=B/'MetaMllama_own_completed3_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
 assert not (R/newproto).exists() and not (R/newcfg).exists() and not J.exists();put(R/newproto,p2);c2['name']=Path(newcfg).stem;c2['model']['kwargs'].update(protocol_file=str(R/newproto),protocol_sha256=sha(R/newproto));put(R/newcfg,c2)
 spec=copy.deepcopy(job['spec']);oldproto=str(Path(p['protocol_file']).relative_to(R)) if 'protocol_file' in p else str(Path(js(oldcfg)['model']['kwargs']['protocol_file']).relative_to(R));spec['source_sha256']={n:h for n,h in spec['source_sha256'].items() if n not in [job['spec']['config'],oldproto]};spec.update(approved=True,config=newcfg,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));spec['source_sha256'].update({newcfg:sha(R/newcfg),newproto:sha(R/newproto),SELF:sha(R/SELF),CONFIG:sha(R/CONFIG)})
 scope(spec,[3,200],False);batch.validate_config(c2,MID)
 for n in [3,200]:
  path,v=batch.config_for(spec,js(D/'state.json')['models'][MID],n);assert v['sample_range']==f'0:{n-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(spec);nextjob=copy.deepcopy(job);nextjob.update(spec=spec,stages=[3,200],created_at=now());J.mkdir();put(J/'job.json',nextjob)
 rc,done=launch_runner(J,'approved_NEWfresh3_fixed200_original_runner_active');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='NEWfresh3_fixed200_natural_actual_completion',job=str(J),completion=done,no_G_or_canonical_write=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True,all_original_values_reports_sources_configs_logs_retained=True,chat_notification_not_claimed=True))
  raise

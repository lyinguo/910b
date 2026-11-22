"""Serial max1 original probability NEW3, full numerical proof then actual capacity."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.1V-9B-Base';H=B/'GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122';READY=B/'GLM41Base_own_original_option_probability_BF16_native_source_readiness_v1_20261122';J3=B/'GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';PREP='scripts/prepare_GLM41Base_own_original_option_probability_true3_v1_20261122.py';SELF='scripts/continue_GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.py';CONFIG='configs/queues/GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.json'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def quiet():
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 if runner_lock_owners():return False
 if any(p['PID']!=os.getpid() and p['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'} for p in project_processes()):return False
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  except OSError:continue
  if a and Path(a[0]).name.startswith('python') and any(v.startswith(str(R)+'/') and (Path(v).name in ['run.py',RUNNER.name] or Path(v).name.startswith(('verify_','audit_'))) for v in a[1:]):return False
 return True
def launch(job,phase):
 with (H/(phase+'_stdout.log')).open('xb') as out,(H/(phase+'_stderr.log')).open('xb') as err:
  c=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,runner_PID=c.pid,job=str(job),job_SHA256=sha(job/'job.json'),max_concurrent_models=1,no_G_or_canonical_write=True));rc=c.wait()
 done=js(job/'completion.json');assert done['actual_returncode']==rc;return rc,done
def capacity(accepted):
 import numpy as np
 proof=js(accepted['proof']);report=js(Path(proof['report_dir'])/'eval_report.json');p=js(report['operator_lineage']['actual_execution']['protocol_file']);front=js(p['own_frontend200']);trace=[json.loads(s) for s in Path(report['operator_lineage']['actual_execution']['sample_trace']).read_text().splitlines()];assert [v['index'] for v in trace]==[0,1,2]
 parts=[];size3=0;metadata=0
 for small in trace:
  record=Path(small['actual_execution_file']);assert sha(record)==small['actual_execution_SHA256'];r=js(record);archive=Path(r['archive']);assert sha(archive)==r['archive_SHA256'];size3+=archive.stat().st_size;metadata+=record.stat().st_size;compressed=0
  with zipfile.ZipFile(archive) as z:
   assert len(z.namelist())==len(set(z.namelist()))==r['unique_complete_arrays']
   for item in z.infolist():
    with z.open(item) as f:
     version=np.lib.format.read_magic(f);assert version in [(1,0),(2,0)];shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f));assert not dtype.hasobject
    parts.append(dict(index=r['index'],name=item.filename,shape=list(shape),dtype=dtype.str,fortran_order=fortran,compressed_bytes=item.compress_size,complete_uncompressed_NPY_bytes=item.file_size,archive=str(archive),archive_SHA256=r['archive_SHA256']));compressed+=item.compress_size
  metadata+=archive.stat().st_size-compressed
 shapes=[dict(index=r['index'],option_index=j,letter=o['letter'],original_prompt_length=len(o['own_complete_input_ids']),pixel_shape=o['all_input_tensor_metadata']['pixel_values']['shape'],original_max_tokens_per_option=1,original_fallback_max_tokens_unused=2048,physical_KV_layers=40,physical_KV_heads=2,physical_KV_dim=128,full_vocabulary=151552) for r in front for j,o in enumerate(r['options'])];small=[s for s in shapes if s['index']<3];ratio=max(1.,(max(s['original_prompt_length'] for s in shapes)/min(s['original_prompt_length'] for s in small))**2,(max(s['pixel_shape'][0] for s in shapes)/min(s['pixel_shape'][0] for s in small))**2);required=math.ceil((size3+metadata)*len(shapes)/len(small)*ratio*4+6*2**30)
 cap=dict(UTC=now(),model_id=MID,all_complete_ZIP_parts=parts,all200_own_original_all_option_full_shapes=shapes,all200_option_counts=[len(r['options']) for r in front],total_original_option_requests=len(shapes),original_unknown_axis_conservative_max_growth=ratio,complete_three_archive_bytes=size3,complete_metadata_bytes=metadata,original_factor=4,original_reserve_bytes=6*2**30,hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_reserve_bytes=2*2**30,actual_inner_serial_batch=1,original_outer_batch=10,empirical_estimate_bytes=required,actual_free_bytes=shutil.disk_usage(R).free,not_a_worst_case_guarantee=True,original_probability_sampling_and_all_precision_math_guards_unchanged=True);cap['capacity_passed']=cap['actual_free_bytes']>=required;put(H/'own_passed3_complete_all200_option_shape_capacity.json',cap);return cap,p
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 q=js(R/CONFIG);assert q['model_id']==MID and q['max_concurrent_models']==1 and not H.exists() and not J3.exists() and not READY.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';unchanged({str(R/n):h for n,h in q['exact_source_SHA256'].items()})
 while not quiet():put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_all_actual_previous_NN_NF_original_EX_release_no_NN',no_original_runner_lock_held=True));time.sleep(20)
 with (H/'readiness_stdout.log').open('xb') as out,(H/'readiness_stderr.log').open('xb') as err:
  c=subprocess.Popen(['python3','-B',str(R/PREP)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own_complete_source_configuration_validation_no_NN',readiness_PID=c.pid));rc=c.wait()
 assert rc==0,dict(readiness_returncode=rc);ready=js(READY/'readiness.json');job=js(J3/'job.json');assert sha(J3/'job.json')==ready['job_SHA256'] and ready['ready_for_unique_original_runner_true3'];batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(job['spec'])
 rc,done=launch(J3,'own_genuine_NEWtrue3_original_probability_fullNF_active')
 if rc!=0:put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='own_genuine_NEWtrue3_or_fullNF_failed_evidence_preserved',job=str(J3),completion=done,no_G_or_canonical_write=True));return rc
 accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['all40_physical_KV_complete_values_verified'] and proof['actual_forward_count']>0 and quiet();cap,p=capacity(accepted)
 if not cap['capacity_passed']:put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='own_true3_fullNF_passed_formal200_capacity_refused',true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_option_shape_capacity.json'),no_approval_NEWfresh3_or200_or_G=True,all_original_values_reports_sources_configs_logs_retained=True));return 1
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_option_shape_capacity.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_all200_option_shape_capacity.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);put(H/'own_first200_measurement_approval.json',approval)
 newproto='configs/protocols/glm41base_own_original_option_logprob_BF16_approved_first200_v1_20261122.json';newcfg='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Base-own-original-option-logprob-BF16-approved-first200-v1-20261122.yaml';J=B/'GLM41Base_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122';assert not (R/newproto).exists() and not (R/newcfg).exists() and not J.exists();p2=copy.deepcopy(p);p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')});p2['first200_not_approved']=False;put(R/newproto,p2);oldcfg=R/job['spec']['config'];oldproto=js(oldcfg)['model']['kwargs']['protocol_file'];cfg=js(oldcfg);cfg['name']=Path(newcfg).stem;cfg['model']['kwargs'].update(protocol_file=str(R/newproto),protocol_sha256=sha(R/newproto));put(R/newcfg,cfg)
 spec=copy.deepcopy(job['spec']);spec['source_sha256']={n:h for n,h in spec['source_sha256'].items() if n not in [job['spec']['config'],str(Path(oldproto).relative_to(R))]};spec.update(approved=True,config=newcfg,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));spec['source_sha256'].update({newcfg:sha(R/newcfg),newproto:sha(R/newproto)});scope(spec,[3,200],False);batch.validate_config(cfg,MID)
 for n in [3,200]:
  path,c=batch.config_for(spec,js(D/'state.json')['models'][MID],n);assert c['sample_range']==f'0:{n-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(spec);j=copy.deepcopy(job);j.update(spec=spec,stages=[3,200],created_at=now());J.mkdir();put(J/'job.json',j);rc,done=launch(J,'approved_NEWfresh3_fixed200_original_probability_active');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='NEWfresh3_fixed200_natural_actual_completion',job=str(J),completion=done,no_G_or_canonical_write=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise

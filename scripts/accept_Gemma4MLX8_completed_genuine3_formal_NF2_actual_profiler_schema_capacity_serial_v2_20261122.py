"""Same genuine3 full NF with current profiler schema, then original capacity and NEWfresh3/fixed200. No model replay of completed3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,fcntl,hashlib,json,math,os,shutil,subprocess,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='unsloth/gemma-4-E4B-it-MLX-8bit';J3=B/'Gemma4E4BMLX8_own_original_Transformers_BF16_true3_v1_real_FLOCK_20261122'
H=B/'Gemma4E4BMLX8_completed_genuine3_formal_NF2_actual_profiler_schema_then_capacity_serial_v2_20261122'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
NF2='scripts/verify_gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v2_actual_profiler_schema_20261122.py'
SELF='scripts/accept_Gemma4MLX8_completed_genuine3_formal_NF2_actual_profiler_schema_capacity_serial_v2_20261122.py'
CONFIG='configs/queues/gemma4_e4b_MLX8_own_original_Transformers_BF16_true3_capacity_serial_v1_20261122.json'

def now():return datetime.now(timezone.utc).isoformat()

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def js(p):return json.loads(Path(p).read_bytes())

def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)

def quiet():
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 if runner_lock_owners():return False
 roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
 if any(p['PID']!=os.getpid() and p['role'] in roles for p in project_processes()):return False
 for path in Path('/proc').iterdir():
  if not path.name.isdigit() or int(path.name)==os.getpid():continue
  try:args=[a.decode(errors='replace') for a in (path/'cmdline').read_bytes().split(b'\0') if a];cwd=(path/'cwd').resolve()
  except OSError:continue
  if not args or not Path(args[0]).name.startswith('python'):continue
  for arg in args[1:]:
   p=Path(arg);p=p if p.is_absolute() else cwd/p
   if p.is_relative_to(R) and (p.name in ['run.py',RUNNER.name] or p.name.startswith(('verify_','audit_'))):return False
 return True

def launch(job,phase):
 with (H/(phase+'_stdout.log')).open('xb') as out,(H/(phase+'_stderr.log')).open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd='/root',stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,runner_PID=child.pid,job=str(job),job_SHA256=sha(job/'job.json'),max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
 done=js(job/'completion.json');assert done['actual_returncode']==rc;return rc,done

def capacity(accepted):
 import numpy as np
 proof=js(accepted['proof']);line=js(Path(proof['report_dir'])/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]
 assert len(front)==200 and [r['index'] for r in trace]==[0,1,2]
 entries=[];parts=[]
 for row in trace:
  sample=js(row['actual_execution_file']);assert sha(row['actual_execution_file'])==row['actual_execution_SHA256'];archive=Path(sample['archive']);assert sha(archive)==sample['archive_SHA256'] and archive.stat().st_size==sample['archive_bytes']
  with zipfile.ZipFile(archive) as z:
   retained={name.replace('.','_')+'.npy' for name in sample['activation_prefill']};members={i.filename:i for i in z.infolist()};assert retained.issubset(members) and len(members)==len(z.namelist())
   prefix_payload=sum(members[n].compress_size for n in retained)
   for item in z.infolist():
    with z.open(item) as file:
     version=np.lib.format.read_magic(file);assert version in [(1,0),(2,0)];shape,fortran,dtype=(np.lib.format.read_array_header_1_0(file) if version==(1,0) else np.lib.format.read_array_header_2_0(file));assert not dtype.hasobject
    parts.append(dict(index=sample['index'],name=item.filename,shape=list(shape),dtype=dtype.str,fortran_order=fortran,compressed_bytes=item.compress_size,complete_NPY_bytes=item.file_size,first3_only_prefill_member=item.filename in retained,archive=str(archive),archive_SHA256=sample['archive_SHA256']))
  common=sample['archive_bytes']-prefix_payload;assert common>0
  entries.append(dict(index=sample['index'],prompt_length=sample['prompt_length'],forwards=sample['actual_native_forwards'],archive=str(archive),archive_SHA256=sample['archive_SHA256'],archive_bytes=sample['archive_bytes'],first3_only_prefill_payload_bytes=prefix_payload,all_sample_mandatory_payload_and_headers_bytes=common))
 full3=sum(e['archive_bytes'] for e in entries);base=max(e['all_sample_mandatory_payload_and_headers_bytes'] for e in entries);minimum=min(e['prompt_length'] for e in entries)
 lengths=[x['inputs']['input_ids']['shape'][1] for x in front];forecast=[math.ceil(base*max(1.,(n/minimum)**2)) for n in lengths[3:]];needed=4*(2*full3+sum(forecast))+6*2**30
 cap=dict(passed=True,UTC=now(),model_id=MID,true3_proof_SHA256=accepted['proof_sha256'],all_complete_ZIP_parts=parts,entries=entries,all200_original_prompt_lengths=lengths,all200_original_input_shapes=[dict(index=i,inputs=x['inputs'],actual_soft_tokens=x['actual_soft_tokens']) for i,x in enumerate(front)],all197_forecast_mandatory_common_bytes=forecast,forecast_formula='4*(2*actual_full3+sum197(max_common*max(1,(own_prompt/min_observed_prompt)**2)))+6GiB',empirical_estimate_bytes=needed,actual_free_bytes=shutil.disk_usage(D).free,empirical_not_2048_worst_case_guarantee=True,original_factor=4,original_reserve_bytes=6*2**30,hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_guard_bytes=2*2**30,configured_outer10_actual_inner1_unchanged=True,no_matrix_deleted=True,no_model_math_precision_budget_or_retention_change=True)
 cap['capacity_passed']=cap['actual_free_bytes']>=needed;put(H/'own_completed3_all_ZIP_all200_shape_capacity.json',cap);return cap,p

def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_completed_exit
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
 assert quiet() and not H.exists();H.mkdir()
 job=js(J3/'job.json');assert sha(J3/'job.json')=='9d4fd5c1c2b4359f203bfc229e6f4b8a70bbd40e466b9edba75becd37f9e7c9a'
 old=js(J3/'completion.json');assert old['actual_returncode']==old['model_returncode']==old['verifier_returncode']==1 and old['phase']=='failed_evidence_preserved' and old['accepted']==[]
 folder=Path(old['report_dir']);validate_completed_exit(1,folder)
 original_nf=R/job['spec']['verifier'];assert sha(original_nf)=='6f34e6fb44347adb14b3d768d863885e67953639a8caa63fd6dec510dfc424ad'
 before_reports={p.name:sha(p) for p in folder.iterdir()};assert set(before_reports)=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
 scope(job['spec'],[3],False);batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);core_unchanged()
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock')
  with (H/'formal_NF2_stdout.json').open('xb') as out,(H/'formal_NF2_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(R/NF2),str(folder),'3'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
   put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='formal_full_NF2_same_genuine3_no_NEW_model_forward',verifier_PID=child.pid,actual_NEW_model_forwards=0,original_job=str(J3),original_job_SHA256=sha(J3/'job.json')));rc=child.wait()
  put(H/'formal_NF2_actual_process_returncode.json',dict(UTC=now(),actual_returncode=rc,actual_NEW_model_forwards=0));assert rc==0
  proof=js(H/'formal_NF2_stdout.json');line=js(folder/'operator_lineage.json')['actual_execution'];original_proof=Path(line['evidence_root'])/'independent_report_verification_profiler_v2.json'
  assert js(original_proof)==proof and proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==24 and proof['all2130_own_source_actual_NPU_and_unused_SHA256_verified'] and proof['all_native_KV_values_losslessly_reconstructed'] and proof['all_full_head_rows_independently_recomputed']
  assert {p.name:sha(p) for p in folder.iterdir()}==before_reports and js(J3/'completion.json')==old
  score=validate_report(R,folder,MID,'image-text-to-text',3,proof,fresh_after=datetime.fromisoformat(old['started_at']).timestamp())
  proof=copy.deepcopy(proof);proof.update(actual_NEW_model_forwards_in_this_acceptance=0,formal_same_current_genuine_NPU3_acceptance=True,original_genuine_NPU_job=str(J3),original_genuine_NPU_job_SHA256=sha(J3/'job.json'),original_runner_failure_kept=True,original_NF1_profiler_schema_failure_kept=True,new_NF2_only_actual_profiler_schema_and_output_name_changed=True,original_full_NF2_proof=str(original_proof),original_full_NF2_proof_SHA256=sha(original_proof),formal_acceptance_controller_source_SHA256=sha(R/SELF))
  output=H/'formal_same_genuine3_fullNF2_acceptance_proof.json';put(output,proof);accepted=dict(sample_count=3,score=score,report_dir=str(folder),proof=str(output),proof_sha256=sha(output))
  put(H/'formal_same_genuine3_acceptance.json',dict(UTC=now(),actual_returncode=0,accepted=accepted,actual_NEW_model_forwards=0,original_NF_and_runner_returncode=1,formal_NF2_returncode=0,original_job_not_modified=True))
 assert quiet();cap,p=capacity(accepted)
 if not cap['capacity_passed']:
  put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='same_genuine3_fullNF2_passed_formal200_capacity_refused',accepted=accepted,capacity_review=str(H/'own_completed3_all_ZIP_all200_shape_capacity.json'),no_approval_NEWfresh3_or200_or_G=True));return 1
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],formal_same_current_genuine3_NF2_acceptance=str(H/'formal_same_genuine3_acceptance.json'),formal_same_current_genuine3_NF2_acceptance_SHA256=sha(H/'formal_same_genuine3_acceptance.json'),capacity_review=str(H/'own_completed3_all_ZIP_all200_shape_capacity.json'),capacity_review_SHA256=sha(H/'own_completed3_all_ZIP_all200_shape_capacity.json'),small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);ap=H/'first200_measurement_approval.json';put(ap,approval)
 proto='configs/protocols/gemma4_e4b_MLX8_completed_true3_fullNF2_original_Transformers_BF16_approved_first200_v2_20261122.json';cfg='configs/Gemma4ForConditionalGeneration/unsloth--gemma-4-E4B-it-MLX-8bit-completed-true3-fullNF2-original-Transformers-BF16-approved-first200-v2-20261122.yaml';J200=B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122';assert not (R/proto).exists() and not (R/cfg).exists() and not J200.exists()
 p2=copy.deepcopy(p);p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(ap):sha(ap),str(H/'formal_same_genuine3_acceptance.json'):sha(H/'formal_same_genuine3_acceptance.json'),str(H/'own_completed3_all_ZIP_all200_shape_capacity.json'):sha(H/'own_completed3_all_ZIP_all200_shape_capacity.json')});p2['implementation_source_SHA256'].update({NF2:sha(R/NF2),SELF:sha(R/SELF)});p2['first200_not_approved']=False;put(R/proto,p2)
 c2=js(R/job['spec']['config']);oldproto=c2['model']['kwargs']['protocol_file'];c2['name']=Path(cfg).stem;c2['model']['kwargs'].update(protocol_file=str(R/proto),protocol_sha256=sha(R/proto));put(R/cfg,c2)
 spec=copy.deepcopy(job['spec']);spec['source_sha256']={n:h for n,h in spec['source_sha256'].items() if n not in [job['spec']['config'],str(Path(oldproto).relative_to(R))]};spec.update(approved=True,config=cfg,verifier=NF2,alignment_approval_review=str(ap),alignment_approval_review_sha256=sha(ap));spec['source_sha256'].update({cfg:sha(R/cfg),proto:sha(R/proto),SELF:sha(R/SELF),NF2:sha(R/NF2),CONFIG:sha(R/CONFIG)})
 scope(spec,[3,200],False);batch.validate_config(c2,MID)
 for count in [3,200]:
  path,config=batch.config_for(spec,js(D/'state.json')['models'][MID],count);assert config['sample_range']==f'0:{count-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(spec)
 following=copy.deepcopy(job);following.update(spec=spec,stages=[3,200],created_at=now());J200.mkdir();put(J200/'job.json',following)
 rc,done=launch(J200,'approved_NEWfresh3_fixed200_original_runner_active');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='NEWfresh3_fixed200_natural_actual_completion',job=str(J200),completion=done,no_G_or_canonical_write=True));return rc
if __name__=='__main__':
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True,all_original_values_reports_sources_configs_logs_retained=True))
  raise

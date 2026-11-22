"""Private exact current profiler schema repair and same-genuine3 formal acceptance."""
from pathlib import Path
import ast,json,hashlib
V=Path(__file__).resolve().parent
old=(V/'verify_gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v1_20261122.py').read_text(encoding='utf-8')
start=old.index("    tasks=[Path(x) for x in profile")
end=old.index('    score=sum(scores)/count',start)
block='''    tasks=[Path(x) for x in profile if Path(x).name=='task_time.csv' or re.fullmatch(r'task_time_[^.]+\\.csv',Path(x).name)];kernel_count=0
    assert tasks,'Actual complete native task_time/kernel_details correspondence is required'
    observed_schemas=[]
    for t in tasks:
        with t.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        core_types={'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'}
        kernel_types={'KERNEL_AIVEC','KERNEL_MIX_AIV','KERNEL_AICPU','KERNEL_AICORE','KERNEL_MIX_AIC'}
        selected=[r for r in rows if r['kernel_type'] in core_types|kernel_types]
        assert selected and all(r['Device_id']=='4' and r['kernel_name']!='N/A' and Decimal(r['task_time(us)'].strip())>0 for r in selected)
        actual_ids=[(r['Device_id'],r['stream_id'],r['task_id']) for r in selected];assert len(actual_ids)==len(set(actual_ids))
        detail_path=check(t.with_name('kernel_details.csv'))
        with detail_path.open(encoding='utf-8-sig',newline='') as f:detail_rows=list(csv.DictReader(f))
        assert detail_rows and all(r['Device_id']=='4' and r['Name']!='N/A' and Decimal(r['Duration(us)'].strip())>0 for r in detail_rows)
        if 'Stream ID' in detail_rows[0] and 'Task ID' in detail_rows[0]:
            assert all(r['Accelerator Core'] in core_types for r in detail_rows)
            detail_ids=[(r['Device_id'],r['Stream ID'],r['Task ID']) for r in detail_rows]
            assert len(actual_ids)==len(detail_ids)==len(set(detail_ids)) and set(actual_ids)==set(detail_ids)
            observed_schemas.append('native_AI_CORE_device_stream_task_exact_full_join')
        else:
            assert all(r['kernel_type'] in kernel_types for r in selected) and all(r['Accelerator Core']=='N/A' for r in detail_rows)
            # This actual native KERNEL schema omits stream/task IDs in the
            # details CSV. Exact original decimal timestamp+duration is unique
            # for every kernel; match the entire table, never a sampled subset.
            actual_keys=[(r['Device_id'],Decimal(r['task_start(us)'].strip()),Decimal(r['task_time(us)'].strip())) for r in selected]
            detail_keys=[(r['Device_id'],Decimal(r['Start Time(us)'].strip()),Decimal(r['Duration(us)'].strip())) for r in detail_rows]
            assert len(actual_keys)==len(set(actual_keys))==len(detail_keys)==len(set(detail_keys)) and set(actual_keys)==set(detail_keys)
            observed_schemas.append('native_KERNEL_unique_device_exact_start_duration_full_join')
        kernel_count+=len(selected)
    assert kernel_count>100,'Real original vision and prefill NPU kernels required'
'''
new=old[:start]+block+old[end:]
new=new.replace("home/'independent_report_verification.json'","home/'independent_report_verification_profiler_v2.json'")
# All numerical operations, head/KV/mask/input/source checks and thresholds stay byte-for-byte.
assert new[:start]==old[:start] and new[new.index('    score=sum(scores)/count'):].replace("home/'independent_report_verification_profiler_v2.json'","home/'independent_report_verification.json'")==old[end:]
name='verify_gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v2_actual_profiler_schema_20261122.py'
ast.parse(new);(V/name).write_text(new,encoding='utf-8',newline='\n')

controller=(V/'continue_gemma4_e4b_MLX8_own_original_Transformers_BF16_true3_capacity_serial_v1_20261122.py').read_text(encoding='utf-8')
tree=ast.parse(controller)
functions='\n\n'.join(ast.get_source_segment(controller,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['now','sha','js','put','quiet','launch','capacity'])
header='''"""Same genuine3 full NF with current profiler schema, then original capacity and NEWfresh3/fixed200. No model replay of completed3."""
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
'''
main='''
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
'''
wrapper=header+'\n'+functions+'\n'+main
ast.parse(wrapper)
wrappername='accept_Gemma4MLX8_completed_genuine3_formal_NF2_actual_profiler_schema_capacity_serial_v2_20261122.py'
(V/wrappername).write_text(wrapper,encoding='utf-8',newline='\n')
records=[]
for rel in ['scripts/'+name,'scripts/'+wrappername,'scripts/windows_controls/'+Path(__file__).name]:
 p=V/Path(rel).name;records.append(dict(path=rel,local_path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
(V/'Gemma4MLX8_completed3_profiler_NF2_formal_serial_primary_paths_20261122.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False))

"""Own Turkish original serial genuine3/fullNF after same Base actual release, no extra small3 D gate."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='ytu-ce-cosmos/Turkish-LLaVA-v0.1';H=B/'TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122';READY=B/'TurkishLlava_own_original_Transformers_FP16_2048_source_readiness_v1_own_original_20261122';J3=B/'TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';PREPARE='scripts/prepare_turkish_llava_own_original_Transformers_FP16_2048_true3_v1_own_original_20261122.py';SELF='scripts/continue_turkish_llava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122.py';CONFIG='configs/queues/turkish_llava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122.json'
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
def js(path):return json.loads(Path(path).read_bytes())
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()));temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(temp,path)
def quiet():
    from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
    if runner_lock_owners():return False
    roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
    if any(p['PID']!=os.getpid() and p['role'] in roles for p in project_processes()):return False
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:args=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
        except OSError:continue
        if not args or not Path(args[0]).name.startswith('python'):continue
        if any(arg.startswith(str(R)+'/') and (Path(arg).name in ['run.py',RUNNER.name] or Path(arg).name.startswith(('verify_','audit_'))) for arg in args[1:]):return False
    return True
def launch(job,phase):
    with (H/(phase+'_stdout.log')).open('xb') as out,(H/(phase+'_stderr.log')).open('xb') as err:
        child=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=phase,runner_PID=child.pid,job=str(job),job_SHA256=sha(job/'job.json'),max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
    done=js(job/'completion.json');assert done['actual_returncode']==rc;return rc,done
def capacity(accepted):
    import numpy as np
    proof=js(accepted['proof']);report=js(Path(proof['report_dir'])/'eval_report.json');line=report['operator_lineage']['actual_execution'];protocol=js(line['protocol_file']);front=js(protocol['own_frontend200']);trace=[json.loads(z) for z in Path(line['per_sample_evidence']).read_text().splitlines()];assert [r['index'] for r in trace]==[0,1,2]
    parts=[];size3=0;metadata=0
    for entry in trace:
        record=Path(entry['group_execution']);assert sha(record)==entry['group_execution_SHA256'];group=js(record);archive=Path(group['archive']);assert sha(archive)==group['archive_SHA256'];size3+=archive.stat().st_size;metadata+=record.stat().st_size;compressed=0
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist())==len(set(z.namelist()))==group['unique_full_NPY_values']
            for item in z.infolist():
                with z.open(item) as file:
                    version=np.lib.format.read_magic(file);assert version in [(1,0),(2,0)];shape,fortran,dtype=(np.lib.format.read_array_header_1_0(file) if version==(1,0) else np.lib.format.read_array_header_2_0(file));assert not dtype.hasobject
                parts.append(dict(index=entry['index'],name=item.filename,shape=list(shape),dtype=dtype.str,fortran_order=fortran,compressed_bytes=item.compress_size,complete_uncompressed_NPY_bytes=item.file_size,archive=str(archive),archive_SHA256=group['archive_SHA256']));compressed+=item.compress_size
        metadata+=archive.stat().st_size-compressed
    lengths=[len(row['input_ids']) for row in front];shapes=[dict(index=i,pixels=[1,3,336,336],CLIP_sequence=577,original576_projected_visual_tokens=576,text_prefill_sequence=length,physical_KV_heads=8,physical_KV_dim=128,original_generation_budget=2048) for i,length in enumerate(lengths)]
    ratio=max(1.,max(lengths)/min(lengths[:3]));ratio=max(ratio,ratio**2)
    required=math.ceil((size3+metadata)*200/3*ratio*4+6*2**30)
    cap=dict(UTC=now(),model_id=MID,all_complete_ZIP_parts=parts,all200_own_original_full_shapes=shapes,all200_original_prompt_lengths=lengths,original_unknown_axis_conservative_max_growth=ratio,complete_three_archive_bytes=size3,complete_metadata_bytes=metadata,original_factor=4,original_reserve_bytes=6*2**30,hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_reserve_bytes=2*2**30,outer_actual_guard_batch=1,original_configured_outer_batch_size=10,empirical_estimate_bytes=required,actual_free_bytes=shutil.disk_usage(R).free,not_a_2048_worst_case_guarantee=True,no_original_values_precision_budget_math_or_guards_reduced=True)
    cap['capacity_passed']=cap['actual_free_bytes']>=required;put(H/'own_passed3_complete_all200_shape_capacity.json',cap);return cap,protocol
def main():
    import batch_910b as batch
    from A100_alignment_evidence_runner_20261103 import scope,unchanged
    queue=js(R/CONFIG);assert queue['model_id']==MID and queue['max_concurrent_models']==1 and not H.exists() and not J3.exists() and READY.exists()
    H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';unchanged({str(R/n):h for n,h in queue['exact_source_SHA256'].items()})
    previous_job=Path(queue['previous_Base_job']);previous_done=js(previous_job/'completion.json');assert sha(previous_job/'completion.json')==queue['previous_Base_completion_SHA256'] and previous_done['actual_returncode']==1 and previous_done['verifier_returncode']==1 and previous_done['accepted']==[]
    error_file=previous_job/'fresh3/verification_stderr.log';assert sha(error_file)==queue['previous_Base_verifier_stderr_SHA256'] and queue['exact_previous_visual_failure'] in error_file.read_text()
    old_wait=B/'TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_own_original_20261122';old_exit=js(old_wait/'actual_completion.json');assert old_exit['actual_returncode']==1 and 'Previous Base task ended with an unresolved NN/NF/configuration failure' in old_exit['error']
    ready=js(READY/'readiness.json');old_job=Path(ready['job']);assert sha(old_job/'job.json')==ready['job_SHA256']==queue['own_prepared_unused_job_SHA256'] and not (old_job/'fresh3').exists() and not (old_job/'completion.json').exists() and ready['ready_for_unique_original_runner_true3']
    assert not any((R/'workspace/diagnostics/model_runs').glob('TurkishLlava_own_original_FP16_2048_*/generation_progress.jsonl'))
    def alive_owned(pid,needle):
        p=Path('/proc')/str(pid)/'cmdline'
        try:return needle.encode() in p.read_bytes()
        except OSError:return False
    while not quiet() or alive_owned(1716268,'continue_GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.py') or alive_owned(1719058,'continue_turkish_llava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_own_original_20261122.py'):
        put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_actual_previous_NN_NF_controllers_original_EX_release_no_NN',no_original_runner_lock_held=True));time.sleep(20)
    job=copy.deepcopy(js(old_job/'job.json'));job['spec']['source_sha256'].update({SELF:sha(R/SELF),CONFIG:sha(R/CONFIG)});job.update(created_at=now(),private_stage_type='First own genuine NEW3; reuse passed own input/constructor/YAML readiness, exact previous Base numerical failure fully preserved')
    scope(job['spec'],[3],False);batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(job['spec']);J3.mkdir();put(J3/'job.json',job)
    put(H/'exact_previous_failure_review_and_own_unused_stage_resume.json',dict(UTC=now(),previous_Base_actual_completion=previous_done,previous_Base_completion_SHA256=sha(previous_job/'completion.json'),previous_Base_verifier_stderr_SHA256=sha(error_file),exact_known_failure=queue['exact_previous_visual_failure'],Base_failure_not_accepted_or_reclassified=True,Base_original_3samples_6forwards_and_all_full_values_reports_sources_configs_logs_preserved=True,Base_original_RMSE_limit_not_changed=.02,own_passed_readiness=ready,own_input200_and_CPU_constructor_not_replayed=True,own_original_model_NF_YAML_protocol_math_dtype_head_single_EOS_budget_and_thresholds_unchanged=True,all_previous_NN_NF_controllers_original_EX_released=True,new_private_job=str(J3),new_private_job_SHA256=sha(J3/'job.json'),no_G_or_canonical_write=True))
    rc,done=launch(J3,'own_genuine_NEWtrue3_fullNF_original_runner_active')
    if rc!=0:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='own_genuine_NEWtrue3_or_fullNF_failed_evidence_preserved',job=str(J3),completion=done,no_G_or_canonical_write=True));return rc
    accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['all686_own_source_cast_actual_NPU_SHA256_verified'] and proof['actual_forward_count']>0;assert quiet()
    cap,p=capacity(accepted)
    if not cap['capacity_passed']:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='own_true3_fullNF_passed_formal200_capacity_refused',true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),no_approval_NEWfresh3_or200_or_G=True,all_original_values_reports_sources_configs_logs_retained=True));return 1
    approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_all200_shape_capacity.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);put(H/'own_first200_measurement_approval.json',approval)
    oldcfg=R/job['spec']['config'];oldproto=js(oldcfg)['model']['kwargs']['protocol_file'];newproto='configs/protocols/turkish_llava_own_original_Transformers_FP16_2048_approved_first200_v2_after_known_Base_visual_failure_20261122.json';newcfg='configs/LlavaLlamaForCausalLM/ytu-ce-cosmos--Turkish-LLaVA-v0.1-own-original-Transformers-FP16-2048-approved-first200-v2-after-known-Base-visual-failure-20261122.yaml';J=B/'TurkishLlava_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v2_after_known_Base_visual_failure_real_FLOCK_20261122'
    assert not (R/newproto).exists() and not (R/newcfg).exists() and not J.exists()
    p2=copy.deepcopy(p);p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')});p2['first200_not_approved']=False;put(R/newproto,p2)
    c2=js(oldcfg);c2['name']=Path(newcfg).stem;c2['model']['kwargs'].update(protocol_file=str(R/newproto),protocol_sha256=sha(R/newproto));put(R/newcfg,c2)
    spec=copy.deepcopy(job['spec']);spec['source_sha256']={n:h for n,h in spec['source_sha256'].items() if n not in [job['spec']['config'],str(Path(oldproto).relative_to(R))]};spec.update(approved=True,config=newcfg,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));spec['source_sha256'].update({newcfg:sha(R/newcfg),newproto:sha(R/newproto),SELF:sha(R/SELF),CONFIG:sha(R/CONFIG)})
    scope(spec,[3,200],False);batch.validate_config(c2,MID)
    for n in [3,200]:
        path,config=batch.config_for(spec,js(D/'state.json')['models'][MID],n);assert config['sample_range']==f'0:{n-1}'
        if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
    batch.check_sources(spec);unchanged(job['protected_canonical_hashes']);assert quiet() and batch.device_idle(spec)
    nextjob=copy.deepcopy(job);nextjob.update(spec=spec,stages=[3,200],created_at=now());J.mkdir();put(J/'job.json',nextjob)
    rc,done=launch(J,'approved_NEWfresh3_fixed200_original_runner_active');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='NEWfresh3_fixed200_natural_actual_completion',job=str(J),completion=done,no_G_or_canonical_write=True));return rc
if __name__=='__main__':
    try:sys.exit(main())
    except SystemExit:raise
    except BaseException:
        if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True,all_original_values_reports_sources_configs_logs_retained=True))
        raise

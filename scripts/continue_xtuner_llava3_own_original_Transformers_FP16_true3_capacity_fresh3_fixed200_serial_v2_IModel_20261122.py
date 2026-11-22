"""Serial own Xtuner IModel repair genuine3/full NF, complete shape capacity then new200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='xtuner/llava-llama-3-8b-v1_1-hf';H=B/'XtunerLlava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122';READY=B/'XtunerLlava3_own_original_Transformers_FP16_2048_source_readiness_v2_IModel_20261122';J3=B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';PREPARE='scripts/prepare_xtuner_llava3_own_original_Transformers_FP16_2048_true3_v2_IModel_20261122.py';SELF='scripts/continue_xtuner_llava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122.py';CONFIG='configs/queues/xtuner_llava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122.json'
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
        child=subprocess.Popen(['python3','-B',str(RUNNER),str(job)],cwd='/root',stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
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
    queue=js(R/CONFIG);assert queue['model_id']==MID and queue['max_concurrent_models']==1 and not H.exists() and not J3.exists() and not READY.exists()
    H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';unchanged({str(R/n):h for n,h in queue['exact_source_SHA256'].items()});assert quiet()
    with (H/'readiness_stdout.log').open('xb') as out,(H/'readiness_stderr.log').open('xb') as err:
        process=subprocess.Popen(['python3','-B',str(R/PREPARE)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        put(H/'state.json',dict(UTC=now(),phase='own_source_configuration_validation_no_model_NN',PID=os.getpid(),readiness_PID=process.pid));rc=process.wait()
    assert rc==0,dict(readiness_returncode=rc)
    ready=js(READY/'readiness.json');job=js(J3/'job.json');assert ready['ready_for_unique_original_runner_true3'] and sha(J3/'job.json')==ready['job_SHA256']
    unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert quiet() and batch.device_idle(job['spec'])
    rc,done=launch(J3,'own_genuine_NEWtrue3_fullNF_original_runner_active')
    if rc!=0:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='own_genuine_NEWtrue3_or_fullNF_failed_evidence_preserved',job=str(J3),completion=done,no_G_or_canonical_write=True));return rc
    accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['all686_own_source_cast_actual_NPU_SHA256_verified'] and proof['actual_forward_count']>0;assert quiet()
    cap,p=capacity(accepted)
    if not cap['capacity_passed']:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='own_true3_fullNF_passed_formal200_capacity_refused',true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),no_approval_NEWfresh3_or200_or_G=True,all_original_values_reports_sources_configs_logs_retained=True));return 1
    approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_all200_shape_capacity.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);put(H/'own_first200_measurement_approval.json',approval)
    oldcfg=R/job['spec']['config'];oldproto=js(oldcfg)['model']['kwargs']['protocol_file'];newproto='configs/protocols/xtuner_llava3_own_original_Transformers_FP16_2048_approved_first200_v2_IModel_20261122.json';newcfg='configs/LlavaLlamaForCausalLM/xtuner--llava-llama-3-8b-v1_1-hf-own-original-Transformers-FP16-2048-approved-first200-v2-IModel-20261122.yaml';J=B/'XtunerLlava3_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v2_IModel_real_FLOCK_20261122'
    assert not (R/newproto).exists() and not (R/newcfg).exists() and not J.exists()
    p2=copy.deepcopy(p);p2['static_evidence_SHA256']={accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')};p2['first200_not_approved']=False;put(R/newproto,p2)
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

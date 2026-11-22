"""Formal receive of the same completed genuine3 numerical success; no NN/NF replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='ICTNLP/llava-mini-llama-3.1-8b';H=B/'LlavaMini_completed_formal_single_JSON_NF_acceptance_only_capacity_resume_v4_20261122';J3=B/'LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';SELF='scripts/continue_LlavaMini_completed_formal_fullNF_only_capacity_v4_zipfile_20261122.py';CONFIG='configs/queues/llava_mini_own_original_FP16_true3_capacity_fresh3_fixed200_serial_v2_actual_reserved_devices_20261122.json';NF2='scripts/verify_llava_mini_own_original_FP16_2048_full_values_v2_single_JSON_stdout_20261122.py'

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
    lengths=[len(row['input_ids']) for row in front];shapes=[dict(index=i,pixels=[1,1,3,336,336],CLIP_sequence=577,prefusion_sequence=577+length,text_prefill_sequence=length,physical_KV_heads=8,physical_KV_dim=128,original_generation_budget=2048) for i,length in enumerate(lengths)]
    ratio=max(1.,max(lengths)/min(lengths[:3]));ratio=max(ratio,ratio**2)
    required=math.ceil((size3+metadata)*200/3*ratio*4+6*2**30)
    cap=dict(UTC=now(),model_id=MID,all_complete_ZIP_parts=parts,all200_own_original_full_shapes=shapes,all200_original_prompt_lengths=lengths,original_unknown_axis_conservative_max_growth=ratio,complete_three_archive_bytes=size3,complete_metadata_bytes=metadata,original_factor=4,original_reserve_bytes=6*2**30,hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_reserve_bytes=2*2**30,outer_actual_guard_batch=1,empirical_estimate_bytes=required,actual_free_bytes=shutil.disk_usage(R).free,not_a_2048_worst_case_guarantee=True,no_original_values_precision_budget_math_or_guards_reduced=True)
    cap['capacity_passed']=cap['actual_free_bytes']>=required;put(H/'own_passed3_complete_all200_shape_capacity.json',cap);return cap,protocol

def main():
    import batch_910b as batch
    from batch_910b_support import validate_report,validate_completed_exit
    from A100_alignment_evidence_runner_20261103 import scope,unchanged
    from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
    assert quiet() and not H.exists();H.mkdir()
    assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
    job=js(J3/'job.json');assert sha(J3/'job.json')=='99994d88480b6b9fef627de652cbf1837b05080650389d5243676e3a2787f9b7'
    done=js(J3/'completion.json');assert sha(J3/'completion.json')=='5f5c2194bf584c2525ce8f9693285f0aa25344208d8a2eede6bfaddb4c14114a'
    assert done['actual_returncode']==1 and done['model_returncode']==1 and done['verifier_returncode']==0 and done['phase']=='failed_evidence_preserved' and 'JSONDecodeError: Extra data' in done['error'] and done['accepted']==[]
    scope(job['spec'],[3],False);batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);core_unchanged()
    stdout=J3/'fresh3/verification_stdout.json';assert sha(stdout)=='d69302e20a39f621fb7b61b6314ec0a97289f61a185849aa58934638c7b26753'
    original_nf=R/'scripts/verify_llava_mini_own_original_FP16_2048_full_values_v1_20261122.py';assert sha(original_nf)=='f8ecda229e1269bfc181715de78439401d2bf83e21bbe2919b58eba7f544b73c'
    assert sha(R/NF2)=='96ffaae669faf1c5cc91a1877e96c2b4307997b42367b590db70e953de416d9d'
    old=original_nf.read_text();new=(R/NF2).read_text()
    assert new.replace('all_unique_full_arrays=len(arrays))),file=sys.stderr,flush=True)','all_unique_full_arrays=len(arrays))),flush=True)').replace("/'independent_report_verification_stdout_v2.json'","/'independent_report_verification.json'")==old
    original_acceptance_home=B/'LlavaMini_completed_genuine3_successful_fullNF_single_JSON_formal_acceptance_capacity_serial_v3_20261122'
    formal=original_acceptance_home/'formal_acceptance.json';assert sha(formal)=='d3640947f1e358ce936f7ee41d6b1d3dccea68a8d243c130200199f093dcac9b'
    acceptance=js(formal);assert acceptance['actual_returncode']==0 and acceptance['actual_NEW_model_forwards']==acceptance['actual_NEW_numerical_replay']==0
    accepted=acceptance['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256']=='bf232cfab3c8743b22a5a142716c1b4cd9637e7a774872f16330c27f4a08e375'
    proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==6
    put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='completed_formal_fullNF_proof_reused_only_unfinished_capacity_no_NN_or_numerical_replay',actual_NEW_model_forwards=0,actual_NEW_numerical_replay=0,completed_formal_acceptance=str(formal),completed_formal_acceptance_SHA256=sha(formal)))
    put(H/'completed_formal_acceptance_reuse.json',dict(UTC=now(),formal_acceptance=str(formal),formal_acceptance_SHA256=sha(formal),proof=accepted['proof'],proof_SHA256=accepted['proof_sha256'],all_numerical_acceptance_already_completed_no_replay=True))
    assert quiet();cap,p=capacity(accepted)
    if not cap['capacity_passed']:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='same_genuine3_full_numerical_formally_accepted_STDOUT_fixed_but_first200_capacity_refused',formal_acceptance=str(formal),formal_acceptance_SHA256=sha(formal),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),actual_NEW_model_forwards=0,actual_NEW_numerical_replay=0,no_approval_NEWfresh3_or200_or_G=True,all_original_values_reports_sources_configs_logs_retained=True))
        assert quiet();selector=R/'scripts/select_existing_original_resource_queue_1245_20261117.py';assert sha(selector)=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494'
        with (H/'immediate_next_original_resource_queue_stdout.json').open('xb') as out,(H/'immediate_next_original_resource_queue_stderr.log').open('xb') as err:result=subprocess.run(['python3','-B',str(selector)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        put(H/'immediate_next_original_resource_queue_actual.json',dict(UTC=now(),actual_returncode=result.returncode,selector_SHA256=sha(selector),stdout=str(H/'immediate_next_original_resource_queue_stdout.json'),stdout_SHA256=sha(H/'immediate_next_original_resource_queue_stdout.json'),no_old_main_restart_no_signal_no_G_or_protectedstate_clear=True));return 1
    approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_all200_shape_capacity.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);put(H/'own_first200_measurement_approval.json',approval)
    oldcfg=R/job['spec']['config'];oldproto=js(oldcfg)['model']['kwargs']['protocol_file'];newproto='configs/protocols/llava_mini_own_original_FP16_2048_approved_first200_v4_zipfile_capacity_20261122.json';newcfg='configs/LlavaMiniLlamaForCausalLM/ICTNLP--llava-mini-llama-3.1-8b-own-original-FP16-2048-approved-first200-v4-zipfile-capacity-20261122.yaml';J=B/'LlavaMini_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v4_zipfile_capacity_real_FLOCK_20261122'
    assert not (R/newproto).exists() and not (R/newcfg).exists() and not J.exists()
    p2=copy.deepcopy(p);p2['implementation_source_SHA256'].update({NF2:sha(R/NF2),SELF:sha(R/SELF)});p2['runtime_and_original_source_sha256'].update({str(R/NF2):sha(R/NF2),str(R/SELF):sha(R/SELF)});p2['static_evidence_SHA256']={accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')};p2['first200_not_approved']=False;put(R/newproto,p2)
    c2=js(oldcfg);c2['name']=Path(newcfg).stem;c2['model']['kwargs'].update(protocol_file=str(R/newproto),protocol_sha256=sha(R/newproto));put(R/newcfg,c2)
    spec=copy.deepcopy(job['spec']);spec['verifier']=NF2;spec['source_sha256'][NF2]=sha(R/NF2);spec['source_sha256']={n:h for n,h in spec['source_sha256'].items() if n not in [job['spec']['config'],str(Path(oldproto).relative_to(R))]};spec.update(approved=True,config=newcfg,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'));spec['source_sha256'].update({newcfg:sha(R/newcfg),newproto:sha(R/newproto),SELF:sha(R/SELF),CONFIG:sha(R/CONFIG)})
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

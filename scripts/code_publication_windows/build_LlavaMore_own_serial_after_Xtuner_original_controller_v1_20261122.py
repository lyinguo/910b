"""Build a source-bound max1 successor; no takeover or active-job edits."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent
old=(V/'continue_xtuner_llava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122.py').read_text();tree=ast.parse(old)
def function(name):
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return '\n'.join(old.splitlines()[f.lineno-1:f.end_lineno])+'\n'
name='continue_LlavaMore_own_original_letter_after_Xtuner_actual_release_true3_capacity_fresh3_fixed200_serial_v1_20261122.py'
queue_name='LlavaMore_own_original_letter_after_Xtuner_actual_release_serial_v1_20261122.json'
header='''"""Immediate serial own More probability after real previous completion; no NN while waiting."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning';H=B/'LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122';READY=B/'LlavaMore_own_original_FP16_letter_source_readiness_v1_20261122';J3=B/'LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';PREPARE='scripts/prepare_LlavaMore_own_original_FP16_letter_true3_source_ready_v1_20261122.py';SELF='scripts/CONTROLLER';CONFIG='configs/queues/QUEUE'
PREVIOUS=B/'XtunerLlava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122';PREVIOUS_JOB=B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122'
'''.replace('CONTROLLER',name).replace('QUEUE',queue_name)
common=''.join(function(n) for n in ['now','sha','js','put','quiet','launch','capacity'])
common=common.replace('original_generation_budget=2048','original_generation_budget=2048,original_probability_forward_count=1,original_use_cache=False')
main='''def previous_qualified200_needs_publication():
    p=PREVIOUS/'actual_completion.json'
    if not p.exists():return None
    d=js(p)
    for row in d.get('completion',{}).get('accepted',[]):
        proof=js(row['proof']);assert sha(row['proof'])==row['proof_sha256']
        if proof.get('sample_count')==200 and proof.get('original_protocol_final_qualification_complete') and proof.get('formal_alignment_within_5pct'):return True
    return False

def main():
    import batch_910b as batch
    from A100_alignment_evidence_runner_20261103 import scope,unchanged
    q=js(R/CONFIG);assert q['model_id']==MID and q['max_concurrent_models']==1 and not H.exists()
    H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';unchanged({str(R/n):h for n,h in q['exact_source_SHA256'].items()})
    assert sha(PREVIOUS_JOB/'job.json')==q['previous_immutable_genuine3_job_SHA256']=='f422bbc90f71e573de81f5d62e008375b64785da92dfbdcf24efe24ba35fa114'
    with (H/'readiness_stdout.log').open('xb') as out,(H/'readiness_stderr.log').open('xb') as err:
        child=subprocess.Popen(['python3','-B',str(R/PREPARE)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own_source_preparation_no_model_NN',readiness_PID=child.pid));rc=child.wait()
    assert rc==0,dict(readiness_returncode=rc)
    ready=js(READY/'readiness.json');job=js(J3/'job.json');assert ready['ready_for_unique_original_runner_true3_when_all_real_quiet'] and sha(J3/'job.json')==ready['job_SHA256']
    while True:
        unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);prior=previous_qualified200_needs_publication();clear=quiet()
        put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_previous_real_NN_NF_controller_and_original_EX_release_no_NN',previous_actual_completion_exists=prior is not None,previous_qualified200_publication_must_precede_next_NN=(prior is True),all_real_quiet=clear,own_source_ready=True,genuine_own_NPU3_not_started=True,max_concurrent_models=1,waiter_does_not_hold_original_runner_lock=True,no_G_or_canonical_write=True))
        if prior is False and clear and batch.device_idle(job['spec']):break
        time.sleep(20)
    unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert quiet() and batch.device_idle(job['spec'])
    rc,done=launch(J3,'own_genuine_NEWtrue3_fullNF_original_runner_active')
    if rc!=0:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='own_genuine_NEWtrue3_or_fullNF_failed_evidence_preserved',job=str(J3),completion=done,no_G_or_canonical_write=True));return rc
    accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==3 and proof['all686_own_source_cast_actual_NPU_SHA256_verified'] and proof['original_probability_protocol_not_replaced_by_generation'];assert quiet()
    cap,p=capacity(accepted)
    if not cap['capacity_passed']:
        put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='own_true3_fullNF_passed_formal200_capacity_refused',true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),no_approval_NEWfresh3_or200_or_G=True,all_original_values_reports_sources_configs_logs_retained=True));return 1
    approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'own_passed3_complete_all200_shape_capacity.json'),capacity_review_SHA256=sha(H/'own_passed3_complete_all200_shape_capacity.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,original_factor4_and_all_runtime_guards_unchanged=True,all_real_previous_model_NN_NF_and_original_EX_released=True);put(H/'own_first200_measurement_approval.json',approval)
    oldcfg=R/job['spec']['config'];oldproto=js(oldcfg)['model']['kwargs']['protocol_file'];newproto='configs/protocols/llava_more_own_original_FP16_letter_pad_approved_first200_v1_20261122.json';newcfg='configs/LlavaLlamaForCausalLM/aimagelab--LLaVA_MORE-llama_3_1-8B-finetuning-own-original-FP16-letter-pad-approved-first200-v1-20261122.yaml';J=B/'LlavaMore_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122'
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
'''
text=header+common+main;ast.parse(text);(V/name).write_text(text,encoding='utf8',newline='\n')
primary={}
for file in ['LlavaMore_own_native_probability_primary_paths_v1_20261122.json','LlavaMore_own_probability_fullNF_primary_paths_v1_20261122.json','parallel_NPZ_complete_storage_primary_paths_20261122.json']:primary.update(json.loads((V/file).read_bytes()))
primary['scripts/'+name]=hashlib.sha256(text.encode()).hexdigest()
prep='prepare_LlavaMore_own_original_FP16_letter_true3_source_ready_v1_20261122.py';ast.parse((V/prep).read_bytes());primary['scripts/'+prep]=hashlib.sha256((V/prep).read_bytes()).hexdigest()
queue=dict(model_id='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning',max_concurrent_models=1,previous_immutable_genuine3_job_SHA256='f422bbc90f71e573de81f5d62e008375b64785da92dfbdcf24efe24ba35fa114',exact_source_SHA256={k:h for k,h in primary.items() if not k.startswith('scripts/code_publication_windows/')},wait_previous_actual_NN_NF_controller_and_original_EX_release=True,qualified_previous200_must_be_dual_published_before_next_NN=True,original_probability_letter_mode_not_generation=True,no_G_or_canonical_write=True)
qtext=json.dumps(queue,ensure_ascii=False,indent=2)+'\n';(V/queue_name).write_text(qtext,encoding='utf8',newline='\n');primary['configs/queues/'+queue_name]=hashlib.sha256(qtext.encode()).hexdigest();primary['scripts/code_publication_windows/'+Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(V/'LlavaMore_own_original_probability_true3_all_primary_v1_20261122.json').write_text(json.dumps(primary,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(primary=primary,max_concurrent_models=1,no_current_Xtuner_source_job_deadline_or_signal_change=True)))

"""Bind own completed frontend, original FP16 graph and full verifier to NEW3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,inspect,json,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='ICTNLP/llava-mini-llama-3.1-8b'
FRONT=B/'LlavaMini_own_original_FP16_frontend200_meta_v3_actual_full_npz_interface_no_learned_NN_20261122'
H=B/'LlavaMini_own_original_FP16_2048_source_readiness_v1_20261122';J=B/'LlavaMini_own_original_FP16_2048_true3_v1_real_FLOCK_20261122'
MODEL='universal_eval/model/adapters/llava_mini_own_original_FP16_2048_native_v1_20261122.py';ALGO='universal_eval/algorithm/llava_mini_own_original_FP16_2048_MC_20261122.py';NF='scripts/verify_llava_mini_own_original_FP16_2048_full_values_v1_20261122.py';SELF='scripts/prepare_llava_mini_own_original_FP16_2048_true3_v1_20261122.py'
CFG='configs/LlavaMiniLlamaForCausalLM/ICTNLP--llava-mini-llama-3.1-8b-own-original-FP16-2048-native-v1-20261122.yaml';PROTO='configs/protocols/llava_mini_own_original_FP16_2048_native_v1_20261122.json'
def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
def js(path):return json.loads(Path(path).read_bytes())
def create(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as file:json.dump(value,file,ensure_ascii=False,indent=2)
def main():
    import batch_910b as batch
    from A100_alignment_evidence_runner_20261103 import scope,unchanged
    from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
    assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]
    H.mkdir(exist_ok=False);ready=js(FRONT/'readiness.json');assert sha(FRONT/'readiness.json')=='8b6ab57ab658964ab1b2b465f543e1e33a64ef4b005f756e26ca7316eb14948c' and ready['passed'] and ready['model_id']==MID and ready['all200_own_full_F32_pixels_FP16_cast_tokens_exact']
    for name,digest in ready['source_sha256'].items():assert sha(R/name)==digest,name
    assert sha(ready['own_frontend200'])==ready['own_frontend200_SHA256']
    assert sha(R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')=='0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1'
    from batch_910b_support import workbook_rows
    row,=[r for r in workbook_rows(R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')[0] if r['model_id']==MID];assert row['row']==365 and row['score'] is None and row['task']=='image-text-to-text'
    parent=js(B/'OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121/job.json');protected={name:sha(name) for name in parent['protected_canonical_hashes']}
    assert sha(D/'state.json')=='b001dfe5c178d8ba1191ba8ec894fbd8e7ab81b62a50193262e6d8a7fe4cd450' and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd'
    rels=list(ready['source_sha256'])+[MODEL,ALGO,NF,SELF,'scripts/egm8b_original_independent_cpu_910b.py','scripts/strict_mc_independent_reference_910b.py','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py','universal_eval/utils/readonly_npz_indexed.py','universal_eval/model/interface.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py']
    rels=list(dict.fromkeys(rels))
    for name in rels:ast.parse((R/name).read_bytes())
    importlib.import_module(MODEL[:-3].replace('/','.'));importlib.import_module(ALGO[:-3].replace('/','.'))
    sources={name:sha(R/name) for name in rels};external={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.is_file() and not path.is_relative_to(R):external[str(path)]=sha(path)
    protocol=copy.deepcopy(ready);protocol.update(implementation_source_SHA256=sources,runtime_and_original_source_sha256={**{str(R/name):digest for name,digest in sources.items()},**external},current_A100_D_values=ready['three_D_values'],hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_lossless_space_estimate_bytes=2*2**30,original_factor4_plus6GiB_capacity_guard=True,first200_not_approved=True)
    create(R/PROTO,protocol)
    config=dict(name=Path(CFG).stem,description='Own732 source; original FP16 CLIP24/compressor1/prefusion4/Llama32/full128256 head, plain RoPE500000 original adapter branch, max2048/outer1/3EOS. Complete own frontend200 reused, private original4D empty cache, native model math unchanged; complete observations and independent full values. Historical A100 runtime uncaptured.',task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=ready['dataset_path']),model=dict(name=MID,adapter=MODEL+':OwnLlavaMiniOriginalFP16',model_path=ready['own_weight_path'],download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_device=4)),algorithm=dict(name='llava_mini_own_original_FP16_greedy2048',adapter=ALGO+':OwnLlavaMiniGreedy',batch_size=1,kwargs=dict(extract_answer=True)),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnLlavaMiniAccuracy'],include_failed_predictions=True),baseline=dict(score=.745,threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(R/CFG,config)
    spec=copy.deepcopy(parent['spec']);spec.update(approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)},review_revision=config['description'])
    for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
    job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own FP16 original2048 genuine NEW3 full numerical verification; no first200 approval')
    scope(spec,[3],False);batch.validate_config(config,MID);batch.check_sources(spec)
    entry=js(D/'state.json')['models'][MID]
    for n in [3,200]:_,small=batch.config_for(spec,entry,n);assert small['sample_range']==f'0:{n-1}'
    create(J/'job.json',job);unchanged(protected)
    value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,ready_for_unique_original_runner_true3=True,model_id=MID,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=ready['three_D_values'],NPU_device=4,max_concurrent_models=1,actual_NEW_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True)
    create(H/'readiness.json',value);create(H/'actual_completion.json',dict(UTC=value['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if H.exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
        raise

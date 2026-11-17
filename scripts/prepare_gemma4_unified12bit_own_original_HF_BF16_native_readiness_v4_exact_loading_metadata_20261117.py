"""Bind own Unified12B original HF batch1 to the frozen private runner.

No original adapter mock initializer or learned CPU/NPU forward is executed.
The already independently verified own frontend200 is reused byte-for-byte.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,json,hashlib,importlib,inspect,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='google/gemma-4-12B-it';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4UnifiedForConditionalGeneration')/MID
H=B/'Gemma4Unified12Bit_own_original_HF_BF16_native_v4_exact_loading_metadata_readiness_20261117'
F=B/'Gemma4Unified12Bit_own_original_HF_false_thinking_frontend200_exact_names_20261116'
M=B/'Gemma4Unified12Bit_own_original_source_assets_META_only_next_candidate_20261116'
J=B/'Gemma4Unified12Bit_own_original_HF_BF16_true3_v4_exact_loading_metadata_real_FLOCK_20261117'

def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def create(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    import batch_910b as batch,yaml
    from batch_910b_support import validate_config,workbook_rows
    from A100_alignment_evidence_runner_20261103 import scope
    from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
    assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
    pre=js(F/'review.json');assert pre['passed'] and pre['samples']==200 and pre['own_source_count']==677 and pre['all200_F32_pixels_positions_Rust_IDs_mask_media_equal'] and js(F/'actual_completion.json')['actual_returncode']==0
    assert sha(F/'frontend200.jsonl')==pre['frontend_SHA256']=='fd159ff39030ea943578c82383ce8c85c8b1ff7a7e77690a163b1b0ae80d19e0'
    assert js(M/'actual_completion.json')['actual_returncode']==0
    saved=R/'configs/Gemma4UnifiedForConditionalGeneration/google--gemma-4-12B-it.yaml';old=yaml.safe_load(saved.read_bytes())
    assert old['model']['name']==MID and old['model']['adapter']=='universal_eval/model/adapters/gemma4_unified_vlm.py:Gemma4UnifiedVLLMModel'
    assert old['model']['kwargs']==dict(max_new_tokens=2048,torch_dtype='bfloat16',enable_thinking=False) and old['algorithm']==dict(name='greedy_inference',batch_size=1,kwargs=dict(extract_answer=True)) and old['metrics']['override']==['Accuracy']
    book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');bind=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==bind['local_workbook_sha256']
    state=js(D/'state.json');entry=state['models'][MID];assert state['current'] is None and entry['score'] is None
    rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['score'] is None and row['task']==entry['task']=='image-text-to-text'
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(book) as z:d=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
    preserved=Path(bind['local_A100_reference_snapshot']);current=Path(bind['current_user_workbook_D_reference_snapshot'])
    assert sha(preserved)==bind['local_A100_reference_snapshot_sha256'] and sha(current)==bind['current_user_workbook_D_reference_snapshot_sha256']
    origD=str(js(preserved)['models'][MID]['A100_D']);localD=str(js(current)['models'][MID]['A100_D']);assert all(float(v)==.95 for v in [d,origD,localD])
    H.mkdir(exist_ok=False);reference=H/'immutable_equal_verified175_workbook_at_selection.xlsx';reference.write_bytes(book.read_bytes());assert sha(reference)==sha(book)=='2f05d90e41b69081eb347bf8d40d0761b5988504a660e4846cefe905eb3001cd'
    native='universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_clean_worker_v4_exact_loading_metadata_20261117.py';verifier='scripts/verify_gemma4_unified_12bit_own_BF16_clean_worker_v4_exact_loading_metadata_20261117.py'
    sources=[native,verifier,'universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_native_v4_exact_loading_metadata_20261117.py','universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_native_v1_20261117.py','scripts/verify_gemma4_unified_12bit_own_BF16_native_v1_20261117.py','scripts/gemma4_unified12bit_genuine_clean_worker_v4_exact_loading_metadata_20261117.py','universal_eval/model/gemma4_unified_12bit_own_20261117/__init__.py','universal_eval/model/gemma4_unified_12bit_own_20261117/observer.py','universal_eval/model/reference_sources/gemma4_unified_owned_checkpoint_layout.py',
             'scripts/prepare_gemma4_unified12bit_own_original_HF_BF16_native_readiness_v4_exact_loading_metadata_20261117.py','scripts/continue_gemma4_unified12bit_own_original_HF_BF16_true3_fresh3_fixed200_v4_exact_loading_metadata_20261117.py','scripts/measure_gemma4_unified12bit_own_original_HF_BF16_actual_capacity_v4_exact_loading_metadata_20261117.py',
             'scripts/prepare_gemma4unified12bit_own_source_assets_META_no_learned_CPU_20261116.py','scripts/prepare_gemma4unified12bit_own_frontend200_and_exact_META_names_20261116.py',
             'universal_eval/model/adapters/gemma4_unified_vlm.py','universal_eval/model/interface.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py',str(saved.relative_to(R))]
    core_old=ast.parse((R/'universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_native_v1_20261117.py').read_bytes());core_new=ast.parse((R/'universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_native_v4_exact_loading_metadata_20261117.py').read_bytes())
    for method in ['predict','batch_predict','get_execution_lineage','get_sample_execution']:
        before=next(n for n in ast.walk(core_old) if isinstance(n,ast.FunctionDef) and n.name==method);after=next(n for n in ast.walk(core_new) if isinstance(n,ast.FunctionDef) and n.name==method);assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    for name in sources:
        if name.endswith('.py'):ast.parse((R/name).read_bytes())
    static={str(path):sha(path) for path in [F/'review.json',F/'frontend200.jsonl',F/'exact_own_677_source_678_native_META_mapping.json',M/'own_tensor_headers_and_shards.json',M/'genuine_installed_META_contract.json',saved,preserved,current,D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json']}
    static.update(pre['own_assets_SHA256']);static.update(pre['original_processor_source_SHA256'])
    for path in ['/usr/local/python3.12.16/lib/python3.12/site-packages/transformers/models/gemma4_unified/modeling_gemma4_unified.py','/usr/local/python3.12.16/lib/python3.12/site-packages/transformers/cache_utils.py']:static[path]=sha(path)
    boundary='Own genuine Unified12B original HF Greedy2048 outer1/serial1 BF16 with original enable_thinking=False, F32 merged48px/280 patch inputs, own48 layers slidingGQA8/head256 and fullGQA1/head512, K-equals-V full layers and zero shared KV. Own677 BF16 tensors and tied own head; exact legacy vision name mapping only, no checkpoint replacement. The original CLI parent compatibility bootstrap is isolated from a fresh genuine learned worker; the learned worker does not import or use that initializer. Workbook D is authorized A100 comparison; historical A100 engine/tensor bytes are not claimed captured. Original filter_emptyTrue is replaced solely by the mandatory full fixed200 no-filter setting, with all200 identical own verified inputs and full denominator retained.'
    proto=dict(model_id=MID,own_original_root=str(W),own_headers_manifest=str(M/'own_tensor_headers_and_shards.json'),own_META_contract=str(M/'genuine_installed_META_contract.json'),frontend200=str(F/'frontend200.jsonl'),dataset_path=str(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),dataset_SHA256=pre['dataset_SHA256'],implementation_source_SHA256={n:sha(R/n) for n in sources},static_evidence_SHA256=static,current_A100_D_values=dict(server=d,local=localD,preserved_original=origD),workbook_D_reference_authorized=True,saved_original_configuration_used=True,saved_original_YAML=str(saved),saved_original_YAML_SHA256=sha(saved),preserved_original_snapshot=str(preserved),preserved_original_snapshot_SHA256=sha(preserved),reference_workbook=str(reference),reference_workbook_SHA256=sha(reference),hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3,historical_source_boundary=boundary)
    pp=R/'configs/protocols/gemma4_unified12bit_own_original_HF_BF16_native_v4_exact_loading_metadata_20261117.json';create(pp,proto)
    parent=js(B/'RedHatGemma4E2B_own_original_BF16_true3_v1_saved_HF_F32_real_FLOCK_20261116/job.json');cfg=js(R/parent['spec']['config'])
    cf=R/'configs/Gemma4UnifiedForConditionalGeneration/google--gemma-4-12B-it-own-original-HF-BF16-2048-native-v4-exact-loading-metadata-20261117.yaml';cfg.update(name=cf.stem,description=boundary,task_type=row['task']);cfg['model']=dict(name=MID,adapter=native+':OwnUnified12BitCleanWorker',model_path=str(W),download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]));cfg['algorithm']=copy.deepcopy(old['algorithm']);cfg['baseline'].update(score=float(d),primary_metric='Accuracy');create(cf,cfg)
    spec=dict(approved=False,review_revision='Unified12Bit_own_original_HF_BF16_serial1_2048_genuine_clean_worker_exact_48_persistent_scalars_v3',config=str(cf.relative_to(R)),verifier=verifier,npu_devices=[4],source_sha256={n:sha(R/n) for n in sources+[str(cf.relative_to(R)),str(pp.relative_to(R))]},smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
    batch.check_sources(spec)
    for count in [3,200]:_,effective=batch.config_for(spec,entry,count);validate_config(effective,MID);assert effective['algorithm']['batch_size']==1
    scope(spec,[3],False);module=importlib.import_module('universal_eval.model.adapters.gemma4_unified_12bit_own_BF16_clean_worker_v4_exact_loading_metadata_20261117');assert not inspect.isabstract(module.OwnUnified12BitCleanWorker)
    job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={n:sha(n) for n in parent['protected_canonical_hashes']});create(J/'job.json',job)
    ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,ready_for_original_runner_true3=True,true3_job=str(J),job_SHA256=sha(J/'job.json'),standalone_YAML=str(cf),current_three_D_values=proto['current_A100_D_values'],original_outer_batch1_not_ordinary_Gemma4_outer10=True,no_NPU_before_original_runner=True,frontend200_reused_not_recomputed=True)
    create(H/'readiness.json',ready);print(json.dumps(ready,ensure_ascii=False))

if __name__=='__main__':main()

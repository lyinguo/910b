"""Bind full own Bee inputs/source, independent YAML/verifier and original locked true3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,importlib,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='Open-Bee/Bee-8B-RL';H=B/'Bee8BRL_own_original_BF16_4096_native_readiness_true3_20261116';J=B/'Bee8BRL_own_original_BF16_4096_true3_v1_real_FLOCK_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 pre=B/'Bee8BRL_own_original_frontend200_v1_no_NPU_20261116/readiness.json';ready=js(pre);assert ready['passed'] and ready['all_own_source_and_meta_parameter_names_shapes_bijective'] and ready['all200_own_original_tokens_pixels_boundaries_exact'] and ready['source_tensor_count']==827
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 d=cells['D'+str(row['row'])];preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 historical=js(preserved)['models'][MID];local=js(current)['models'][MID];assert historical['task']==local['task']==entry['task']
 rels=[str(p.relative_to(R)) for p in (R/'universal_eval/model/bee8brl_own_original_20261116').glob('*.py')]
 rels+=['universal_eval/model/adapters/bee8brl_own_frontend_bf16_20261116.py','universal_eval/model/adapters/bee8brl_own_native_bf16_4096_20261116.py','scripts/verify_bee8brl_own_native_bf16_4096_20261116.py','scripts/prepare_bee8brl_own_frontend200_20261116.py','scripts/prepare_bee8brl_own_readiness_true3_20261116.py','scripts/prepare_bee8brl_own_resource_meta_v5_actual_shards_BF16_native_vision_prefix_20261116.py','universal_eval/model/adapters/bee_adapter.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','universal_eval/data/loader.py','universal_eval/data/registry.py','universal_eval/data/preprocessor.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels};boundary='User-confirmed A100 workbook D, not an online A100 replay prerequisite. Own saved BeeAdapter and YAML: torch_dtype auto maps to BF16, own Siglip26/384 anyres_max6 + own Qwen3_36 GQA8x128/152000 head, original4096 greedy outer10/serial1. Own827 source tensors are BF16 and preserve original BF16 execution exactly as saved adapter load; complete source/cast/actualNPU SHA required. Original four shard files and original index are unmodified; original index has stale shard locations, while actual unique whole-file headers have the exact full original index key set. Native version removes only the old vision_model parameter wrapper name, with full name/shape bijection. Original Bee chat_template enable_thinking=False is a saved adapter default and is preserved, not a newly introduced suppression; no template or checkpoint borrowed. Original direct-forward first-token EOS boundary is preserved, no early EOS or shorter budget. Native HF SDPA and native causal positions/cache are disclosed. Historical A100 runtime/tensors/checkpoint bytes were not captured and are not claimed.'
 proto={k:copy.deepcopy(ready[k]) for k in ['model_id','weight_path','source_tensor_count','own_weight_files','own_assets_SHA256','own_source_manifest','own_source_manifest_SHA256','own_frontend200','own_frontend200_SHA256','dataset_path','dataset_SHA256','original_stop_ids','original_budget4096_preserved']}
 static=[pre,preserved,current,Path(ready['parent_resource_review']),Path(ready['actual_shard_inventory']),R/'configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-RL.yaml']
 proto.update(workbook_D_reference_authorized=True,reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=d,local=local['A100_D'],preserved_original=historical['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in static},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 pp=R/'configs/protocols/bee8brl_own_original_BF16_4096_native_v1_20261116.json';create(pp,proto)
 cfg=js(R/'configs/MiniCPMO/openbmb--MiniCPM-o-2_6-own-original-BF16-4096-native-v5-authorized-ledger-approved-first200-20261116.yaml');config=R/'configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-RL-own-original-BF16-4096-native-v1-20261116.yaml'
 cfg.update(name=config.stem,description=boundary);cfg['data'].update(dataset_path=ready['dataset_path'],download=False,filter_empty=False)
 cfg['model']=dict(name=MID,adapter='universal_eval/model/adapters/bee8brl_own_native_bf16_4096_20261116.py:BeeRLOwnNativeBF16',model_path=ready['weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]))
 cfg['baseline']['score']=float(d);validate_config(cfg,MID);create(config,cfg)
 spec=dict(approved=False,review_revision='Bee8BRL_own_original_BF16_4096_native_v1_20261116',config=str(config.relative_to(R)),verifier='scripts/verify_bee8brl_own_native_bf16_4096_20261116.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.bee8brl_own_native_bf16_4096_20261116')
 parent=js(B/'MiniCPMO26_authorized_ledger_CAS_BF16_4096_approved_fresh3_fixed200_v5_real_FLOCK_20261116/job.json')
 job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});create(J/'job.json',job)
 create(H/'readiness.json',dict(passed=True,ready_for_original_runner_true3=True,model_id=MID,source_tensor_count=827,standalone_YAML=str(config),job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=proto['current_A100_D_values'],no_NPU_inference_yet=True,historical_source_boundary=boundary));print(json.dumps(js(H/'readiness.json'),ensure_ascii=False))
if __name__=='__main__':main()

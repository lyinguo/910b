"""Bind own original MiniCPM-V-4.6 protocol to standalone YAML and one private real-NPU3 job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,copy,zipfile,xml.etree.ElementTree as E,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='openbmb/MiniCPM-V-4.6';H=B/'MiniCPMV46_own_original_native_readiness_v4_original_cache_dict_20261116';J=B/'MiniCPMV46_own_original_BF16_2048_true3_v4_original_cache_dict_real_FLOCK_20261116'
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
 pre=B/'MiniCPMV46_own_original_BF16_MCq_fullsource_frontend200_v3_original_fused_uint8_tuple_20261116/readiness.json';ready=js(pre);assert ready['passed'] and ready['all200_own_input_pixels_tokens_checked']
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 d=cells['D'+str(row['row'])];preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 historical=js(preserved)['models'][MID];local=js(current)['models'][MID];assert historical['task']==local['task']==entry['task']
 rels=['universal_eval/model/adapters/minicpmv46_own_frontend_20261116.py','universal_eval/model/adapters/minicpmv46_own_native_bf16_2048_v4_original_cache_dict_20261116.py','universal_eval/model/reference_sources/minicpmv46_original_mcq_protocol.py','universal_eval/model/reference_sources/minicpmv46_original_window_merger_patch.py','scripts/verify_minicpmv46_own_native_bf16_2048_v2_original_EOS_20261116.py','scripts/prepare_minicpmv46_own_native_source_frontend200_v3_original_fused_uint8_tuple_20261116.py','scripts/prepare_minicpmv46_own_native_true3_v4_original_cache_dict_20261116.py','scripts/continue_minicpmv46_own_true3_capacity_fresh3_fixed200_v4_original_cache_dict_20261116.py','universal_eval/model/interface.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','universal_eval/data/loader.py','universal_eval/data/registry.py','universal_eval/data/preprocessor.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels}
 boundary='User-confirmed workbook A100 D; own MiniCPM-V-4.6 original saved MCQ adapter and BF16/2048/outer4-inner1 deployment. Original static two-format examples, own chat template without enable_thinking override, original reference-independent normalization and original per-image vision-window patch are preserved, not newly introduced. Own779 BF16 tensors plus own declared tied head248094, own vision27 and Qwen3_5 text24(18 native linear recurrent layers plus6 full-attention layers) execute on NPU4. Native original F32 recurrent accumulation/analytic RoPE are retained inside the BF16 architecture; no learned CPU inference/CPU grade. Full input/head/all24 hybrid-state matrices are archived losslessly, own masks/positions/stops/cache and original2048 are not reduced. Installed native torch recurrent fallback is used, no CUDA-only kernel stub/package. A100 historical engine/tensor bytes were not captured and are not claimed. Original parser cases that could mistake an ordinary article for A are separately flagged and cannot qualify for publication.'
 fields=['model_id','weight_path','source_tensor_count','own_weight_files','own_assets_SHA256','own_source_manifest','own_source_manifest_SHA256','own_native_meta_layout','own_native_meta_layout_SHA256','own_frontend200','own_frontend200_SHA256','dataset_path','dataset_SHA256','native_libraries_SHA256','original_stop_ids'];proto={k:copy.deepcopy(ready[k]) for k in fields}
 proto.update(workbook_D_reference_authorized=True,reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=d,local=local['A100_D'],preserved_original=historical['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in [pre,preserved,current,R/'universal_eval/model/adapters/minicpmv4_6_mcq.py',R/'universal_eval/model/adapters/transformers_vlm.py',R/'configs/MiniCPMV4_6ForConditionalGeneration/openbmb--MiniCPM-V-4.6.yaml']},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 proto['original_stop_ids']=js(Path(ready['weight_path'])/'generation_config.json')['eos_token_id'];assert proto['original_stop_ids']==[248044,248046]
 proto['original_frontend_ready_stop_metadata_was_single_EOS_and_corrected_only_in_new_binding']=True
 pp=R/'configs/protocols/minicpmv46_own_original_BF16_2048_native_v4_original_cache_dict_20261116.json';create(pp,proto)
 cfg=js(R/'configs/MiniCPMV/openbmb--MiniCPM-V-4_5-own-original-BF16-2048-native-v3-approved-first200-20261115.yaml');config=R/'configs/MiniCPMV4_6ForConditionalGeneration/openbmb--MiniCPM-V-4.6-own-original-BF16-2048-native-v4-original-cache-dict-20261116.yaml'
 cfg.update(name=config.stem,description=boundary);cfg['data'].update(dataset_path=ready['dataset_path'],download=False,filter_empty=False);cfg['algorithm']['batch_size']=4
 cfg['model']=dict(name=MID,adapter='universal_eval/model/adapters/minicpmv46_own_native_bf16_2048_v4_original_cache_dict_20261116.py:MiniCPMV46OwnNativeOriginalCacheDict',model_path=ready['weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]));cfg['baseline']['score']=float(d);validate_config(cfg,MID);create(config,cfg)
 spec=dict(approved=False,review_revision='MiniCPMV46_own_original_native_BF16_2048_20261116',config=str(config.relative_to(R)),verifier='scripts/verify_minicpmv46_own_native_bf16_2048_v2_original_EOS_20261116.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,MID)
 scope(spec,[3],False);module=importlib.import_module('universal_eval.model.adapters.minicpmv46_own_native_bf16_2048_v4_original_cache_dict_20261116')
 import torch
 from transformers import AutoConfig,MiniCPMV4_6ForConditionalGeneration
 meta_cfg=AutoConfig.from_pretrained(ready['weight_path'],local_files_only=True,trust_remote_code=False);meta_cfg._attn_implementation=meta_cfg.text_config._attn_implementation=meta_cfg.vision_config._attn_implementation='eager'
 with torch.device('meta'):meta=MiniCPMV4_6ForConditionalGeneration(meta_cfg)
 observer=module.MiniCPMV46OwnNativeOriginalCacheDict(str(pp),sha(pp),{'model_id':MID},[4]);observer.model=meta;observer.handles=[];observer._install()
 assert len(meta.model.vision_tower.encoder.layers)==27 and len(meta.model.language_model.layers)==24 and len(observer.handles)>50
 for handle in observer.handles:handle.remove()
 del observer,meta

 parent=js(B/'VCFlash7B_own_original_BF16_2048_approved_fresh3_fixed200_v2_effective_defaults_real_FLOCK_20261116/job.json');job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});create(J/'job.json',job)
 result=dict(passed=True,ready_for_original_runner_true3=True,model_id=MID,own_source_tensor_count=779,current_three_D=proto['current_A100_D_values'],standalone_YAML=str(config),job=str(J),job_SHA256=sha(J/'job.json'),no_NPU_inference_yet=True,historical_source_boundary=boundary);create(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

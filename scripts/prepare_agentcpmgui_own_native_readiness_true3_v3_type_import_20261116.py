"""Bind own AgentCPM-GUI full source/frontend to independent YAML and private true3 job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,importlib,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='openbmb/AgentCPM-GUI';H=B/'AgentCPMGUI_own_original_native_readiness_v2_unpadded_vllm_user_continue_20261116';J=B/'AgentCPMGUI_own_original_BF16_2048_true3_v2_real_FLOCK_user_continue_20261116'
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
 pre=B/'AgentCPMGUI_own_original_fullsource_frontend200_v3_type_import_user_continue_20261116/readiness.json';ready=js(pre)
 assert ready['passed'] and ready['all_own_source_and_meta_parameter_names_shapes_bijective'] and ready['all200_own_original_tokens_pixels_boundaries_exact']
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 d=cells['D'+str(row['row'])]
 preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 historical=js(preserved)['models'][MID];local=js(current)['models'][MID];assert historical['task']==local['task']==entry['task']
 rels=[str(p.relative_to(R)) for p in (R/'universal_eval/model/agentcpmgui_own_original_20261116').glob('*.py')]
 rels+=['universal_eval/model/adapters/agentcpmgui_own_frontend_bf16_v2_unpadded_vllm.py','universal_eval/model/adapters/agentcpmgui_own_native_bf16_v2_unpadded_vllm.py','scripts/verify_agentcpmgui_own_native_bf16_v2_unpadded_vllm.py','scripts/prepare_agentcpmgui_own_native_fullsource_frontend200_v3_type_import_20261116.py','scripts/prepare_agentcpmgui_own_native_readiness_true3_v3_type_import_20261116.py','universal_eval/model/adapters/minicpmv4_original_pixel_check_v1.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','universal_eval/data/loader.py','universal_eval/data/registry.py','universal_eval/data/preprocessor.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels}
 boundary='User-confirmed original workbook A100 D is the comparison. Own saved AgentCPM-GUI VLLMModel BF16/Greedy outer10 inner1/2048, own Qwen2_28 GQA4x128 + own NavitSiglip27/resampler64, own448 max9slice/151679 head and raw tokenizer/chat template; original generation_config EOS exactly preserved. Current own native HF eager reconstruction and CPU integer pixel geometry/analytic RoPE are disclosed; historical A100 vLLM engine/backend/tensors were not captured. Full own source functions unchanged. No borrowed checkpoint/data/tokenizer/template, no original HF chat default beam3 or repetition1.2 substituted for saved VLLM greedy. Original standard input preprocessing/fixed200 and strict MC retain the original Greedy instruction; all own image boundaries and per-patch pixels independently verified.'
 contract=B/'AgentCPMGUI_original_own_HF_mask_vs_saved_VLLM_unpadded_contract_v2_user_continue_20261116/review.json'
 boundary+=' The own HF processor ne0 mask is retained verbatim separately from the saved vLLM unpadded sequence mask; no borrowed tokenizer or GUI template. Its original mask is retained verbatim as diagnostic. Saved vLLM passes token ids and absolute positions without that HF token-value mask, so the candidate original unpadded causal-mask math was not changed; exact current vLLM static source and original HF mask are archived, no historical engine capture claim.'
 proto={k:copy.deepcopy(ready[k]) for k in ['model_id','weight_path','source_tensor_count','own_weight_files','own_assets_SHA256','own_source_manifest','own_source_manifest_SHA256','own_frontend200','own_frontend200_SHA256','dataset_path','dataset_SHA256','original_stop_ids']}
 proto.update(workbook_D_reference_authorized=True,reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=d,local=local['A100_D'],preserved_original=historical['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in [pre,preserved,current,contract,B/'AgentCPMGUI_own_resampler_type_import_only_recovery_20261116/review.json']},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 pp=R/'configs/protocols/agentcpmgui_own_original_BF16_2048_native_v2_unpadded_vllm.json';create(pp,proto)
 cfg=js(R/'configs/MiniCPMV/openbmb--MiniCPM-V-4-own-native-BF16-2048-v4-original-preprocessing-20261115.yaml')
 config=R/'configs/MiniCPMV/openbmb--AgentCPM-GUI-own-original-BF16-2048-native-v2-unpadded-vllm-20261116.yaml'
 cfg.update(name=config.stem,description=boundary)
 cfg['data'].update(dataset_path=ready['dataset_path'],download=False,filter_empty=False)
 cfg['model']=dict(name=MID,adapter='universal_eval/model/adapters/agentcpmgui_own_native_bf16_v2_unpadded_vllm.py:AgentCPMGUIOwnNativeBF16V2',model_path=ready['weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]))
 cfg['baseline']['score']=float(d);validate_config(cfg,MID);create(config,cfg)
 spec=dict(approved=False,review_revision='AgentCPMGUI_own_original_native_BF16_2048_v1_20261116',config=str(config.relative_to(R)),verifier='scripts/verify_agentcpmgui_own_native_bf16_v2_unpadded_vllm.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.agentcpmgui_own_native_bf16_v2_unpadded_vllm')
 parent=js(B/'DotsOCR_own_original_BF16_OCR12_approved_fresh3_fixed200_v6_real_FLOCK_user_continue_20261115/job.json')
 job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});create(J/'job.json',job)
 ready=dict(passed=True,ready_for_original_runner_true3=True,model_id=MID,own_source_tensor_count=ready['source_tensor_count'],original_stop_ids=proto['original_stop_ids'],current_three_D=proto['current_A100_D_values'],standalone_YAML=str(config),job=str(J),job_SHA256=sha(J/'job.json'),no_NPU_inference_yet=True,historical_source_boundary=boundary)
 create(H/'readiness.json',ready);print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':main()

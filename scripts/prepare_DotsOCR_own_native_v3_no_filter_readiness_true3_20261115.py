"""Bind own DotsOCR original OCR/CER protocol to standalone YAML and frozen runner."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,importlib,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='rednote-hilab/dots.ocr';H=B/'DotsOCR_own_original_BF16_OCR12_native_readiness_v3_no_filter_user_continue_20261115';J=B/'DotsOCR_own_original_BF16_OCR12_true3_v3_real_FLOCK_user_continue_20261115'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 pre=B/'DotsOCR_own_original_source_frontend200_v3_exact_original_rescale_user_continue_20261115/readiness.json'
 ready=js(pre);assert ready['passed'] and ready['all200_original_project_preprocessing_tokens_pixels_checked'] and ready['own_meta_parameter_names_shapes_bijective']
 state=js(D/'state.json');entry=state['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:c',ns)}
 server=cells['D583'];assert float(server)==.283 and cells.get('G583') in [None,'']
 preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 historical=js(preserved)['models'][MID];local=js(current)['models'][MID];assert historical['task']==local['task']==entry['task']=='image-text-to-text'
 own_dir=R/'universal_eval/model/dotsocr_own_original_sources_20261115'
 rels=[str(p.relative_to(R)) for p in own_dir.glob('*.py')]
 rels+=['universal_eval/model/adapters/dotsocr_own_frontend_bf16_v2_original_rescale.py','universal_eval/model/adapters/dotsocr_own_native_bf16_v2_original_rescale.py','scripts/verify_dotsocr_own_native_bf16_v3_fixed200_no_filter.py','scripts/prepare_dotsocr_own_fullsource_frontend200_v3_original_rescale_20261115.py','scripts/prepare_DotsOCR_own_native_v3_no_filter_readiness_true3_20261115.py','universal_eval/algorithm/dotsocr_original_greedy_evidence_v1.py','universal_eval/algorithm/inference.py','universal_eval/metric/ocr.py','universal_eval/data/loader.py','universal_eval/data/registry.py','universal_eval/data/preprocessor.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels}
 boundary='User-confirmed workbook A100 D is the comparison. Own saved DotsOCR VLLMModel OCR3000/fixed0:199, original max_tokens12, Greedy serial1 with original final-answer suffix and OCR extraction/CER; own BF16 weights and 42-block original vision/28-Qwen2/151936 head, native 28 GQA2x128 cache, original own max11289600 pixels and patch14/merge2, own multimodal chat_template.json and own imgpad151665. Original stored VLLM engine/template dispatch/backend bytes were not captured; current native Qwen2 eager and original missing-FA2 VisionAttention fallback disclosed, no historical bitwise claim. No generic mock/stub initialization, own source functions unchanged, no new token ids or borrowed assets. Independent pixel reference follows installed original F64 multiply then F32 normalization; actual processor pixels were not altered.'
 front=json.loads(Path(ready['own_frontend200']).read_bytes())
 assert len(front)==200 and all(r['prompt'] and r['reference'] and Path(r['media_path']).is_file() for r in front)
 # Preflight retained all3000 original rows, so false filtering changes no original fixed200 input.
 assert ready['all200_original_project_preprocessing_tokens_pixels_checked']
 boundary+=' All3000 original OCR rows survived the original preprocessing; fixed200 is kept without filtering, and every frozen original input/ref/image is verified unchanged. Downloads explicitly disabled.'
 proto={k:copy.deepcopy(ready[k]) for k in ['model_id','weight_path','source_tensor_count','own_weight_files','own_assets_SHA256','own_source_manifest','own_source_manifest_SHA256','own_frontend200','own_frontend200_SHA256','dataset_path','dataset_SHA256']}
 proto.update(workbook_D_reference_authorized=True,reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=server,local=local['A100_D'],preserved_original=historical['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(x):sha(x) for x in [pre,preserved,current]},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 pp=R/'configs/protocols/dotsocr_own_original_BF16_OCR12_native_v3_fixed200_no_filter.json';put(pp,proto)
 cfg=dict(name='rednote-hilab--dots.ocr-own-original-BF16-OCR12-native-v3-fixed200-no-filter-20261115',description=boundary,task_type='image-text-to-text',is_ocr=True,sample_range='0:199',data=dict(download=False,preprocess=True,apply_template=False,filter_empty=False,dataset_path=ready['dataset_path']),model=dict(name=MID,adapter='universal_eval/model/adapters/dotsocr_own_native_bf16_v2_original_rescale.py:DotsOCROwnNativeBF16V2',model_path=ready['weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='dotsocr_original_greedy_evidence',adapter='universal_eval/algorithm/dotsocr_original_greedy_evidence_v1.py:DotsOriginalGreedyEvidence',batch_size=1,kwargs=dict(extract_answer=True,ocr_extract=True)),metrics=dict(override=['CER'],adapters=['universal_eval/algorithm/dotsocr_original_greedy_evidence_v1.py:DotsOriginalCER'],include_failed_predictions=True),baseline=dict(score=float(server),threshold=.05),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
 config=R/'configs/DotsOCRForCausalLM/rednote-hilab--dots.ocr-own-original-BF16-OCR12-native-v3-fixed200-no-filter-20261115.yaml';put(config,cfg);validate_config(cfg,MID)
 spec=dict(approved=False,review_revision='DotsOCR_own_original_BF16_OCR12_fullsource_workbookD_v2_20261115',config=str(config.relative_to(R)),verifier='scripts/verify_dotsocr_own_native_bf16_v3_fixed200_no_filter.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:_,candidate=batch.config_for(spec,entry,count);validate_config(candidate,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.dotsocr_own_native_bf16_v2_original_rescale')
 parent=js(B/'MiniCPMV4_original_preprocessing_approved_fresh3_fixed200_v5_real_FLOCK_user_continue_20261115/job.json')
 job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});put(J/'job.json',job)
 result=dict(passed=True,ready_for_original_runner_true3=True,own_source_tensor_count=ready['source_tensor_count'],current_three_A100_D_values=proto['current_A100_D_values'],standalone_YAML=str(config),protocol=str(pp),job=str(J),job_SHA256=sha(J/'job.json'),no_NPU_inference_yet=True,historical_source_boundary=boundary)
 put(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

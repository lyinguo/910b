"""Bind full own GLM-OCR inputs/source, independent YAML/verifier and original locked true3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,importlib,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='unsloth/GLM-OCR';H=B/'GLMOCRUnsloth_own_original_BF16_4096_native_readiness_true3_20261116';J=B/'GLMOCRUnsloth_own_original_BF16_4096_true3_v1_real_FLOCK_20261116'
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
 pre=B/'GLMOCRUnsloth_own_original_BF16_4096_frontend200_meta_v1_exact_F32_fused_constants_no_NPU_20261116/readiness.json';ready=js(pre);assert ready['passed'] and ready['all200_own_original_tokens_pixels_boundaries_exact'] and ready['source_tensor_count']==510
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 d=cells['D'+str(row['row'])];preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 historical=js(preserved)['models'][MID];local=js(current)['models'][MID];assert historical['task']==local['task']==entry['task']
 rels=['universal_eval/model/adapters/glmocr_unsloth_own_frontend_20261116.py','universal_eval/model/adapters/glmocr_unsloth_own_native_bf16_4096_20261116.py','scripts/verify_glmocr_unsloth_own_native_bf16_4096_v1_CER_primary_20261116.py','scripts/prepare_glmocr_unsloth_own_readiness_true3_20261116.py','scripts/prepare_glmocr_unsloth_own_original_4096_frontend_meta_v1_exact_F32_constants_20261116.py','universal_eval/model/adapters/transformers_vlm.py','universal_eval/algorithm/inference.py','universal_eval/metric/ocr.py','universal_eval/data/loader.py','universal_eval/data/registry.py','universal_eval/data/preprocessor.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels};boundary='User-confirmed workbook D is the comparison reference; online A100 replay is not an extra first200 gate. Own GLM-OCR BF16 checkpoint526 tensors, saved installed standard HF16 main decoder/vision24 path uses510 and does not execute16 auxiliary MTP parameters. All526 physical own tensors are hashed,510 active ones must match actual NPU. Own Glm46V processor/chat template uses the saved generic GlmOcr enable_thinking=False branch, not newly changed after a measurement. Original4096/greedy/outer1 serial1/native16cache/EOS59246,59253/CER OCR data and extraction retained. No mock initializer, package install, borrowed checkpoint/data/tokenizer/font, altered model math or shortened budget. Saved YAML names this same model at /mnt2/afs/download_models; current original-model inventory owns the checkpoint at /mnt/afs/huggingfaceautoclonemodels, whose own whole-file and per-tensor SHA are frozen. Historical path bytes and A100 engine/tensor bytes were not captured and are not claimed. No other model checkpoint is substituted.'

 proto={k:copy.deepcopy(ready[k]) for k in ['model_id','weight_path','source_tensor_count','own_weight_files','own_assets_SHA256','own_source_manifest','own_source_manifest_SHA256','own_frontend200','own_frontend200_SHA256','dataset_path','dataset_SHA256','original_stop_ids','original_BF16_4096_outer1_inner1_CER_protocol_preserved','own_unused_MTP_source_manifest','own_unused_MTP_source_manifest_SHA256','original_physical_source_tensor_count','original_unused_MTP_source_tensor_count']}
 static=[pre,preserved,current,Path(ready['own_unused_MTP_source_manifest']),R/'configs/GlmOcrForConditionalGeneration/unsloth--GLM-OCR.yaml']
 proto.update(workbook_D_reference_authorized=True,reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=d,local=local['A100_D'],preserved_original=historical['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in static},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 pp=R/'configs/protocols/glmocr_unsloth_own_original_BF16_4096_native_v1_20261116.json';create(pp,proto)
 cfg=js(R/'configs/MiniCPMO/openbmb--MiniCPM-o-2_6-own-original-BF16-4096-native-v5-authorized-ledger-approved-first200-20261116.yaml');config=R/'configs/GlmOcrForConditionalGeneration/unsloth--GLM-OCR-own-original-BF16-4096-native-v1-20261116.yaml'
 cfg.update(name=config.stem,description=boundary);cfg['data'].update(dataset_path=ready['dataset_path'],download=False,filter_empty=False);cfg['algorithm']=dict(name='greedy_inference',batch_size=1,kwargs=dict(extract_answer=True,ocr_extract=True));cfg['metrics']=dict(override=['CER']);cfg['is_ocr']=True
 cfg['model']=dict(name=MID,adapter='universal_eval/model/adapters/glmocr_unsloth_own_native_bf16_4096_20261116.py:UnslothGlmOcrOwnNativeBF16',model_path=ready['weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]))
 cfg['baseline']['score']=float(d);cfg['baseline']['primary_metric']='CER';validate_config(cfg,MID);create(config,cfg)
 spec=dict(approved=False,review_revision='GLMOCRUnsloth_own_original_BF16_4096_native_v1_20261116',config=str(config.relative_to(R)),verifier='scripts/verify_glmocr_unsloth_own_native_bf16_4096_v1_CER_primary_20261116.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.glmocr_unsloth_own_native_bf16_4096_20261116')
 parent=js(B/'MiniCPMO26_authorized_ledger_CAS_BF16_4096_approved_fresh3_fixed200_v5_real_FLOCK_20261116/job.json')
 job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});create(J/'job.json',job)
 create(H/'readiness.json',dict(passed=True,ready_for_original_runner_true3=True,model_id=MID,source_tensor_count=510,standalone_YAML=str(config),job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=proto['current_A100_D_values'],no_NPU_inference_yet=True,historical_source_boundary=boundary));print(json.dumps(js(H/'readiness.json'),ensure_ascii=False))
if __name__=='__main__':main()

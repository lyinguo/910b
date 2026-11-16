"""Bind own Falcon original task/source and real small-sample private job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,importlib,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='tiiuae/Falcon-OCR';H=B/'FalconOCR_own_original_BF16_512_native_readiness_true3_20261116';J=B/'FalconOCR_own_original_BF16_512_true3_v1_real_FLOCK_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import numpy as np,torch
 import batch_910b as batch
 from batch_910b_support import validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 from scripts.verify_falconocr_own_original_bf16_512_20261116 import independent_pixels
 pre=B/'FalconOCR_own_original512_source_dtype_CPU_frontend200_v3_actual_original_dataset_20261116/review.json';ready=js(pre);assert ready['passed'] and ready['samples']==200 and ready['source_tensor_count']==115
 ctl=B/'FalconOCR_real_NPU_attention_predicate_LSE_sinks_FFN_primitive_controls_20261116/review.json';assert js(ctl)['passed'] and js(ctl)['no_model_forward']
 H.mkdir(exist_ok=False);front=[json.loads(x) for x in Path(ready['own_frontend200']).read_text().splitlines()];pixel_checks=[]
 for i,row in enumerate(front):
  p,m=independent_pixels(row['media_path']);assert hashlib.sha256(p.tobytes()).hexdigest()==row['tensors']['pixel_values']['SHA256'] and hashlib.sha256(m.tobytes()).hexdigest()==row['tensors']['pixel_mask']['SHA256'],('Independent original pixels',i)
  pixel_checks.append(dict(index=i,F32_pixels_and_mask_independent_uint8_PIL_formula_exact=True))
 create(H/'independent_all200_input_pixels_review.json',dict(passed=True,samples=200,checks=pixel_checks,own_frontend200_SHA256=ready['own_frontend200_SHA256'],no_CPU_or_NPU_grade=True))
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns)}
 d=cells['D'+str(row['row'])];preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 historical=js(preserved)['models'][MID];local=js(current)['models'][MID];assert historical['task']==local['task']==entry['task']
 rels=[str(p.relative_to(R)) for folder in ['universal_eval/model/falconocr_own_npu_20261116','universal_eval/model/falconocr_own_original_20261116'] for p in sorted((R/folder).glob('*.py'))]
 rels+=['universal_eval/model/adapters/falconocr_own_original_bf16_native_512_20261116.py','scripts/verify_falconocr_own_original_bf16_512_20261116.py','scripts/prepare_falconocr_own_native512_readiness_true3_20261116.py','scripts/continue_falconocr_own_true3_full_shape_capacity_fresh3_fixed200_20261116.py','scripts/prepare_falconocr_own_original512_frontend200_v3_actual_dataset_20261116.py','scripts/diagnostic_falconocr_npu_original_attention_sinks_ffn_20261116.py','universal_eval/model/adapters/transformers_vlm.py','universal_eval/algorithm/inference.py','universal_eval/metric/ocr.py','universal_eval/data/loader.py','universal_eval/data/registry.py','universal_eval/data/preprocessor.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels};boundary=ready['source_original_hardware_boundary']+' Own saved HF plain/category, BF16 parameters/F32 golden buffer/512 rounded128 allocation/seed42/temp0/compileFalse/outer8 inner1 are preserved; every terminal token still gets its original extra forward. NPU replaces only actual hardware backend and storage of the unchanged attention predicate, using CANN BF16 output/F32 LSE then original separate sink scaling; original Triton gate uses F32 internal arithmetic with one BF16 store. No new online A100 run or historical engine-byte claim. No download/stub/borrowed weights/data/fonts/tokenizer or scoring change.'
 proto={k:copy.deepcopy(ready[k]) for k in ['model_id','source_tensor_count','own_weight_files','own_assets_SHA256','own_source_dtype_probe','own_source_dtype_probe_SHA256','own_frontend200','own_frontend200_SHA256','dataset_SHA256']}
 proto.update(weight_path=ready['own_weight_path'],dataset_path=ready['dataset'],workbook_D_reference_authorized=True,own_original_plain_BF16_512_outer8_inner1_protocol_preserved=True,reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=d,local=local['A100_D'],preserved_original=historical['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in [pre,ctl,H/'independent_all200_input_pixels_review.json',preserved,current,R/'configs/FalconOCRForCausalLM/tiiuae--Falcon-OCR.yaml',Path(ready['own_source_dtype_probe'])]},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 pp=R/'configs/protocols/falconocr_own_original_BF16_512_native_v1_20261116.json';create(pp,proto)
 cfg=js(R/'configs/GlmOcrForConditionalGeneration/zai-org--GLM-OCR-own-original-BF16-2048-native-v2-CER-primary-approved-first200-20261116.yaml');config=R/'configs/FalconOCRForCausalLM/tiiuae--Falcon-OCR-own-original-BF16-512-native-v1-20261116.yaml'
 cfg.update(name=config.stem,description=boundary);cfg['data'].update(dataset_path=ready['dataset'],download=False,filter_empty=False,apply_template=False);cfg['algorithm']=dict(name='greedy_inference',batch_size=8,kwargs=dict(extract_answer=True,ocr_extract=True));cfg['metrics']=dict(override=['CER']);cfg['is_ocr']=True
 cfg['model']=dict(name=MID,adapter='universal_eval/model/adapters/falconocr_own_original_bf16_native_512_20261116.py:FalconOCROwnOriginal512',model_path=ready['own_weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]))
 cfg['baseline'].update(score=float(d),primary_metric='CER');validate_config(cfg,MID);create(config,cfg)
 spec=dict(approved=False,review_revision='FalconOCR_own_original_BF16_512_native_v1_20261116',config=str(config.relative_to(R)),verifier='scripts/verify_falconocr_own_original_bf16_512_20261116.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,MID)
 scope(spec,[3],False);mod=importlib.import_module('universal_eval.model.adapters.falconocr_own_original_bf16_native_512_20261116');assert not inspect_abstract(mod.FalconOCROwnOriginal512)
 parent=js(B/'OSAtlas4B_own_original_mcscore_BF16_approved_fresh3_fixed200_v7_final_rope_verifier_binding_real_FLOCK_20261116/job.json')
 job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});create(J/'job.json',job)
 create(H/'readiness.json',dict(passed=True,ready_for_original_runner_true3=True,model_id=MID,source_tensor_count=115,standalone_YAML=str(config),job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=proto['current_A100_D_values'],all200_original_pixels_independently_checked=True,no_NPU_model_inference_yet=True,historical_source_boundary=boundary));print(json.dumps(js(H/'readiness.json'),ensure_ascii=False))
def inspect_abstract(cls):
 import inspect
 return inspect.isabstract(cls)
if __name__=='__main__':main()

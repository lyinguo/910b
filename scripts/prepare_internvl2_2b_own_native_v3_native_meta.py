LOCAL={'D': '0.8350', 'book_SHA256': '3f25bdf0c0da51840b2ef604e7723616bab838729ee6a46b6bfe7020bcbe230e', 'row': 481}
"""Prepare own original InternVL2-2B native NPU3, standalone YAML and guards."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,zipfile,xml.etree.ElementTree as E,importlib,yaml
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='OpenGVLab/InternVL2-2B';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/InternVLChatModel')/MID
F=B/'InternVL2_2B_own_frontend200_source_v2_original_image_reencode_1535_20261115';H=B/'InternVL2_2B_own_native_readiness_v3_1535_20261115';J=B/'InternVL2_2B_own_native_true3_v3_real_FLOCK_1535_20261115'
MODEL='universal_eval/model/adapters/internvl2_2b_own_native_bf16_v3_native_meta.py';VERIFIER='scripts/verify_internvl2_2b_own_native_bf16_v3_native_meta.py';SELF='scripts/prepare_internvl2_2b_own_native_v3_native_meta.py'
CFG='configs/InternVLChatModel/OpenGVLab--InternVL2-2B-own-native-BF16-2048-v3-native-meta-20261115.yaml';PROTO='configs/protocols/internvl2_2b_own_native_bf16_v3_native_meta.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import batch_910b as batch
 from batch_910b_support import workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 H.mkdir(exist_ok=False);review=js(F/'review.json');assert js(F/'actual_completion.json')['actual_returncode']==0 and review['passed'] and review['model_id']==MID and review['samples']==200 and review['source_tensors']==517
 assert sha(review['frontend'])==review['frontend_SHA256']
 orig=R/'configs/InternVLChatModel/OpenGVLab--InternVL2-2B.yaml';saved=yaml.safe_load(orig.read_bytes());assert saved['model']['name']==MID and saved['model']['kwargs']['max_tokens']==2048 and saved['algorithm']['batch_size']==10
 paths=[D/'state.json',D/'registry.json',D/'PAUSE',D/'POLICY.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];protected={str(p):sha(p) for p in paths};assert protected[str(paths[-1])]==LOCAL['book_SHA256']
 rows,sheet,_=workbook_rows(paths[-1]);row,=[r for r in rows if r['model_id']==MID];assert row['score'] is None and row['task']=='image-text-to-text'
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(paths[-1]) as z:currentD=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert currentD==LOCAL['D'];binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');original=Path(binding['local_A100_reference_snapshot']);assert sha(original)=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba';originalD=js(original)['models'][MID]['A100_D']
 decision=D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json';assert sha(decision)=='22cbb4581a69875f643902001babfeffa694a2275e0f894a447866e356bccf3d'
 code=[MODEL,VERIFIER,SELF,'scripts/prepare_internvl2_2b_own_frontend200_v2_original_image_reencode.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/utils/readonly_npz_indexed.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','run.py','universal_eval/orchestrator/pipeline.py','universal_eval/model/internvl35_flash4b_own_native_0123_20261113/own_original_timm_drop_path_0123_20261113.py']+[str(p.relative_to(R)) for p in (R/'universal_eval/model/internvl2_own_native_v2_meta_cpu_20261115').glob('*.py')]
 meta_review=B/'InternVL2_2B_native_meta_no_accelerate_scope_v3_1535_20261115/review.json';assert js(meta_review)['passed'] and js(meta_review)['parameters']==517
 static={str(p):sha(p) for p in [F/'review.json',F/'frontend200.jsonl',F/'source_headers.json',meta_review,B/'InternVL2_2B_own_frontend200_launch_1535_20261115/own_source_transplant.json',orig,decision,original]}
 p=dict(model_id=MID,weight_path=str(W),own_assets_SHA256=review['own_assets_SHA256'],own_weight_files=review['own_weight_files'],own_frontend200=review['frontend'],dataset_path=review['dataset'],dataset_SHA256=review['dataset_SHA256'],implementation_source_SHA256={n:sha(R/n) for n in code},static_evidence_SHA256=static,reference_workbook=str(paths[-1]),reference_workbook_SHA256=LOCAL['book_SHA256'],current_A100_D_values=dict(server=currentD,local=LOCAL['D'],preserved_original=originalD),workbook_D_reference_authorized=True,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3,historical_source_boundary='Saved own vLLM eager/Greedy2048/outer10, own tokenizer and current original vLLM processor; NPU uses own native InternVision24+InternLM2 eager BF16 and exact original dynamic factor2 RoPE. Historical A100 engine/tensor bytes not captured. Workbook D is the user-confirmed comparison reference.')
 put(R/PROTO,p)
 cfg=dict(name=Path(CFG).stem,description=p['historical_source_boundary'],task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':InternVL2OwnNative',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='own_native_greedy_explicit_MC',adapter='universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py:E4BActualOriginalHighTenMC',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Strict_MC_Accuracy'],adapters=['universal_eval/algorithm/tinyllava_strict_mc.py:StrictMCAccuracy'],include_failed_predictions=True),baseline=dict(score=float(currentD),threshold=.05,primary_metric='Strict_MC_Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));put(R/CFG,cfg)
 parent=B/'MLXE4BitBF16_own_chat_native_approved_fresh3_fixed200_v1_real_FLOCK_1210_20261115';job=copy.deepcopy(js(parent/'job.json'));spec=job['spec'];spec.update(approved=False,review_revision='InternVL2_2B_own_native_BF16_full517_workbook_D_20261115',config=CFG,verifier=VERIFIER,source_sha256={n:sha(R/n) for n in code+[CFG,PROTO]})
 for k in ['alignment_approval_review','alignment_approval_review_sha256','protocol_file','protocol_sha256']:spec.pop(k,None)
 job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,write_workbook=False,created_at=datetime.now(timezone.utc).isoformat());J.mkdir(exist_ok=False);put(J/'job.json',job);scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID];c3,e3=batch.config_for(spec,entry,3);c200,e200=batch.config_for(spec,entry,200);assert e3['sample_range']=='0:2' and e200['sample_range']=='0:199'
 module=importlib.import_module('universal_eval.model.internvl2_own_native_v2_meta_cpu_20261115.modeling_internvl_chat');assert hasattr(module,'InternVLChatModel')
 assert protected=={n:sha(n) for n in protected}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),true3_job=str(J),job_SHA256=sha(J/'job.json'),source_SHA256=spec['source_sha256'],ready_for_original_runner_true3=True,own_source_tensors=517,no_new_NPU_claimed=True,no_G_or_canonical_write=True);put(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

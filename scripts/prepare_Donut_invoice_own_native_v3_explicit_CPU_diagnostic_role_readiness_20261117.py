"""Own input/source proofs -> independent native YAML and unapproved true3 job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,ast,copy,shutil,zipfile,xml.etree.ElementTree as ET,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='katanaml-org/invoices-donut-model-v1';H=B/'DonutInvoice_own_original_FP16_poscap2048_native_v3_explicit_CPU_diagnostic_role_readiness_20261117';J=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_explicit_CPU_diagnostic_role_real_FLOCK_20261117'
js=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 from universal_eval.model.adapters.donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117 import DonutInvoiceOwnFP16MetricBindingsV2 as DonutInvoiceOwnFP16
 assert not inspect.isabstract(DonutInvoiceOwnFP16);H.mkdir(exist_ok=False);assert not J.exists()
 old=B/'DonutInvoices_own_original_auto_FP16_caption200_source_META_v3_reference_digest_defined_20261117';done=js(old/'actual_completion.json');review=js(old/'review.json');assert done['actual_returncode']==0 and done['own_inputs200_completed']==200 and review['passed'] and review['samples']==200
 front=old/'frontend200.jsonl';assert sha(front)==review['frontend200_SHA256']=='446d2131c4fca565d95b3c6e7967035c625a1837cf9761c64db13a56729dc26a'
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
 rows,sheet,_=workbook_rows(book);row,=[x for x in rows if x['model_id']==MID];assert row['task']=='image-to-text' and row['score'] is None and js(D/'state.json')['models'][MID]['status']=='needs_adaptation'
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:value=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert Decimal(value)==Decimal('.0039')
 local=js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D'];original=js(binding['local_A100_reference_snapshot'])['models'][MID]['A100_D']
 pixel=R/'universal_eval/model/reference_sources/donut_invoice_own_input_reference.py';tree=ast.parse(pixel.read_bytes());names=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)];assert len(names)==1
 sources=['universal_eval/model/adapters/donut_invoice_own_native_fp16_20261117.py','universal_eval/model/reference_sources/donut_invoice_caption_metrics_reference.py','universal_eval/model/reference_sources/donut_invoice_own_input_reference.py','scripts/verify_donut_invoice_own_native_fp16_v4_explicit_CPU_diagnostic_role_full_linear_KV_20261117.py','scripts/prepare_Donut_invoice_own_native_v3_explicit_CPU_diagnostic_role_readiness_20261117.py','universal_eval/metric/rouge_910b_porter.py','run.py','universal_eval/orchestrator/pipeline.py']
 sources.extend(['universal_eval/model/adapters/donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117.py','universal_eval/metric/rouge_reference_910b.py','universal_eval/metric/text_gen.py']);source_map={n:sha(R/n) for n in sources};source_map.update(review['original_saved_source_SHA256'])
 static={str(old/n):sha(old/n) for n in ['review.json','actual_completion.json']};static[str(book)]=sha(book)
 original_cfg=R/'configs/VisionEncoderDecoderModel/katanaml-org--invoices-donut-model-v1.yaml';static[str(original_cfg)]=sha(original_cfg)
 decision=D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json';assert sha(decision)=='22cbb4581a69875f643902001babfeffa694a2275e0f894a447866e356bccf3d';static[str(decision)]=sha(decision)
 p=dict(model_id=MID,weight_path=str(Path(review['own_weight']['file']).parent),weight_SHA256=review['own_weight']['SHA256'],own_asset_SHA256={n:v['SHA256'] for n,v in review['own_assets'].items()},implementation_source_SHA256=source_map,static_evidence_SHA256=static,own_frontend200=str(front),own_frontend200_SHA256=sha(front),own_pixel_reference_module='universal_eval.model.reference_sources.donut_invoice_own_input_reference',own_pixel_reference_function=names[0],own_image_processor=review['own_image_processor'],original_A100_reference_samples=8,current_A100_metric='ROUGE-L',current_A100_D_values=dict(server=value,local=str(local),preserved_original=str(original)),original_A100_selected_run=review['original_A100_selected_run'],historical_full200_A100_engine_or_tensor_capture_claimed=False,source_dtype='F32_and_original_I64_buffers',execution_dtype='FP16',original_auto_to_accelerator_FP16_proved=True,max_new_tokens=2048,original_own_position_cap=1536,original_prompt_prefix_raw_slicing_preserved=True,no_extra_OCR_logits_constraint=True,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,preflight_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3,independent_FP16_normalized_RMSE_guard=.005,independent_FP16_max_abs_guard=.1,final_original_same_range_qualification_pending_due_A1008_vs_fixed200=True)
 ownreview=B/'DonutInvoice_existing453_full_linear_KV_CPU_vision_diagnostic_only_v3_20261117/review.json'
 assert sha(ownreview)=='5c607da03a06f259f98fbb0fb407a12e2d97716b133628f9ddd35d4403cd7d6e'
 nr=js(ownreview);assert nr['numeric_review_passed'] and nr['diagnostic_only'] and not nr['passed'] and nr['actual_forward_count']==453
 policy=B/'Nougat_actual_A100_FP16_preflight_2116_20261103_20261103_212757_158611/native_adapter_v1/full_numerics_disclosed_v2/numerical_review_with_exact_sources.json'
 assert sha(policy)=='8226f4e2d551080937f726defb43457ba7068a8e129f42a5e53cc3ed53da05c8'
 for f in [ownreview,policy]:static[str(f)]=sha(f)
 p['CPU_vision_comparison_role']='diagnostic_only_no_model_math_change_no_CPU_score'
 p['same_operand_linear_guards_unchanged']={'normalized_RMSE':.005,'max_abs':.1}
 p['own_existing453_no_acceptance_numeric_review']=str(ownreview)
 p['prior_FP16_CPU_validation_role_policy_metadata_only']=str(policy)
 p['old_failed_runner_and_report_not_accepted_or_rewritten']=True
 p['no_other_model_checkpoint_or_array_used_as_numeric_reference']=True
 p['evidence_job']=str(J/'job.json')
 pp=R/'configs/protocols/donut_invoice_own_original_auto_FP16_poscap_native_v3_explicit_CPU_diagnostic_role_20261117.json';create(pp,p)
 cf=R/'configs/VisionEncoderDecoderModel/katanaml-org--invoices-donut-model-v1-own-original-auto-FP16-poscap2048-native-v3-explicit-CPU-diagnostic-role-20261117.yaml'
 cfg=dict(name=cf.stem,description='Own original auto->FP16 Donut; dataset prompt decoder prefix and native1536 position cap of requested2048. Complete head/KV/input/source evidence. Original A100 ROUGE-L only8 samples; no historical full200 capture/qualification claim.',task_type='image-to-text',is_ocr=False,sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=str(R/'workspace/datasets/image-to-text-short/eval.jsonl')),model=dict(name=MID,adapter='universal_eval/model/adapters/donut_invoice_own_native_fp16_v2_offline_metric_bindings_20261117.py:DonutInvoiceOwnFP16MetricBindingsV2',model_path=p['weight_path'],download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),npu_devices=[4])),algorithm=dict(name='greedy_inference',batch_size=1,kwargs=dict(extract_answer=False,truncate_first_sentence=False)),metrics=dict(override=['ROUGE-L','BLEU-4'],include_failed_predictions=True),baseline=dict(score=float(value),threshold=.05),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(cf,cfg)
 template=B/'CoderGemma4E4B_own_original_BF16_true3_v1_saved_HF_F32_real_FLOCK_20261116/job.json';job=copy.deepcopy(js(template));job['model_id']=MID;job['stages']=[3];job['created_at']=datetime.now(timezone.utc).isoformat();job['protected_canonical_hashes']={n:sha(n) for n in job['protected_canonical_hashes']}
 source_map.update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)})
 job['spec']=dict(approved=False,config=str(cf.relative_to(R)),verifier='scripts/verify_donut_invoice_own_native_fp16_v4_explicit_CPU_diagnostic_role_full_linear_KV_20261117.py',npu_devices=[4],source_sha256=source_map,smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 assert all(isinstance(job['spec'][key],int) and job['spec'][key]>0 for key in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds'])
 batch.check_sources(job['spec']);scope(job['spec'],[3],False);entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,c=batch.config_for(job['spec'],entry,count);validate_config(c,MID)
 assert sha(cfg['data']['dataset_path'])==review['dataset_SHA256']=='ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b';assert shutil.disk_usage(D).free>=12*1024**3
 for n,h in job['spec']['source_sha256'].items():
  assert sha(R/n)==h;f=H/'source_original_bytes'/(n+'.txt');f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/n).read_bytes())
 create(J/'job.json',job);ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,true3_job=str(J),job_SHA256=sha(J/'job.json'),standalone_YAML=str(cf),standalone_YAML_SHA256=sha(cf),validate_check_sources_config3_and200_and_concrete_import_passed=True,own_input200_control_not_repeated=True,all484_own_source_cast_to_actual_NPU_pending_true_load=True,real_NPU_not_started=True,current_three_D_values=p['current_A100_D_values'],original_A1008_scope_boundary_and_MLU_BLEU_not_hidden=True,old_passed_other_models_not_rerun=True);create(H/'readiness.json',ready);print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':main()

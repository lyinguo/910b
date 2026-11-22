"""Own physical BF16 Gemma4 E4B, original HF input/default template, current three D."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,inspect,json,os,traceback,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='unsloth/gemma-4-E4B-it-MLX-8bit';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration')/MID
FRONT=B/'Gemma4E4BMLX8_own_original_Transformers_BF16_single_SF_frontend200_v1_20261122'
H=B/'Gemma4E4BMLX8_own_original_Transformers_BF16_source_readiness_v1_20261122';J=B/'Gemma4E4BMLX8_own_original_Transformers_BF16_true3_v1_real_FLOCK_20261122'
MODEL='universal_eval/model/adapters/gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v1_20261122.py';NF='scripts/verify_gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v1_20261122.py';SELF='scripts/prepare_gemma4_e4b_MLX8_own_original_Transformers_BF16_true3_v1_20261122.py'
CFG='configs/Gemma4ForConditionalGeneration/unsloth--gemma-4-E4B-it-MLX-8bit-own-original-Transformers-BF16-native-v1-20261122.yaml';PROTO='configs/protocols/gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v1_20261122.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from batch_910b_support import workbook_rows
 assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]
 H.mkdir(exist_ok=False);pre=js(FRONT/'review.json');assert pre['passed'] and pre['samples']==200 and pre['own_checkpoint_tensor_count']==2130 and pre['own_original_execution_dtype']=='BF16' and pre['actual_source_is_BF16_not_native_8bit'] and pre['original_pixel_input_dtype']=='torch.float32'
 assert pre['all200_independent_F32_patch_pixel_position_replays_equal'] and pre['all200_independent_Rust_token_expansions_equal'] and pre['original_template_default_false_preserved_without_any_override'] and pre['original_saved_filter_empty_fixed200_equal']
 assert sha(FRONT/'frontend200.jsonl')==pre['frontend_jsonl_SHA256'] and js(FRONT/'actual_completion.json')['actual_returncode']==0
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1'
 row,=[r for r in workbook_rows(book)[0] if r['model_id']==MID];assert row['row']==664 and row['score'] is None and row['task']=='image-text-to-text'
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:serverD=E.fromstring(z.read(workbook_rows(book)[1])).find('.//s:c[@r="D664"]',ns).findtext('s:v',namespaces=ns)
 threeD=dict(server=serverD,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']);assert threeD==dict(server='0.8350',local='0.8350',preserved_original='0.835')
 parent=js(B/'LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122/job.json');protected={n:sha(n) for n in parent['protected_canonical_hashes']}
 assert sha(D/'state.json')=='b001dfe5c178d8ba1191ba8ec894fbd8e7ab81b62a50193262e6d8a7fe4cd450' and sha(D/'registry.json')=='42347681b4fe082b1172b6de023a2cc1d1e2f343a54fdf8d656fa2b7fd40614c'
 rels=[MODEL,NF,SELF,'universal_eval/model/gemma4_e4b_MLX8_own_20261122/__init__.py','universal_eval/model/gemma4_e4b_MLX8_own_20261122/observer.py','scripts/inspect_gemma4_e4b_MLX8_own_original_Transformers_template_generation_no_NN_20261122.py','scripts/prepare_gemma4_e4b_MLX8_own_original_Transformers_BF16_single_SF_frontend200_v1_20261122.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','scripts/egm8b_original_independent_cpu_910b.py','scripts/strict_mc_independent_reference_910b.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/model/interface.py','universal_eval/model/adapters/transformers_vlm.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for name in rels:ast.parse((R/name).read_bytes())
 module=importlib.import_module(MODEL[:-3].replace('/','.'));assert not inspect.isabstract(module.Gemma4E4BMLX8OwnBF16)
 saved=R/'configs/Gemma4ForConditionalGeneration/unsloth--gemma-4-E4B-it-MLX-8bit.yaml'
 static=[FRONT/'review.json',FRONT/'frontend200.jsonl',FRONT/'source_tensor_headers.json',FRONT/'source_tensor_files.json',FRONT/'own_source_native_meta_mapping.json',B/'Gemma4E4BMLX8_own_saved_Transformers_template_generation_contract_no_NN_20261122/own_original_interface_contract.json',saved,preserved,current,D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json']
 boundary='Own registered unsloth/gemma-4-E4B-it-MLX-8bit source is actually one physical BF16 safetensors file, 2130 BF16 tensors; no native8bit or converted-engine equality is claimed. Historical converted /mnt2/afs engine bytes were not captured and are unavailable here. Original saved TransformersVLM BF16/Greedy2048/outer10 serial image MMBench branch retained. Own chat template default false is used with NO enable_thinking override; original F32 pixels enter the native model, original native BF16 graph/casts, native131072 context, EOS[1,106,50], full262144 head and42LM/16vision/physical24 plus shared18KV retained. All200 own tokens/pixels and saved original AFS/R data order/images verified independently. No foreign checkpoint/learned tensors/output reuse, online A100 prerequisite, score substitution or math/budget/filter/extraction tuning.'
 protocol=dict(model_id=MID,weight_path=str(W),report_task=row['task'],workbook_B_task=row['task'],saved_original_benchmark_task='image-text-to-text',own_runtime_assets_SHA256=pre['own_runtime_assets_SHA256'],own_weight_files=pre['own_weight_files'],own_source_headers=str(FRONT/'source_tensor_headers.json'),own_source_tensor_files=str(FRONT/'source_tensor_files.json'),original_template_default_false_without_override=True,original_no_enable_thinking_kwarg=True,original_configured_batch_size=10,own_frontend200=str(FRONT/'frontend200.jsonl'),dataset_path=str(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),dataset_SHA256=pre['dataset_SHA256'],implementation_source_SHA256={n:sha(R/n) for n in rels},static_evidence_SHA256={str(p):sha(p) for p in static},current_A100_D_values=threeD,workbook_D_reference_authorized=True,saved_original_configuration_used=True,saved_original_YAML=str(saved),saved_original_YAML_SHA256=sha(saved),preserved_original_snapshot=str(preserved),preserved_original_snapshot_SHA256=sha(preserved),reference_workbook=str(book),reference_workbook_SHA256=sha(book),hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_lossless_space_estimate_bytes=2*2**30,historical_source_boundary=boundary,first200_not_approved=True,actual_source_is_BF16_not_native_8bit=True)
 create(R/PROTO,protocol)
 ALGO='universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py'
 config=dict(name=Path(CFG).stem,description=boundary,task_type=row['task'],sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=protocol['dataset_path']),model=dict(name=MID,adapter=MODEL+':Gemma4E4BMLX8OwnBF16',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_device=4,npu_devices=[4])),algorithm=dict(name='gemma4_e4b_MLX8_own_original_HF_greedy2048',adapter=ALGO+':E4BActualOriginalHighTenMC',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Accuracy'],include_failed_predictions=True),baseline=dict(score=float(serverD),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(R/CFG,config)
 spec=copy.deepcopy(parent['spec']);spec.update(approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**{n:sha(R/n) for n in rels},CFG:sha(R/CFG),PROTO:sha(R/PROTO)},review_revision=boundary)
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own original Transformers BF16 genuine NEW3, no first200 approval')
 scope(spec,[3],False);batch.validate_config(config,MID);batch.check_sources(spec)
 entry=js(D/'state.json')['models'][MID];assert entry['score'] is None and entry['status']=='needs_adaptation'
 for count in [3,200]:_,small=batch.config_for(spec,entry,count);assert small['sample_range']==f'0:{count-1}'
 create(J/'job.json',job);unchanged(protected)
 value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,ready_for_unique_original_runner_true3=True,model_id=MID,job=str(J),job_SHA256=sha(J/'job.json'),YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],three_D_values=threeD,actual_NEW_model_forwards=0,no_G_or_canonical_write=True)
 create(H/'readiness.json',value);create(H/'actual_completion.json',dict(UTC=value['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_failure.json').exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

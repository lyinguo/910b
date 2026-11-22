"""Bind own completed inputs/constructor and native full NF; source-only preparation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,inspect,json,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning'
FRONT=B/'LlavaMore_own_original_FP16_discriminative_letter_pad_frontend200_meta_no_NN_v1_20261122';CPU=B/'LlavaMore_own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122'
H=B/'LlavaMore_own_original_FP16_letter_source_readiness_v1_20261122';J=B/'LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122'
MODEL='universal_eval/model/adapters/llava_more_own_original_FP16_discriminative_letter_native_v1_20261122.py';GRAPH='universal_eval/model/llava_more_own_original_saved_Transformers_FP16_graph_20261122.py';ALGO='universal_eval/algorithm/llava_more_own_original_FP16_discriminative_letter_MC_20261122.py';NF='scripts/verify_LlavaMore_own_original_FP16_discriminative_letter_complete_values_v1_20261122.py'
SELF='scripts/prepare_LlavaMore_own_original_FP16_letter_true3_source_ready_v1_20261122.py';CONTROLLER='scripts/continue_LlavaMore_own_original_letter_after_Xtuner_actual_release_true3_capacity_fresh3_fixed200_serial_v1_20261122.py';QUEUE='configs/queues/LlavaMore_own_original_letter_after_Xtuner_actual_release_serial_v1_20261122.json'
CFG='configs/LlavaLlamaForCausalLM/aimagelab--LLaVA_MORE-llama_3_1-8B-finetuning-own-original-FP16-letter-pad-native-v1-20261122.yaml';PROTO='configs/protocols/llava_more_own_original_FP16_letter_pad_native_v1_20261122.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from batch_910b_support import workbook_rows
 from universal_eval.model.loader import ModelLoader
 from universal_eval.model.interface import IModel
 H.mkdir(exist_ok=False);ready=js(FRONT/'readiness.json');ctor=js(CPU/'actual_original_loader_resize_remap_review.json')
 assert js(FRONT/'actual_completion.json')['actual_returncode']==0 and ready['passed'] and ready['model_id']==MID and ready['all4329_order_and_all200_own_images_tokens_pixels_verified'] and ready['own_frontend200_SHA256']==sha(ready['own_frontend200'])=='0a02f42ae2d23459e6aa47c801fa9327ca94cb570fa6172caed118fd228d749c'
 assert ready['source_dtype_counts']=={'F16':686} and ready['original_probability_not_generation_protocol_preserved'] and ready['original_raw_llama31_scaling_not_in_saved_legacy_graph']
 assert js(CPU/'actual_completion.json')['actual_returncode']==0 and ctor['passed'] and ctor['model_id']==MID and ctor['no_learned_CPU_or_NPU_forward'] and ctor['original_resize_before_manual_remap_order_preserved'] and ctor['native_head_and_embedding_rows_after_original_resize']==128257 and ctor['original_own_first128256_source_rows_bitwise_exact_after_remap'] and ctor['own_actual_source_dtype_counts']=={'torch.float16':686}
 assert sha(ctor['original_extra_head_and_image_embedding_row_values'])==ctor['original_extra_head_and_image_embedding_row_values_SHA256']=='5aa9faee75f47f692dfc7c7a107256f5652f6d8e7868bfac162d4b48ed6ec9e8'
 assert sha(ctor['own_full686_CPU_source_cast_rows_SHA256'])==ctor['own_full686_CPU_source_cast_rows_SHA256_file_SHA']=='266e0a9ccaa3178e9d5699eaee39abbb36412f2a7bc08ebe6720357f303ed422'
 assert ctor['original_generation_config']['eos_token_id']==[128001,128008,128009] and ctor['original_probability_letter_one_forward_cacheFalse_not_generation']
 storage=B/'complete_standard_parallel_NPZ_own_input3_all_values_storage_diagnostic_only_v1_20261122/complete_storage_diagnostic_review.json';s=js(storage);assert s['passed'] and s['actual_returncode']==0 and s['actual_NEW_model_forwards']==0 and s['unit_review']['tests_run']==7 and s['all_complete_own_three_input_examples_every_bit_and_whole_archive_bytes_equal']
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
 row,=[r for r in workbook_rows(book)[0] if r['model_id']==MID];assert row==dict(row=159,model_id=MID,task='image-text-to-text',score=None)
 parent=js(B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122/job.json');protected={n:sha(n) for n in parent['protected_canonical_hashes']}
 assert sha(D/'state.json')=='47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3' and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0'
 rels=[MODEL,GRAPH,ALGO,NF,SELF,CONTROLLER,QUEUE,'universal_eval/utils/complete_standard_LZMA_NPZ_parallel_values_20261122.py','scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py','universal_eval/model/adapters/llava_more_vlm.py','universal_eval/model/adapters/transformers_vlm.py','configs/LlavaLlamaForCausalLM/aimagelab--LLaVA_MORE-llama_3_1-8B-finetuning.yaml','scripts/prepare_LlavaMore_own_original_FP16_letter_pad_frontend200_meta_no_NN_v1_20261122.py','scripts/inspect_LlavaMore_own_original_FP16_loader_resize_remap_no_forward_v1_20261122.py','scripts/egm8b_original_independent_cpu_910b.py','scripts/strict_mc_independent_reference_910b.py','universal_eval/utils/readonly_npz_indexed.py','universal_eval/model/interface.py','universal_eval/model/loader.py','universal_eval/model/manager.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:
  if n.endswith('.py'):ast.parse((R/n).read_bytes())
 actual=ModelLoader.load_class(MODEL+':OwnLlavaMoreOriginalFP16Letter');assert issubclass(actual,IModel) and not inspect.isabstract(actual) and not actual.__abstractmethods__
 module=importlib.import_module(MODEL[:-3].replace('/','.'));importlib.import_module(ALGO[:-3].replace('/','.'));graph=importlib.import_module(GRAPH[:-3].replace('/','.'))
 from universal_eval.model.adapters.llava_more_vlm import LlavaMoreVLM
 assert module.OwnObservedOriginalLlavaMore._predict_letter is LlavaMoreVLM._predict_letter and module.OwnObservedOriginalLlavaMore.predict is LlavaMoreVLM.predict and module.OwnObservedOriginalLlavaMore._prepare_inputs is LlavaMoreVLM._prepare_inputs
 raw=js(Path(ready['own_weight_path'])/'config.json');meta=graph.build_own_meta(raw);assert len(dict(meta.named_parameters()))==686 and tuple(meta.lm_head.weight.shape)==(128257,4096) and meta.config.text_config.max_position_embeddings==131072 and meta.config.text_config.rope_parameters['rope_type']=='default'
 sources={n:sha(R/n) for n in rels};external={}
 for m in list(sys.modules.values()):
  n=getattr(m,'__file__',None)
  if not n:continue
  p=Path(n).resolve()
  if p.is_file() and not p.is_relative_to(R):external[str(p)]=sha(p)
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';afs=Path('/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl');assert sha(dataset)==sha(afs)==ready['dataset_SHA256']
 static={str(p):sha(p) for p in [dataset,afs,FRONT/'readiness.json',FRONT/'own_original_frontend200.json',Path(ready['own_original_legacy_mapping_meta_review']),CPU/'actual_completion.json',CPU/'actual_original_loader_resize_remap_review.json',Path(ctor['own_full686_CPU_source_cast_rows_SHA256']),Path(ctor['original_extra_head_and_image_embedding_row_values']),CPU/'original_actual_CPU_buffer_layout.json',storage]}
 protocol=copy.deepcopy(ready);protocol.update(implementation_source_SHA256=sources,runtime_and_original_source_sha256={**{str(R/n):h for n,h in sources.items()},**external},static_evidence_SHA256=static,own_original_constructor_review=str(CPU/'actual_original_loader_resize_remap_review.json'),own_original_constructor_review_SHA256=sha(CPU/'actual_original_loader_resize_remap_review.json'),current_A100_D_values=ready['three_D_values'],hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_lossless_space_estimate_bytes=2*2**30,original_factor4_plus6GiB_capacity_guard=True,first200_not_approved=True,original_probability_mode='discriminative_letter',original_use_cache=False,original_full_native_head_last_logits_float_then_all_variant_token_ID_max=True,independent_FP16_atol=.01,independent_FP16_rtol=.01,chosen_before_first_own_NPU_execution=True,source_boundary='Own686F16, saved nonCPUautoFP16. Original inherited legacy conversion omits flat raw rope_scaling. Original resize128257 BEFORE own legacy remap with own constructor-derived complete extra rows/RNG. Original padRGB122,116,104 and system, all200 own input pixels and IDs. Original discriminative_letter per ownR YAML: native fullhead/allpositions one forward use_cacheFalse and .float last logits/ALL spaced-unspaced variantID max. It is originally probability scoring, no new generation-to-probability switch, no head/math/dtype/prompt/data/score tuning. Original2048/outer10 preserved. Bounded CPU standard LZMA member compression is byte-identical to serial on complete values; genuine own3/fullNF still mandatory. Historical A100 engine/tensors uncaptured, not claimed.')
 create(R/PROTO,protocol)
 cfg=dict(name=Path(CFG).stem,description=protocol['source_boundary'],task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=str(dataset)),model=dict(name=MID,adapter=MODEL+':OwnLlavaMoreOriginalFP16Letter',model_path=ready['own_weight_path'],download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_device=4,npu_devices=[4])),algorithm=dict(name='llava_more_own_original_probability_letter',adapter=ALGO+':OwnLlavaMoreOriginalGreedyTail',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnLlavaMoreOriginalAccuracy'],include_failed_predictions=True),baseline=dict(score=.79,threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(R/CFG,cfg)
 spec=dict(approved=False,review_revision=cfg['description'],config=CFG,verifier=NF,npu_devices=[4],smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600,source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec)
 for n in [3,200]:
  path,small=batch.config_for(spec,js(D/'state.json')['models'][MID],n);assert small['sample_range']==f'0:{n-1}'
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own original More single-forward probability genuine NEW3/full NF, model dispatch only after actual previous NN/NF/controller/EX release, no first200 approval')
 create(J/'job.json',job);unchanged(protected)
 value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,ready_for_unique_original_runner_true3_when_all_real_quiet=True,ready_does_not_claim_NPU_execution_or_full_qualification=True,current_model_genuine3_not_started_by_this_preparation=True,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=ready['three_D_values'],NPU_device=4,max_concurrent_models=1,actual_NEW_model_forwards=0,first200_not_approved=True,actual_ModelLoader_interface_and_own_original_scoring_function_identity_passed=True,no_G_or_canonical_write=True)
 create(H/'readiness.json',value);create(H/'actual_completion.json',dict(UTC=value['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_failure.json').exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

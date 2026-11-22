"""Bind completed own inputs/original constructor to independent YAML/full-NF genuine3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,json,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='HancomInSpaceAI/HiLLaVA-llama_3_1-8b';FRONT=B/'HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122';CPU=B/'HancomHiLlava31_own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122';H=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_source_readiness_v3_original_registration_20261122';J=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v3_original_registration_real_FLOCK_20261122'
MODEL='universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v3_original_registration_20261122.py';GRAPH='universal_eval/model/hancom_hillava31_own_saved_Transformers_graph_20261122.py';ALGO='universal_eval/algorithm/hancom_hillava31_own_original_FP16_2048_MC_20261122.py';NF='scripts/verify_hancom_hillava31_own_original_Transformers_FP16_2048_full_values_v2_original_registration_20261122.py';SELF='scripts/prepare_hancom_hillava31_own_original_Transformers_FP16_2048_true3_v3_original_registration_20261122.py';CONTROLLER='scripts/continue_hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v3_original_registration_20261122.py';QUEUE='configs/queues/hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v3_original_registration_20261122.json'
CFG='configs/LlavaLlamaForCausalLM/HancomInSpaceAI--HiLLaVA-llama_3_1-8b-own-original-Transformers-FP16-2048-native-v3-original-registration-20261122.yaml';PROTO='configs/protocols/hancom_hillava31_own_original_Transformers_FP16_2048_native_v3_original_registration_20261122.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 import inspect
 from universal_eval.model.interface import IModel
 from universal_eval.model.loader import ModelLoader
 actual_class=ModelLoader.load_class(MODEL+':OwnHancomHiLlava31OriginalFP16')
 assert issubclass(actual_class,IModel) and not inspect.isabstract(actual_class) and not actual_class.__abstractmethods__
 H.mkdir(exist_ok=False)
 old_job=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v2_own_BF16_guard_real_FLOCK_20261122';old_done=js(old_job/'completion.json');old_console=(old_job/'fresh3/console.log').read_text()
 assert old_done['actual_returncode']==1 and "self._tokenizer=AutoTokenizer.from_pretrained" in old_console and "'PreTrainedConfig' object has no attribute 'max_position_embeddings'" in old_console
 assert not any((R/'workspace/diagnostics/model_runs').glob('HancomHiLlava31_own_original_FP16_2048_*/generation_progress.jsonl'))
 old_source=R/'universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v2_own_BF16_guard_20261122.py';assert sha(old_source)=='f67dbaaadad917b612daeadf2826c44d81d23712d312c00e250e6af7295f7d1f'
 old_nf=R/'scripts/verify_hancom_hillava31_own_original_Transformers_FP16_2048_full_values_v1_20261122.py';assert sha(old_nf)=='e60a28b60f00225f3cc28433c7ff60f8ffb92384cd7bfb8b29137a3e3bcc1177'
 original_registration='from universal_eval.model.adapters.transformers_vlm import TransformersVLM as _OwnSavedRegistration\n'
 assert (R/MODEL).read_text().replace(original_registration,'')==old_source.read_text() and (R/NF).read_text().replace(original_registration,'')==old_nf.read_text()
 diagnostic=B/'HancomHiLlava31_original_registered_tokenizer_generation_interface_no_NN_v1_20261122';review=js(diagnostic/'review.json')
 assert sha(diagnostic/'review.json')=='b697ff522a04ee17896b1cac7a864ece7ac2edc124f0042730c9a320e86d26c2' and review['passed'] and review['actual_NEW_model_forwards']==0 and js(diagnostic/'actual_completion.json')['actual_returncode']==0
 from transformers import AutoTokenizer,GenerationConfig
 own_path=Path(js(FRONT/'readiness.json')['own_weight_path']);tokenizer=AutoTokenizer.from_pretrained(own_path,local_files_only=True);generation=GenerationConfig.from_pretrained(own_path,local_files_only=True)
 assert len(tokenizer)==128256 and generation.eos_token_id==[128001,128008,128009] and tokenizer.chat_template==js(own_path/'tokenizer_config.json')['chat_template']
 create(H/'exact_zero_forward_original_registration_recovery.json',dict(passed=True,old_job=str(old_job),old_completion_SHA256=sha(old_job/'completion.json'),old_console_SHA256=sha(old_job/'fresh3/console.log'),old_model_SHA256=sha(old_source),old_verifier_SHA256=sha(old_nf),only_original_saved_registration_import_in_model_and_verifier=True,all_model_math_observers_FP16_cast_budget_EOS_thresholds_unchanged=True,diagnostic_review_SHA256=sha(diagnostic/'review.json'),original_v1_and_v2_failure_evidence_retained=True,own_existing_frontend200_and_constructor_reused_without_replay=True))
 ready=js(FRONT/'readiness.json');ctor=js(CPU/'actual_original_loader_resize_remap_review.json')
 assert ready['passed'] and ready['model_id']==MID and ready['all4329_order_and_all200_own_images_tokens_pixels_verified'] and ready['own_frontend200_SHA256']=='9745e8b0845d5ef250c64d3e2491160ef1a9bc276bd82507e3cd585c9d02a5be' and sha(ready['own_frontend200'])==ready['own_frontend200_SHA256']
 assert js(CPU/'actual_completion.json')['actual_returncode']==0 and ctor['passed'] and ctor['no_learned_CPU_or_NPU_forward'] and ctor['original_resize_before_manual_remap_order_preserved'] and ctor['native_head_and_embedding_rows_after_original_resize']==128257 and ctor['original_own_first128256_source_rows_bitwise_exact_after_remap']
 assert sha(ctor['original_extra_head_and_image_embedding_row_values'])==ctor['original_extra_head_and_image_embedding_row_values_SHA256']
 assert sha(ctor['own_full686_CPU_source_cast_rows_SHA256'])==ctor['own_full686_CPU_source_cast_rows_SHA256_file_SHA']
 assert ctor['original_generation_config']['eos_token_id']==[128001,128008,128009] and ctor['actual_CPU_parameter_dtypes']==['torch.float16']
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
 from batch_910b_support import workbook_rows
 row,=[r for r in workbook_rows(book)[0] if r['model_id']==MID];assert row==dict(row=338,model_id=MID,task='image-text-to-text',score=None)
 parent=js(B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122/job.json');protected={n:sha(n) for n in parent['protected_canonical_hashes']}
 assert sha(D/'state.json')=='47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3' and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0'
 rels=[MODEL,GRAPH,ALGO,NF,SELF,CONTROLLER,QUEUE,'universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v2_own_BF16_guard_20261122.py','scripts/verify_hancom_hillava31_own_original_Transformers_FP16_2048_full_values_v1_20261122.py','scripts/diagnose_HancomHiLlava31_original_registered_tokenizer_generation_interface_no_NN_v1_20261122.py','universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v1_20261122.py','configs/LlavaLlamaForCausalLM/HancomInSpaceAI--HiLLaVA-llama_3_1-8b.yaml','universal_eval/model/adapters/transformers_vlm.py','scripts/prepare_HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py','scripts/inspect_HancomHiLlava31_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py','scripts/egm8b_original_independent_cpu_910b.py','scripts/strict_mc_independent_reference_910b.py','universal_eval/utils/complete_standard_LZMA_NPZ_parallel_values_20261122.py','scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py','universal_eval/utils/readonly_npz_indexed.py','universal_eval/model/interface.py','universal_eval/model/loader.py','universal_eval/model/manager.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:
  if n.endswith('.py'):ast.parse((R/n).read_bytes())
 module=importlib.import_module(MODEL[:-3].replace('/','.'));importlib.import_module(ALGO[:-3].replace('/','.'));graph=importlib.import_module(GRAPH[:-3].replace('/','.'))
 raw=js(Path(ready['own_weight_path'])/'config.json');meta=graph.build_own_meta(raw);assert len(dict(meta.named_parameters()))==686 and tuple(meta.lm_head.weight.shape)==(128257,4096) and meta.config.text_config.max_position_embeddings==131072
 import torch,inspect
 from transformers.cache_utils import DynamicLayer
 key=torch.arange(2*8*3*128,dtype=torch.float16).reshape(2,8,3,128);value=key+1
 layer=module.OwnFourDimensionalDynamicLayer();layer.lazy_initialization(key,value);assert layer.keys.shape==(2,8,0,128) and layer.values.shape==(2,8,0,128)
 assert torch.equal(torch.cat([layer.keys,key],dim=-2),key) and torch.equal(torch.cat([layer.values,value],dim=-2),value)
 sources={n:sha(R/n) for n in rels};external={}
 for m in list(sys.modules.values()):
  n=getattr(m,'__file__',None)
  if not n:continue
  p=Path(n).resolve()
  if p.is_file() and not p.is_relative_to(R):external[str(p)]=sha(p)
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';afs=Path('/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl');assert sha(dataset)==sha(afs)==ready['dataset_SHA256']
 static={str(p):sha(p) for p in [H/'exact_zero_forward_original_registration_recovery.json',diagnostic/'review.json',diagnostic/'actual_completion.json',B/'HancomHiLlava31_own_original_Transformers_FP16_2048_source_readiness_v2_own_BF16_guard_20261122/exact_zero_forward_source_dtype_guard_recovery.json',old_job/'completion.json',old_job/'fresh3/console.log',dataset,afs,FRONT/'readiness.json',FRONT/'own_original_frontend200.json',Path(ready['own_original_legacy_mapping_meta_review']),CPU/'actual_completion.json',CPU/'actual_original_loader_resize_remap_review.json',Path(ctor['own_full686_CPU_source_cast_rows_SHA256']),Path(ctor['original_extra_head_and_image_embedding_row_values']),CPU/'original_actual_CPU_buffer_layout.json']}
 protocol=copy.deepcopy(ready);protocol.update(implementation_source_SHA256=sources,runtime_and_original_source_sha256={**{str(R/n):h for n,h in sources.items()},**external},static_evidence_SHA256=static,own_original_constructor_review=str(CPU/'actual_original_loader_resize_remap_review.json'),own_original_constructor_review_SHA256=sha(CPU/'actual_original_loader_resize_remap_review.json'),current_A100_D_values=ready['three_D_values'],hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_lossless_space_estimate_bytes=2*2**30,original_factor4_plus6GiB_capacity_guard=True,first200_not_approved=True,source_boundary='Own686 BF16 physical sources cast to saved nonCPUauto requested FP16; original saved constructor expands to128257 BEFORE legacy remap, same-model constructor-derived extra rows full values/provenance retained, original FP32 analytic RoPE buffers. No foreign checkpoint/head/vision, no historical A100 runtime/tensor claim. Own2048/outer10/Greedy/EOS[128001,128008,128009], all200 own input IDs/pixels, full source and original source paths preserved. Actual NPU source/dtype/math proven only by forthcoming own genuine3.')
 create(R/PROTO,protocol)
 config=dict(name=Path(CFG).stem,description=protocol['source_boundary'],task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=str(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl')),model=dict(name=MID,adapter=MODEL+':OwnHancomHiLlava31OriginalFP16',model_path=ready['own_weight_path'],download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_device=4,npu_devices=[4])),algorithm=dict(name='hancom_hillava31_own_original_FP16_greedy2048',adapter=ALGO+':OwnHancomHiLlava31Greedy',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnHancomHiLlava31Accuracy'],include_failed_predictions=True),baseline=dict(score=.715,threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(R/CFG,config)
 spec=copy.deepcopy(parent['spec']);spec.update(approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)},review_revision=config['description'])
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own original native FP16 genuine NEW3 complete numeric verification; no first200 approval')
 scope(spec,[3],False);batch.validate_config(config,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID]
 for n in [3,200]:_,small=batch.config_for(spec,entry,n);assert small['sample_range']==f'0:{n-1}'
 create(J/'job.json',job);unchanged(protected);v=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,ready_for_unique_original_runner_true3=True,ready_not_NPU_execution=True,own_original_system_date_26Jul2024_and_BF16_to_original_FP16_preserved=True,model_id=MID,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=ready['three_D_values'],NPU_device=4,max_concurrent_models=1,actual_NEW_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True)
 create(H/'readiness.json',v);create(H/'actual_completion.json',dict(UTC=v['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(v,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_failure.json').exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

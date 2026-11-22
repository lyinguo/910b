"""Own complete source/dtype/three-D proof and one directly runnable private YAML."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,inspect,json,os,traceback,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.1V-9B-Base';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
FRONT=B/'GLM41Base_own_original_option_logprob_native_frontend200_meta_no_NN_v1_20261122';H=B/'GLM41Base_own_original_option_probability_BF16_native_source_readiness_v1_20261122';J=B/'GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122'
MODEL='universal_eval/model/adapters/glm41base_own_original_option_logprob_BF16_native_v1_20261122.py';INPUT='universal_eval/model/adapters/glm41base_own_original_option_inputs_v1_20261122.py';ALGO='universal_eval/algorithm/glm41base_own_original_option_logprob_MC_v1_20261122.py';NF='scripts/verify_glm41base_own_original_option_logprob_BF16_complete_values_v1_20261122.py';SELF='scripts/prepare_GLM41Base_own_original_option_probability_true3_v1_20261122.py';CONTROLLER='scripts/continue_GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.py';QUEUE='configs/queues/GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122.json'
CFG='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Base-own-original-option-logprob-BF16-native-v1-20261122.yaml';PROTO='configs/protocols/glm41base_own_original_option_logprob_BF16_native_v1_20261122.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 os.chdir(R);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');H.mkdir(exist_ok=False)
 import torch,yaml
 from safetensors import safe_open
 from transformers import Glm4vConfig,Glm4vForConditionalGeneration
 from batch_910b_support import workbook_rows
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from universal_eval.model.loader import ModelLoader
 from universal_eval.model.interface import IModel
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4);ready=js(FRONT/'review.json');assert js(FRONT/'actual_completion.json')['actual_returncode']==0 and ready['passed'] and ready['model_id']==MID and ready['own_physical_BF16_sources']==704 and ready['native_dtype']=='torch.bfloat16'
 assert ready['all200_own_native_and_independent_full_pixels_tokens_and_BF16_values_verified'] and ready['all200_have_original_contiguous_options_no_fallback_triggered'] and ready['original_option_mean_logprob_slices_unchanged'] and ready['original_probability_not_replaced_by_generation'] and ready['original_PIL_direct_RGB_not_other_model_white_background']
 assert ready['own_frontend200_SHA256']=='0d4777bbaaab10a0701bc27296850e9441e07c3284717e52bc30f8f089176228' and sha(ready['own_frontend200'])==ready['own_frontend200_SHA256'] and ready['total_native_original_option_requests']==525 and ready['actual_native_stop_sets']==[[151329,151336,151338,151348]]
 original=Path(ready['own_original_saved_YAML']);assert sha(original)==ready['own_original_saved_YAML_SHA256']=='606ac2677328d9b7dfe02dc0e5ef546ea1d56bce4d9c69485278a2e21fb35860';saved=yaml.safe_load(original.read_bytes());assert saved['model']['kwargs']['max_tokens']==2048 and saved['algorithm']['batch_size']==10
 for n,h in ready['own_assets_SHA256'].items():assert sha(W/n)==h
 assert sha(R/'universal_eval/model/adapters/glm41v_base_discriminative.py')==ready['own_original_adapter_SHA256']=='c95d7286200f93a000f7232cd807ec64e80bb4e9ac5911f06976258ed328184d' and sha(R/'universal_eval/model/adapters/glm41v_base_vllm.py')==ready['own_original_parent_adapter_SHA256']=='c435cdcfa181637ed69181ab1d9aa5ca0aaaca17efb6f0015be74e400f10e86c'
 headers=js(ready['own_source_headers']);assert sha(ready['own_source_headers'])==ready['own_source_headers_SHA256'];manifest={}
 for filename,r in ready['own_weight_files'].items():
  assert sha(W/filename)==r['SHA256'] and (W/filename).stat().st_size==r['bytes']
  with safe_open(str(W/filename),framework='pt',device='cpu') as sf:
   assert set(sf.keys())=={n for n,v in headers.items() if v['source_file']==filename}
   for n in sf.keys():
    t=sf.get_tensor(n);assert t.dtype==torch.bfloat16 and torch.isfinite(t).all() and list(t.shape)==headers[n]['shape'];h=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();manifest[n]=dict(source_key=n,runtime_name=n,source_file=filename,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=h,cast_dtype='torch.bfloat16',cast_SHA256=h)
 assert len(manifest)==704;create(H/'own704_source_manifest.json',manifest)
 front=[json.loads(s) for s in Path(ready['own_frontend200']).read_text().splitlines()];assert [r['index'] for r in front]==list(range(200)) and sum(len(r['options']) for r in front)==525
 algorithm=importlib.import_module(ALGO[:-3].replace('/','.'));assert all(algorithm.OwnGLM41BaseProbabilityInference._add_answer_instruction(r['prompt'])==r['actual_model_prompt']==GreedyInference._add_answer_instruction(r['prompt']) for r in front)
 for r in front:
  assert r['all_independent_own_full_pixels_tokens_and_native_BF16_values_equal'] and 2<=len(r['options'])<=5 and [o['letter'] for o in r['options']]==list('ABCDE'[:len(r['options'])])
 create(H/'own_frontend200_proof_field_binding_no_replay.json',front)
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878';local_receipt=B/'GLM41Base_own_original_option_probability_local185_binding_20261122/fresh_local_workbook_SHA_stat.json';lr=js(local_receipt);assert lr['SHA256']==sha(book) and lr['verified185_unchanged'] and lr['no_write']
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None and row['row']==695
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:current_D=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256'];values=dict(server=current_D,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']);assert all(__import__('decimal').Decimal(str(v))==__import__('decimal').Decimal('.91') for v in values.values())
 parent=js(B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122/job.json');protected={n:sha(n) for n in parent['protected_canonical_hashes']};assert sha(D/'state.json')=='47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3' and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0'
 rels=[MODEL,INPUT,ALGO,NF,SELF,CONTROLLER,QUEUE,'universal_eval/utils/complete_module_values_tree_20261122.py','scripts/tests/test_complete_module_values_tree_20261122.py','scripts/prepare_GLM41Base_own_original_option_logprob_frontend200_native_no_NN_v1_20261122.py','universal_eval/model/adapters/glm41v_base_discriminative.py','universal_eval/model/adapters/glm41v_base_vllm.py','universal_eval/model/adapters/vllm_adapter.py','universal_eval/utils/complete_standard_LZMA_NPZ_parallel_values_20261122.py','scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py','universal_eval/utils/readonly_npz_indexed.py','scripts/strict_mc_independent_reference_910b.py','universal_eval/model/interface.py','universal_eval/model/loader.py','universal_eval/model/manager.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:
  if n.endswith('.py'):ast.parse((R/n).read_bytes())
 actual_class=ModelLoader.load_class(MODEL+':OwnGLM41BaseOriginalProbabilityBF16');assert issubclass(actual_class,IModel) and not inspect.isabstract(actual_class)
 ownconfig=Glm4vConfig(**js(W/'config.json'))
 with torch.device('meta'):meta=Glm4vForConditionalGeneration(ownconfig)
 assert len(dict(meta.named_parameters(remove_duplicate=False)))==704 and all(list(t.shape)==manifest[n]['shape'] for n,t in meta.named_parameters(remove_duplicate=False))
 sources={n:sha(R/n) for n in rels};boundary='Own GLM4.1 Base original discriminative option-text mean prompt logprob, not generation or letter-logit ranking. Own704 BF16/40text/GQA2/24vision/full151552 head. Exact original manual prompt, strip-only-existing instruction suffix, original [-k-2:-2] score slice/mean/tie/order; original option request max_tokens1 temperature0 prompt_logprobs0; saved fallback2048/outer10 unchanged, all fixed200 have options so fallback not triggered. Own PIL direct RGB and native completion inputs all525 option requests independently verified. Native HF SDPA fresh-request full40 physical KV, whole-head logits/all prompt probabilities and every captured original module full values checked against CPU diagnostic references; no CPU grade, no historical A100 engine/tensor claim. Original source/dtype/math/EOS/position/mask/head/scoring unchanged; no borrowed checkpoint, task, tokens, font or output.'
 static=[FRONT/'review.json',FRONT/'actual_completion.json',Path(ready['own_frontend200']),original,preserved,current,local_receipt,FRONT/'own_whole_source_files.json',FRONT/'own_meta_only_parameter_layout.json']
 p=dict(model_id=MID,weight_path=str(W),source_tensor_count=704,own_weight_files=ready['own_weight_files'],own_assets_SHA256=ready['own_assets_SHA256'],own_source_manifest=str(H/'own704_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own704_source_manifest.json'),own_frontend200=str(H/'own_frontend200_proof_field_binding_no_replay.json'),own_frontend200_SHA256=sha(H/'own_frontend200_proof_field_binding_no_replay.json'),dataset_path=str(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),dataset_SHA256=sha(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),original_option_text_mean_logprob_protocol_preserved=True,original_stop_ids=[151329,151336,151338,151348],original_answer_instruction=front[0]['actual_model_prompt'][len(front[0]['prompt']):],workbook_D_reference_authorized=True,current_A100_D_values=values,reference_workbook=str(book),reference_workbook_SHA256=sha(book),implementation_source_SHA256=sources,static_evidence_SHA256={str(f):sha(f) for f in static},historical_source_boundary=boundary,hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_lossless_space_estimate_bytes=2*2**30,original_factor4_plus6GiB_capacity_guard=True,first200_not_approved=True,fixed_BF16_numeric_relative_RMSE=.02,fixed_FP32_logprob_absolute_tolerance=1e-4)
 create(R/PROTO,p);cfg=dict(name=Path(CFG).stem,description=boundary,task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':OwnGLM41BaseOriginalProbabilityBF16',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='glm41base_own_original_option_text_mean_logprob',adapter=ALGO+':OwnGLM41BaseProbabilityInference',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnGLM41BaseAccuracy'],include_failed_predictions=True),baseline=dict(score=float(current_D),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(R/CFG,cfg)
 spec=copy.deepcopy(parent['spec']);spec.update(review_revision=boundary,approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 for n in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(n,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own Base original probability NEW NPU3 complete numerical proof before first200 measurement approval')
 scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID]
 for n in [3,200]:_,c=batch.config_for(spec,entry,n);assert c['sample_range']==f'0:{n-1}'
 for n in [MODEL,INPUT,ALGO,NF]:importlib.import_module(n[:-3].replace('/','.'))
 create(J/'job.json',job);unchanged(protected);result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,own704_source_manifest_SHA256=sha(H/'own704_source_manifest.json'),own_original_200samples_525option_inputs_reused_without_replay=True,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],true3_job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=values,C_unique_current_row=695,ready_for_unique_original_runner_true3=True,actual_NEW_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True);create(H/'readiness.json',result);create(H/'actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_failure.json').exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

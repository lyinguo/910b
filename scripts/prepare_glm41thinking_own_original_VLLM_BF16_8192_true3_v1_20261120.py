"""Own704 source bytes, original native8192 protocol and private genuine NPU3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,json,os,traceback,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.1V-9B-Thinking';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
OLD=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v3_actual_media_wrapper_20261120'
H=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_source_readiness_v1_20261120'
J=B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120'
MODEL='universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_native_v2_actual_media.py'
INPUT='universal_eval/model/adapters/glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend_v2_actual_media.py'
ALGO='universal_eval/algorithm/glm41thinking_own_original_VLLM_BF16_8192_20261120_MC.py'
NF='scripts/verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py'
SELF='scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py'
CFG='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Thinking-own-original-VLLM-BF16-8192-native-v1-20261120.yaml'
PROTO='configs/protocols/glm41thinking_own_original_VLLM_BF16_8192_20261120_native_v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 os.chdir(str(R));os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 import torch,yaml
 from safetensors import safe_open
 from batch_910b_support import workbook_rows
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [r for r in project_processes() if r['PID']!=os.getpid() and r['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]
 H.mkdir(exist_ok=False);torch.set_num_threads(4);ready=js(OLD/'review.json')
 assert ready['passed'] and ready['own_physical_BF16_sources']==704 and ready['meta_map_complete_without_conversion'] and ready['actual_original_no_thinking_override_preserved'] and ready['max_new_tokens']==8192 and ready['native_dtype']=='torch.bfloat16'
 assert ready['original_filter_true_and_false_all4329_prompt_ref_order_and_200_images_identical'] and ready['actual_native_stop_sets']==[[151329,151336,151338,151348]]
 assert sha(ready['own_frontend200'])==ready['own_frontend200_SHA256'] and sha(ready['original_generic_adapter_path'])==ready['original_generic_adapter_SHA256']
 original=Path(ready['own_original_saved_YAML']);assert sha(original)==ready['own_original_saved_YAML_SHA256'];saved=yaml.safe_load(original.read_bytes());assert saved['model']['kwargs']['max_tokens']==8192 and saved['algorithm']['batch_size']==10
 headers=js(ready['own_source_headers']);assert sha(ready['own_source_headers'])==ready['own_source_headers_SHA256'];manifest={}
 for filename,record in ready['own_weight_files'].items():
  assert sha(W/filename)==record['SHA256'] and (W/filename).stat().st_size==record['bytes']
  with safe_open(str(W/filename),framework='pt',device='cpu') as sf:
   assert set(sf.keys())=={n for n,v in headers.items() if v['source_file']==filename}
   for name in sf.keys():
    t=sf.get_tensor(name);assert str(t.dtype)=='torch.bfloat16' and list(t.shape)==headers[name]['shape']
    h=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    manifest[name]=dict(source_key=name,runtime_name=name,source_file=filename,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=h,cast_dtype='torch.bfloat16',cast_SHA256=h)
 assert len(manifest)==704 and set(manifest)==set(headers);create(H/'own704_active_source_manifest.json',manifest);create(H/'own_unused_source_manifest.json',{})
 front=[]
 for r in [json.loads(s) for s in Path(ready['own_frontend200']).read_text().splitlines()]:
  assert r['all_independent_own_pixels_tokens_native_media_and_BF16_full_values_equal'] and r['original_native_template_default_no_thinking_override']
  front.append(dict(index=r['index'],prompt=r['prompt'],actual_model_prompt=r['actual_model_prompt'],query=r['own_chat_template_string'],media_path=r['media_path'],media_file_SHA256=r['image_SHA256'],reference=r['reference'],input_ids=r['own_complete_input_ids'],tensor_metadata=r['all_actual_input_tensor_metadata'],original_native_clean_prompt=r['original_native_clean_prompt'],original_native_message=r['original_native_message'],actual_native_pixel_metadata=r['actual_native_pixel_metadata'],actual_native_pixel_BF16=r['actual_native_pixel_BF16'],independent_full_own_resize_pixels_normalization_patches_and_Rust_tokens=True))
 assert [r['index'] for r in front]==list(range(200)) and len({r['actual_model_prompt'][len(r['prompt']):] for r in front})==1;create(H/'own_frontend200_existing_proof_field_binding_no_replay.json',front)
 parent=js(B/'AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120/job.json');protected={p:sha(p) for p in parent['protected_canonical_hashes']}
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None and row['row']==696
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:current_D=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 values=dict(server=current_D,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']);assert all(v is not None and float(v)>0 for v in values.values())
 rels=[MODEL,INPUT,ALGO,NF,SELF,'scripts/prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py','universal_eval/model/reference_sources/glm41thinking_owned_tokenizer_config_20261120.py','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py','universal_eval/model/adapters/vllm_adapter.py','universal_eval/model/interface.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels}
 boundary='User-confirmed actual A100 workbook D, not an online replay gate. Own GLM-4.1V-9B-Thinking four SF/704 BF16 tensors, own vision24/text40/GQA2x128/full151552 head. Saved original VLLM Greedy8192/outer10 serial1/TP1/.9, original image-placeholder removal and white-background ImageMediaIO/native chat default add_generation_prompt, no explicit thinking override, own unchanged template and EOS151329/151336/151338/151348. Current real installed native InputProcessor/renderer and all200 independent full own pixels/Rust own-config tokens prove all actual model BF16 input bytes equal. Own raw tokenizer.json plus its own nine added_tokens_decoder items preserve all IDs; no borrowed vocabulary. Native HF SDPA execution and full40 physical KV/3D positions/head values audited; historical A100 engine/tensor bytes were not captured. Original AFS/R files, all4329 prompt-ref order and all200 image bytes identical; original filter_empty=True removes zero, runtimeFalse preserves exactly fixed0:199. No package installation, generic decoder stubs, borrowed checkpoint/input/output, precision/budget/cache/stop/math/scoring/thinking tuning.'
 static=[OLD/'review.json',Path(ready['own_frontend200']),original,preserved,current,OLD/'own_whole_source_files.json',OLD/'own_meta_only_parameter_layout.json']
 p=dict(model_id=MID,weight_path=str(W),source_tensor_count=704,own_weight_files=ready['own_weight_files'],own_assets_SHA256=ready['own_assets_SHA256'],own_source_manifest=str(H/'own704_active_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own704_active_source_manifest.json'),own_unused_MTP_source_manifest=str(H/'own_unused_source_manifest.json'),own_unused_MTP_source_manifest_SHA256=sha(H/'own_unused_source_manifest.json'),own_frontend200=str(H/'own_frontend200_existing_proof_field_binding_no_replay.json'),own_frontend200_SHA256=sha(H/'own_frontend200_existing_proof_field_binding_no_replay.json'),dataset_path=ready['actual_target_dataset'],dataset_SHA256=ready['dataset_SHA256'],original_stop_ids=[151329,151336,151338,151348],original_BF16_own_template8192_outer10_inner1_Accuracy_protocol_preserved=True,original_answer_instruction=front[0]['actual_model_prompt'][len(front[0]['prompt']):],original_no_thinking_override_adapter_SHA256=ready['original_generic_adapter_SHA256'],original_native_input_library_source_SHA256=ready['installed_source_SHA256'],workbook_D_reference_authorized=True,current_A100_D_values=values,reference_workbook=str(book),reference_workbook_SHA256=sha(book),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in static},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,precheck_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3)
 create(R/PROTO,p)
 cfg=dict(name=Path(CFG).stem,description=boundary,task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':GLM41ThinkingOwnNativeBF16',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='glm41thinking_original_VLLM8192_greedy_explicit_MC',adapter=ALGO+':OwnGLM41ThinkingGreedy',batch_size=10,kwargs=dict(extract_answer=True,answer_instruction=p['original_answer_instruction'])),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnGLM41ThinkingAccuracy'],include_failed_predictions=True),baseline=dict(score=float(current_D),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
 create(R/CFG,cfg)
 spec=copy.deepcopy(parent['spec']);spec.update(review_revision='Own GLM4.1 Thinking704 BF16 saved VLLM8192/native exact input/full-value private stage',approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own GLM4.1 Thinking original VLLM8192 NEW NPU3 before first200 measurement approval')
 scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);assert v['sample_range']==f'0:{count-1}'
 for m in [MODEL,INPUT,ALGO,NF]:importlib.import_module(m[:-3].replace('/','.'))
 create(J/'job.json',job)
 for n,h in protected.items():assert sha(n)==h
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,own704_source_manifest_SHA256=sha(H/'own704_active_source_manifest.json'),own_complete_input200_reused_without_replay=True,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],true3_job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=values,C_unique_current_row=row['row'],ready_for_unique_original_runner_true3=True,actual_NEW_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True)
 create(H/'readiness.json',result);create(H/'actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

"""Own GLM4.6V source704, already verified own input200, standalone YAML and private true3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,json,os,traceback,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/AutoGLM-Phone-9B-Multilingual';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
OLD=B/'AutoGLMPhone9B_own_original_BF16_2048_frontend200_meta_v1_20261119'
H=B/'AutoGLMPhone9B_own_original_BF16_2048_native_source_readiness_v1_20261119'
J=B/'AutoGLMPhone9B_own_original_BF16_2048_true3_v1_real_FLOCK_20261119'
MODEL='universal_eval/model/adapters/autoglm_phone9b_own_original_BF16_2048_20261119_native.py'
INPUT='universal_eval/model/adapters/autoglm_phone9b_own_original_BF16_2048_20261119_frontend.py'
ALGO='universal_eval/algorithm/autoglm_phone9b_own_original_BF16_2048_20261119_MC.py'
NF='scripts/verify_autoglm_phone9b_own_original_BF16_2048_20261119_native.py'
SELF='scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py'
CFG='configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B-Multilingual-own-original-BF16-2048-native-v1-20261119.yaml'
PROTO='configs/protocols/autoglm_phone9b_own_original_BF16_2048_20261119_native_v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import torch,yaml
 from safetensors import safe_open
 from batch_910b_support import workbook_rows
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [v for v in project_processes() if v['PID']!=__import__('os').getpid() and v['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 ready=js(OLD/'review.json');assert ready['passed'] and ready['own_physical_BF16_sources']==704 and ready['meta_map_complete_without_conversion'] and ready['actual_original_existing_thinking_false_preserved']
 assert sha(ready['own_frontend200'])==ready['own_frontend200_SHA256']
 assert sha(ready['original_generic_adapter_path'])==ready['original_generic_adapter_SHA256']
 original=Path(ready['own_original_saved_YAML']);assert sha(original)==ready['own_original_saved_YAML_SHA256'];saved=yaml.safe_load(original.read_bytes())
 headers=js(ready['own_source_headers']);manifest={}
 for filename,record in ready['own_weight_files'].items():
  assert sha(W/filename)==record['SHA256'] and (W/filename).stat().st_size==record['bytes']
  with safe_open(str(W/filename),framework='pt',device='cpu') as f:
   assert set(f.keys())=={n for n,h in headers.items() if h['source_file']==filename}
   for name in f.keys():
    t=f.get_tensor(name);assert t.dtype==torch.bfloat16 and list(t.shape)==headers[name]['shape']
    h=hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    manifest[name]=dict(source_key=name,source_file=filename,runtime_name=name,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=h,cast_SHA256=h)
 assert len(manifest)==704
 create(H/'own704_active_source_manifest.json',manifest);create(H/'own_unused_source_manifest.json',{})
 front=[]
 for row in [json.loads(s) for s in Path(ready['own_frontend200']).read_text().splitlines()]:
  front.append(dict(index=row['index'],prompt=row['prompt'],actual_model_prompt=row['actual_model_prompt'],query=row['own_chat_template_string'],media_path=row['media_path'],media_file_SHA256=row['image_SHA256'],reference=row['reference'],input_ids=row['own_complete_input_ids'],tensor_metadata=row['all_actual_input_tensor_metadata'],independent_full_own_resize_pixels_normalization_patches_and_Rust_tokens=row['own_all_resize_fused_F32_normalization_patch_layout_and_Rust_tokens_bitwise_equal']))
 assert [r['index'] for r in front]==list(range(200)) and len({r['actual_model_prompt'][len(r['prompt']):] for r in front})==1;create(H/'own_frontend200_existing_proof_field_binding_no_replay.json',front)
 parent=js(B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118/job.json');protected={p:sha(p) for p in parent['protected_canonical_hashes']}
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:current_D=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 values=dict(server=current_D,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']);assert all(v is not None and float(v)>0 for v in values.values()),values
 rels=[MODEL,INPUT,ALGO,NF,SELF,'scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_frontend200.py','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py','universal_eval/model/adapters/transformers_vlm.py','universal_eval/model/interface.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels}
 boundary='User workbook D is the A100 score reference, not an online AA replay gate. Own AutoGLM-Phone-9B-Multilingual704 BF16 tensors/40text GQA2x128/vision24/151552 head; own HF greedy2048 outer10 serial1, original generic-adapter default enable_thinking=False, ignored by its own unchanged chat template (no injected /nothink or empty think) and exact original GreedyInference answer instruction, own generation EOS151329/151336/151338/151348. Original installed HF SDPA/model math/native40-layer dynamic cache and all full raw values retained. Historical A100 engine/tensor bytes were not captured. No generic mock initializer, package install, borrowed checkpoint/input source/output, changed precision/budget/thinking/stop/scoring.'
 p=dict(model_id=MID,weight_path=str(W),source_tensor_count=704,own_weight_files=ready['own_weight_files'],own_assets_SHA256=ready['own_assets_SHA256'],own_source_manifest=str(H/'own704_active_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own704_active_source_manifest.json'),own_unused_MTP_source_manifest=str(H/'own_unused_source_manifest.json'),own_unused_MTP_source_manifest_SHA256=sha(H/'own_unused_source_manifest.json'),own_frontend200=str(H/'own_frontend200_existing_proof_field_binding_no_replay.json'),own_frontend200_SHA256=sha(H/'own_frontend200_existing_proof_field_binding_no_replay.json'),dataset_path=ready['dataset'],dataset_SHA256=ready['dataset_SHA256'],original_stop_ids=[151329,151336,151338,151348],original_BF16_own_template2048_outer10_inner1_Accuracy_protocol_preserved=True,original_answer_instruction=front[0]['actual_model_prompt'][len(front[0]['prompt']):],original_existing_enable_thinking_false_adapter_SHA256=ready['original_generic_adapter_SHA256'],workbook_D_reference_authorized=True,current_A100_D_values=values,reference_workbook=str(book),reference_workbook_SHA256=sha(book),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in [OLD/'review.json',Path(ready['own_frontend200']),original,preserved,current]},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,precheck_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3)
 create(R/PROTO,p)
 cfg=dict(name=Path(CFG).stem,description=boundary,task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=True,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':AutoGLMPhone9BOwnNativeBF16',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='autoglm_phone9b_original_existing_thinkingFalse_greedy_explicit_MC',adapter=ALGO+':OwnAutoGLMPhone9BGreedy',batch_size=10,kwargs=dict(extract_answer=True,answer_instruction=p['original_answer_instruction'])),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnAutoGLMPhone9BAccuracy'],include_failed_predictions=True),baseline=dict(score=float(current_D),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
 create(R/CFG,cfg)
 spec=copy.deepcopy(parent['spec']);spec.update(approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own AutoGLM Phone9B original2048 NEW NPU3 before first200 measurement approval')
 scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec)
 entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);assert v['sample_range']==f'0:{count-1}'
 for m in [MODEL,INPUT,ALGO,NF]:importlib.import_module(m[:-3].replace('/','.'))
 create(J/'job.json',job)
 for name,h in protected.items():assert sha(name)==h,name
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,own704_source_manifest_SHA256=sha(H/'own704_active_source_manifest.json'),own_complete_input200_reused_without_replay=True,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],true3_job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=values,C_unique_current_row=row['row'],ready_for_unique_original_runner_true3=True,actual_NEW_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True)
 create(H/'readiness.json',result);create(H/'actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

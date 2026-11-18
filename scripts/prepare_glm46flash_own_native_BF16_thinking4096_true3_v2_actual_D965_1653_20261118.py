"""Own GLM4.6V source704, already verified own input200, standalone YAML and private true3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,json,os,traceback,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/GLM-4.6V-Flash';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
OLD=B/'GLM46Flash_own_original_BF16_thinking4096_frontend200_meta_v1_0616_20261118'
H=B/'GLM46Flash_own_original_BF16_thinking4096_native_source_readiness_v2_actual_D965_1653_20261118'
J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'
MODEL='universal_eval/model/adapters/glm46flash_own_original_BF16_thinking4096_native_v1_1645_20261118.py'
INPUT='universal_eval/model/adapters/glm46flash_own_original_frontend_1645_20261118.py'
ALGO='universal_eval/algorithm/glm46flash_own_original_MC_1645_20261118.py'
NF='scripts/verify_glm46flash_own_original_BF16_thinking4096_native_v1_1645_20261118.py'
SELF='scripts/prepare_glm46flash_own_native_BF16_thinking4096_true3_v2_actual_D965_1653_20261118.py'
CFG='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.6V-Flash-own-original-BF16-thinking4096-native-v2-actual-D965-1653-20261118.yaml'
PROTO='configs/protocols/glm46flash_own_original_BF16_thinking4096_native_v2_actual_D965_1653_20261118.json'
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
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 ready=js(OLD/'review.json');assert ready['passed'] and ready['own_physical_BF16_sources']==704 and ready['meta_map_complete_without_conversion'] and ready['actual_original_thinking_true_not_disabled']
 assert sha(ready['own_frontend200'])==ready['own_frontend200_SHA256']=='adb06eb931792fbbdbc8c199394c498fb33909a647e070f32ef5c079738368bc'
 original=Path(ready['own_original_saved_YAML']);assert sha(original)==ready['own_original_saved_YAML_SHA256'];saved=yaml.safe_load(original.read_bytes())
 previous=B/'GLM46Flash_own_original_BF16_thinking4096_native_source_readiness_v1_1645_20261118'
 assert sha(previous/'actual_failure.json')=='f9cfa92fb2a87026be4bfd6a129dc1c9dafef52976979020baaa930602729967'
 assert "{'server': '0.9650', 'local': '0.9650', 'preserved_original': '0.965'}" in js(previous/'actual_failure.json')['error']
 for filename,record in ready['own_weight_files'].items():assert sha(W/filename)==record['SHA256'] and (W/filename).stat().st_size==record['bytes']
 manifest=js(previous/'own704_active_source_manifest.json');assert len(manifest)==704 and all(v['source_dtype']=='torch.bfloat16' and v['source_SHA256']==v['cast_SHA256'] for v in manifest.values())
 headers=js(ready['own_source_headers']);assert set(manifest)==set(headers) and all(v['shape']==headers[n]['shape'] and v['source_file']==headers[n]['source_file'] for n,v in manifest.items())
 create(H/'own704_active_source_manifest.json',manifest);create(H/'own_unused_source_manifest.json',{})
 front=js(previous/'own_frontend200_existing_proof_field_binding_no_replay.json');assert [r['index'] for r in front]==list(range(200))
 create(H/'own_frontend200_existing_proof_field_binding_no_replay.json',front)
 create(H/'prior_failed_source_only_reuse_proof.json',dict(original_failed_job_preserved=True,no_NPU_was_started=True,original_failure=str(previous/'actual_failure.json'),original_failure_SHA256=sha(previous/'actual_failure.json'),source_manifest=str(previous/'own704_active_source_manifest.json'),source_manifest_SHA256=sha(previous/'own704_active_source_manifest.json'),own_frontend=str(previous/'own_frontend200_existing_proof_field_binding_no_replay.json'),own_frontend_SHA256=sha(previous/'own_frontend200_existing_proof_field_binding_no_replay.json'),all_own_whole_file_SHA256_freshly_checked=True,no_CPU_frontend_or_source_array_replay=True,only_workbook_reference_binding_corrected_from_old_YAML_baseline=True))
 parent=js(B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118/job.json');protected={p:sha(p) for p in parent['protected_canonical_hashes']}
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:current_D=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 values=dict(server=current_D,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']);assert all(float(v)==.965 for v in values.values()),values
 rels=[MODEL,INPUT,ALGO,NF,SELF,'scripts/prepare_glm46flash_own_original_BF16_thinking4096_frontend200_v1_0616_20261118.py','universal_eval/model/interface.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','universal_eval/utils/readonly_npz_indexed.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels}
 boundary='User workbook D is the A100 score reference, not an online AA replay gate. Own GLM-4.6V-Flash704 BF16 tensors/40text GQA2x128/vision24/151552 head; own HF greedy4096 outer10 serial1, original enable_thinking=True and exact saved custom answer instruction, own generation EOS151329/151336/151338/151348. Original installed HF SDPA/model math/native40-layer dynamic cache and all full raw values retained. Historical A100 engine/tensor bytes were not captured. No generic mock initializer, package install, borrowed checkpoint/input source/output, changed precision/budget/thinking/stop/scoring.'
 p=dict(model_id=MID,weight_path=str(W),source_tensor_count=704,own_weight_files=ready['own_weight_files'],own_assets_SHA256=ready['own_assets_SHA256'],own_source_manifest=str(H/'own704_active_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own704_active_source_manifest.json'),own_unused_MTP_source_manifest=str(H/'own_unused_source_manifest.json'),own_unused_MTP_source_manifest_SHA256=sha(H/'own_unused_source_manifest.json'),own_frontend200=str(H/'own_frontend200_existing_proof_field_binding_no_replay.json'),own_frontend200_SHA256=sha(H/'own_frontend200_existing_proof_field_binding_no_replay.json'),dataset_path=ready['dataset'],dataset_SHA256=ready['dataset_SHA256'],original_stop_ids=[151329,151336,151338,151348],original_BF16_thinking4096_outer10_inner1_Accuracy_protocol_preserved=True,original_answer_instruction=saved['algorithm']['kwargs']['answer_instruction'],workbook_D_reference_authorized=True,current_A100_D_values=values,reference_workbook=str(book),reference_workbook_SHA256=sha(book),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in [OLD/'review.json',Path(ready['own_frontend200']),original,preserved,current]},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,precheck_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3)
 create(R/PROTO,p)
 cfg=dict(name=Path(CFG).stem,description=boundary,task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':Glm46FlashOwnNativeBF16',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='glm46flash_original_thinking_greedy_explicit_MC',adapter=ALGO+':OwnGlm46FlashGreedy',batch_size=10,kwargs=dict(extract_answer=True,answer_instruction=p['original_answer_instruction'])),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnGlm46FlashAccuracy'],include_failed_predictions=True),baseline=dict(score=float(current_D),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
 create(R/CFG,cfg)
 spec=copy.deepcopy(parent['spec']);spec.update(approved=False,config=CFG,verifier=NF,npu_devices=[4],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own GLM46 thinking4096 NEW NPU3 before first200 measurement approval')
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

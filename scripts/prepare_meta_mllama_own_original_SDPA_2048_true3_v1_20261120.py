"""Bind Meta's own 906 BF16 source and original HF2048 protocol to genuine NPU3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import ast,copy,hashlib,importlib,inspect,json,os,traceback,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='meta-llama/Llama-3.2-11B-Vision-Instruct';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MllamaForConditionalGeneration')/MID
FRONT=B/'MetaMllama_own_original_HF_BF16_2048_frontend200_meta_sources_v1_20261120';H=B/'MetaMllama_own_original_SDPA_BF16_2048_source_readiness_v1_20261120';J=B/'MetaMllama_own_original_SDPA_BF16_2048_true3_v1_real_FLOCK_20261120'
MODEL='universal_eval/model/adapters/meta_mllama_own_original_BF16_SDPA_2048_native_v1_20261120.py';ALGO='universal_eval/algorithm/meta_mllama_own_original_HF_2048_MC_20261120.py';NF='scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v1_20261120.py';VIS='scripts/meta_mllama_own_visual_value_reference_20261120.py';SELF='scripts/prepare_meta_mllama_own_original_SDPA_2048_true3_v1_20261120.py'
CFG='configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct-own-original-BF16-SDPA-2048-serial-v1-20261120.yaml';PROTO='configs/protocols/meta_mllama_own_original_BF16_SDPA_2048_serial_v1_20261120.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 os.chdir(R);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 import torch
 import batch_910b as batch
 from batch_910b_support import workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from transformers import AutoConfig,MllamaForConditionalGeneration
 from transformers.integrations.sdpa_attention import sdpa_attention_forward
 assert not runner_lock_owners() and not [v for v in project_processes() if v['PID']!=os.getpid() and v['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]
 H.mkdir(exist_ok=False);ready=js(FRONT/'review.json');assert sha(FRONT/'review.json')=='1867c16c80d1ce97a46f19a7a91e13cdf1ade218c71a50ca3aa7f57158ab184d' and ready['passed'] and ready['model_id']==MID and ready['original_outer10_inner_serial1']
 assert len(ready['all906_source_to_native_semantic_name_shape_mapping_verified'])==906 and sha(ready['frontend200'])==ready['frontend200_SHA256']
 assert sha(ready['original_config'])==ready['original_config_SHA256'] and sha(ready['original_adapter'])==ready['original_adapter_SHA256']
 c=AutoConfig.from_pretrained(W,local_files_only=True);assert c._attn_implementation is c.text_config._attn_implementation is c.vision_config._attn_implementation is None
 with torch.device('meta'):m=MllamaForConditionalGeneration(c)
 assert m.config._attn_implementation==m.config.text_config._attn_implementation==m.config.vision_config._attn_implementation=='sdpa'
 assert len(list(m.parameters()))==906
 for name,rec in ready['all_source_file_sha256'].items():assert sha(W/name)==rec['sha256'] and (W/name).stat().st_size==rec['bytes']
 for name,h in ready['own_resource_sha256'].items():assert sha(W/name)==h
 parent=js(B/'AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120/job.json');protected={n:sha(n) for n in parent['protected_canonical_hashes']}
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3'
 book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';assert sha(book)=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81';rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['row']==413 and row['score'] is None and row['task']=='image-text-to-text'
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:dv=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');preserved=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(preserved)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 values=dict(server=dv,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']);assert all(Decimal(str(v))==Decimal('.775') for v in values.values())
 rels=[MODEL,ALGO,NF,VIS,SELF,'scripts/prepare_meta_mllama_own_original_HF_BF16_2048_frontend200_meta_sources_20261120.py','scripts/mllama11b_original_independent_cpu_910b.py','scripts/egm8b_original_independent_cpu_910b.py','scripts/strict_mc_independent_reference_910b.py','scripts/idefics2_original_accuracy_reference_910b.py','scripts/cosmos_reason2_stable_rope_reference_910b.py','scripts/mllama11b_masks_rope_reference_910b.py','universal_eval/model/adapters/transformers_vlm.py','universal_eval/model/interface.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','run.py','universal_eval/orchestrator/pipeline.py']
 for n in rels:ast.parse((R/n).read_bytes())
 sources={n:sha(R/n) for n in rels};external={str(Path(inspect.getfile(x))):sha(inspect.getfile(x)) for x in [MllamaForConditionalGeneration,sdpa_attention_forward]}
 p=copy.deepcopy(ready);p.update(CPU200_protocol=str(FRONT/'review.json'),CPU200_protocol_sha256=sha(FRONT/'review.json'),group_count=200,original_default_attention='sdpa',runtime_and_original_source_sha256={**{str(R/n):h for n,h in sources.items()},**external},implementation_source_SHA256=sources,current_A100_D_values=values,reference_workbook=str(book),reference_workbook_SHA256=sha(book),hard_reserve_bytes=3*2**30,log_reserve_bytes=2**30,precheck_reserve_bytes=512*2**20,per_sample_lossless_space_estimate_bytes=2*2**30,historical_A100_engine_not_captured=True)
 for r in p['input_records']:r.update(media_sha256=r['image_SHA256'],rendered=r['own_chat_template_string'],unpad_ids=r['complete_input_ids'])
 p['static_evidence_SHA256']={str(q):sha(q) for q in [FRONT/'review.json',Path(ready['frontend200']),Path(ready['original_config']),preserved,current]};create(R/PROTO,p)
 boundary='Own Meta5 SF/906BF16/511vision+projector and40text original default native SDPA. Own560/max4tiles/doubleBOS/128256 untiedhead128264embedding/2048/outer10-inner1/3EOS preserved. Old saved YAML incorrectly names unreadable other-brand root; new private YAML binds only this exact Meta own root and keeps original task/data/generation. Full independent own pixels/token200/source/meta; actual source/cast/NPU SHA and complete masks/physicalKV/fullhead audited. Historical A100 engine/tensors uncaptured, CPU values diagnostics only. No borrowed checkpoint/vocabulary/input/output, package install, eager substitution, math/cache/budget/extraction/thinking/stop tuning.'
 cfg=dict(name=Path(CFG).stem,description=boundary,task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':MetaMllamaOwnOriginal2048',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4,5])),algorithm=dict(name='meta_mllama_original_HF2048_greedy_explicit_MC',adapter=ALGO+':OwnMetaMllamaGreedy',batch_size=10,kwargs=dict(extract_answer=True,answer_instruction=p['input_records'][0]['model_prompt'][len(p['input_records'][0]['prompt']):])),metrics=dict(override=['Accuracy'],adapters=[ALGO+':OwnMetaMllamaAccuracy'],include_failed_predictions=True),baseline=dict(score=float(dv),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True));create(R/CFG,cfg)
 spec=copy.deepcopy(parent['spec']);spec.update(review_revision=boundary,approved=False,config=CFG,verifier=NF,npu_devices=[4,5],source_sha256={**sources,CFG:sha(R/CFG),PROTO:sha(R/PROTO)})
 for key in ['alignment_approval_review','alignment_approval_review_sha256']:spec.pop(key,None)
 job=copy.deepcopy(parent);job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='Own Meta original BF16 default SDPA2048 outer10-inner1 NEW NPU3 full values before first200 approval')
 scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID]
 for n in [3,200]:_,c=batch.config_for(spec,entry,n);assert c['sample_range']==f'0:{n-1}'
 importlib.import_module(MODEL[:-3].replace('/','.'));importlib.import_module(ALGO[:-3].replace('/','.'))
 create(J/'job.json',job)
 for n,h in protected.items():assert sha(n)==h
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,ready_for_unique_original_runner_true3=True,model_id=MID,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),source_SHA256=spec['source_sha256'],true3_job=str(J),job_SHA256=sha(J/'job.json'),three_D_values=values,own906_source_to_native_map_verified=True,own_frontend200_complete_reused_without_replay=True,actual_original_default_SDPA_preserved=True,NPU_devices=[4,5],max_concurrent_models=1,actual_NEW_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True)
 create(H/'readiness.json',result);create(H/'actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,no_model_inference=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():create(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

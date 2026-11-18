"""Bind own Phi-3.5 sources, original input controls and a new private real3."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib,json,shutil,struct,zipfile,xml.etree.ElementTree as E
import yaml
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='microsoft/Phi-3.5-vision-instruct';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Phi3VForCausalLM')/MID
F=B/'Phi35_own_original_chat_generation_prefix_native_tokens200_existing_pixel_proof_meta_map_v6_exact_ChatParams_no_NN_20261118'
OLD=B/'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117'
H=B/'Phi35_own_original_native_BF16_4096_source_readiness_v1_0616_20261118'
J=B/'Phi35_own_original_native_BF16_4096_true3_v1_real_FLOCK_0616_20261118'
MODEL='universal_eval/model/adapters/phi35_own_original_native_adapter_v1_0616_20261118.py'
WORKER='universal_eval/model/phi35_own_native_value_0616_20261118/worker.py'
ALGORITHM='universal_eval/algorithm/phi35_own_original_MC_0616_20261118.py'
VERIFIER='scripts/verify_phi35_own_original_native_BF16_4096_v1_0616_20261118.py'
SELF='scripts/prepare_phi35_own_original_native_BF16_4096_true3_v1_0616_20261118.py'
CFG='configs/Phi3VForCausalLM/microsoft--Phi-3.5-vision-instruct-own-original-BF16-native-4096-v1-0616-20261118.yaml'
PROTO='configs/protocols/phi35_own_original_native_BF16_4096_v1_0616_20261118.json'
LOCAL=dict(D='0.8600',row=419,book_SHA256='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34')

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(path):return json.loads(Path(path).read_bytes())
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists(),str(path)
 path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())

def main():
 import batch_910b as batch
 from batch_910b_support import workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 H.mkdir(exist_ok=False);review=js(F/'review.json');prior=js(OLD/'review.json')
 assert review['passed'] and review['actual_NEW_model_forwards']==0 and review['actual_engine_request_stop_sets']==[[2,32000]]
 assert sha(review['own_native_input200'])==review['own_native_input200_SHA256']=='3ab60cd1701f4ea810e7c886c6e7453f767bf62b74f0dafa87ee0979f25e33dd'
 assert prior['passed'] and prior['own_full_BF16_source_count']==592 and prior['own_whole_weight_bytes']==8293330888
 front=[json.loads(s) for s in Path(review['own_native_input200']).read_text().splitlines()];assert [x['index'] for x in front]==list(range(200))
 orig=R/'configs/Phi3VForCausalLM/microsoft--Phi-3.5-vision-instruct.yaml'
 assert sha(orig)=='d16e40ca27659cabbdad21e19079185544ab5ae893832b336a0274c713f2e1cc'
 saved=yaml.safe_load(orig.read_bytes());assert saved['model']['name']==MID and saved['model']['kwargs']['max_tokens']==4096 and saved['algorithm']['batch_size']==10
 protected_paths=[D/n for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']]+[Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]
 protected={str(path):sha(path) for path in protected_paths};assert protected[str(protected_paths[-1])]==LOCAL['book_SHA256']
 rows,sheet,_=workbook_rows(protected_paths[-1]);row,=[r for r in rows if r['model_id']==MID];assert row['score'] is None and row['task']=='image-text-to-text' and row['row']==LOCAL['row']
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(protected_paths[-1]) as z:current_D=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert current_D==LOCAL['D']
 original=Path(js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')['local_A100_reference_snapshot'])
 assert sha(original)=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba'
 original_D=js(original)['models'][MID]['A100_D'];assert original_D=='0.86'
 decision=D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json';assert sha(decision)=='22cbb4581a69875f643902001babfeffa694a2275e0f894a447866e356bccf3d'
 assets={};weights={};headers={}
 for name,record in prior['all_own_assets_and_runtime_sources'].items():
  path=Path(name)
  if path.parent!=W:continue
  assert path.stat().st_size==record['bytes']
  if path.suffix=='.safetensors':weights[path.name]=dict(bytes=record['bytes'],SHA256=record['SHA256'])
  else:assert sha(path)==record['SHA256'];assets[path.name]=record['SHA256']
 assert len(weights)==2 and sum(r['bytes'] for r in weights.values())==8293330888
 for name in sorted(weights):
  with (W/name).open('rb') as file:n=struct.unpack('<Q',file.read(8))[0];header=json.loads(file.read(n))
  for key,value in header.items():
   if key=='__metadata__':continue
   assert key not in headers and value['dtype']=='BF16';headers[key]=dict(value,shard=name)
 assert len(headers)==592
 put(H/'own592_source_headers.json',headers)
 code=[MODEL,WORKER,ALGORITHM,VERIFIER,SELF,'universal_eval/model/phi35_own_native_value_0616_20261118/__init__.py','universal_eval/model/interface.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/utils/readonly_npz_indexed.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','run.py','universal_eval/orchestrator/pipeline.py','universal_eval/model/reference_sources/phi35_owned_checkpoint_layout_20261118.py']+[str(path.relative_to(R)) for path in (R/'universal_eval/model/phi35_own_original_source_20261117').glob('*.py')]
 static={str(path):sha(path) for path in [F/'review.json',Path(review['own_native_input200']),OLD/'review.json',Path(prior['own_frontend']),H/'own592_source_headers.json',orig,decision,original]}
 p=dict(model_id=MID,weight_path=str(W),own_assets_SHA256=assets,own_weight_files=weights,own_source_headers=str(H/'own592_source_headers.json'),own_source_headers_SHA256=sha(H/'own592_source_headers.json'),own_frontend200=review['own_native_input200'],dataset_path=prior['dataset'],dataset_SHA256=prior['dataset_SHA256'],media_root=str(R/'workspace/datasets'),implementation_source_SHA256={name:sha(R/name) for name in code},static_evidence_SHA256=static,reference_workbook=str(protected_paths[-1]),reference_workbook_SHA256=LOCAL['book_SHA256'],current_A100_D_values=dict(server=current_D,local=LOCAL['D'],preserved_original=original_D),workbook_D_reference_authorized=True,original_generation_budget=4096,original_outer_batch=10,original_engine_stop_ids=[2,32000],original_max_model_len_override=None,own_native_source_count=574,original_native_unused_CLIP_source_count=18,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,precheck_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3,historical_source_boundary='Saved own R vLLM/Greedy4096/outer10/TP1/.9 and own original assets; current native original LongRoPE default context4096, actual stop [2,32000], own CLIP hidden_states[-2]/23 layers and full Phi3_32 physical KV. Original historical A100 engine/tensor bytes not captured. Workbook D is the user-confirmed comparison reference. Observation-only wrappers and lossless exact-value sharing do not alter model math, native scheduling, original grouping, input values or budget.')
 put(R/PROTO,p)
 cfg=dict(name=Path(CFG).stem,description=p['historical_source_boundary'],task_type='image-text-to-text',sample_count=200,sample_range='0:199',data=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=p['dataset_path']),model=dict(name=MID,adapter=MODEL+':OwnPhi35Native',model_path=str(W),download=False,kwargs=dict(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID),npu_devices=[4])),algorithm=dict(name='phi35_original_greedy_explicit_MC',adapter=ALGORITHM+':OwnPhi35OriginalGreedy',batch_size=10,kwargs=dict(extract_answer=True)),metrics=dict(override=['Accuracy'],adapters=[ALGORITHM+':OwnPhi35Accuracy'],include_failed_predictions=True),baseline=dict(score=float(current_D),threshold=.05,primary_metric='Accuracy'),output=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True))
 put(R/CFG,cfg)
 template=B/'Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117/job.json'
 job=copy.deepcopy(js(template));spec=job['spec'];spec.update(approved=False,review_revision='Own Phi35 saved original native BF16/4096/outer10, complete source574+unused18 and full numeric observations',config=CFG,verifier=VERIFIER,npu_devices=[4],source_sha256={name:sha(R/name) for name in code+[CFG,PROTO]})
 for key in ['alignment_approval_review','alignment_approval_review_sha256','protocol_file','protocol_sha256']:spec.pop(key,None)
 job.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,write_workbook=False,created_at=datetime.now(timezone.utc).isoformat(),private_stage_type='NEW_real_NPU3_own_native_full_values_before_first200_approval')
 assert shutil.disk_usage(R).free>12*1024**3
 J.mkdir(exist_ok=False);put(J/'job.json',job);scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec)
 entry=js(D/'state.json')['models'][MID];_,small=batch.config_for(spec,entry,3);_,formal=batch.config_for(spec,entry,200)
 assert small['sample_range']=='0:2' and formal['sample_range']=='0:199'
 # Actual imports resolve all observer, independent verifier and algorithm APIs
 # before admission. They instantiate no LLM/worker or learned model.
 for module in ['universal_eval.model.phi35_own_native_value_0616_20261118.worker','universal_eval.model.adapters.phi35_own_original_native_adapter_v1_0616_20261118','universal_eval.algorithm.phi35_own_original_MC_0616_20261118','verify_phi35_own_original_native_BF16_4096_v1_0616_20261118']:importlib.import_module(module)
 assert protected=={path:sha(path) for path in protected}
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id=MID,own_source_count=592,expected_actual_native_source_count=574,expected_original_native_unused_sources=18,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),true3_job=str(J),job_SHA256=sha(J/'job.json'),source_SHA256=spec['source_sha256'],ready_for_original_runner_true3=True,actual_new_model_forwards=0,first200_not_approved=True,no_G_or_canonical_write=True)
 put(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()

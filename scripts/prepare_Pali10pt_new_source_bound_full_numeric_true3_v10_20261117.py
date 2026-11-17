"""New missing numerical-value3 phase, preserving old compatibility3 and B10."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,ast,inspect,shutil,zipfile,xml.etree.ElementTree as ET
from decimal import Decimal
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='google/paligemma2-10b-pt-448';H=B/'Pali10Pt448_v10_correct_full_numeric_contract_source_bound_readiness_20261117';J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 from universal_eval.model.adapters.pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117 import Pali10Pt448OwnFullValuesNamedRPCV4
 assert not inspect.isabstract(Pali10Pt448OwnFullValuesNamedRPCV4);H.mkdir(exist_ok=False);assert not J.exists()
 old=B/'Pali10Pt448_A100_original_native_vllm_compat3_1322_v2_20261110';b10=B/'Pali10Pt448_A100_original_native_vllm_B10_1322_v2_20261110'
 done=js(old/'completion.json');assert done['actual_returncode']==0;a,=done['accepted'];assert sha(a['proof'])==a['proof_sha256']=='13f4049cb3e6440948ec628140a72fff27e5eaa5a8e856dc547fa4f581e12917'
 ten=js(b10/'actual_completion.json');assert ten['actual_returncode']==0 and sha(ten['proof'])==ten['proof_SHA256']=='bf0f1c1bb504fa98af0de2f1999a8c8f8405d326f712553ffc7466713b871b1a'
 original=js(old/'job.json');cfg=js(R/original['spec']['config']);p=copy.deepcopy(js(old/'protocol.json'));report=Path(a['report_dir']);legacy_line=js(report/'operator_lineage.json')['actual_execution'];trace=Path(legacy_line['request_trace']);assert len(trace.read_text().splitlines())==3
 pre=B/'Pali10Pt448_completed_compat3_B10_missing_full_numeric_source_header_preflight_20261117';preflight=js(pre/'readiness_NOT_dispatch_ready.json');assert preflight['source_header_preflight_passed'] and preflight['own_source_count']==903 and not preflight['CPU_or_NPU_model_forward_performed'] and not preflight['dispatch_ready'];header=Path(preflight['own_source_headers']);assert sha(header)==preflight['own_source_headers_SHA256']=='cee3f84c2288ffad55f524e7aff88a830bbeb88e19d61afc947d432c6e4b055f'
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d';rows,sheet,_=workbook_rows(book);row,=[x for x in rows if x['model_id']==MID];assert row['score'] is None and row['task']==cfg['task_type']=='image-text-to-text'
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:value=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert Decimal(value)==Decimal('.0701');current=js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D'];preserved=js(binding['local_A100_reference_snapshot'])['models'][MID]['A100_D'];p['A100_D']=dict(current_remote=value,current_local=str(current),preserved_local=str(preserved))
 nfs='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v9_original_async_EOS_commit_20261117.py';adapter='universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117.py';worker='universal_eval/model/pali10pt448_full_native_value_20261117/worker_v2_ABI.py';me=str(Path(__file__).relative_to(R))
 sources={k:sha(R/k) for k in original['spec']['source_sha256']};sources.update({k:sha(R/k) for k in [nfs,adapter,worker,'universal_eval/model/pali10pt448_full_native_value_20261117/worker_named_RPC_v3.py',me,'universal_eval/model/pali10pt448_full_native_value_20261117/__init__.py']})
 assert sources['universal_eval/model/adapters/paligemma2_10b_pt448_A100_native_vllm_20261110_adapter.py']=='652112898c2d03e7174102439853a7e820ab0581db319f1037e063d8d84ca8b8'
 p.update(observation_only_worker_extension_cls='universal_eval.model.pali10pt448_full_native_value_20261117.worker_named_RPC_v3.OwnPaliFullValueWorkerExtension',full_numeric_observer_not_model_math_change=True,full_numeric_implementation_sources_SHA256=copy.deepcopy(sources),own_source_headers=str(header),own_source_headers_SHA256=sha(header),old_compat3_proof=a['proof'],old_compat3_proof_SHA256=a['proof_sha256'],old_B10_proof=ten['proof'],old_B10_proof_SHA256=ten['proof_SHA256'],old_original_request_trace=str(trace),old_original_request_trace_SHA256=sha(trace),full_numeric_evidence_job=str(J/'job.json'),old_completed_compatibility_stages_not_repeated_as_compatibility_tests=True,new_phase_collects_missing_full_actual_native_values_for_independent_numerical_review=True,no_A100_online_control_extra_prerequisite=True,full_numeric_actual_worker_schema_pending_real3=True,final_original_protocol_qualification_pending_until_fixed200=True)
 for f in [header,pre/'readiness_NOT_dispatch_ready.json',Path(a['proof']),Path(ten['proof']),old/'completion.json',b10/'actual_completion.json',trace]:p['static_evidence_sha256'][str(f)]=sha(f)
 decision=D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json';assert sha(decision)=='22cbb4581a69875f643902001babfeffa694a2275e0f894a447866e356bccf3d';p['static_evidence_sha256'][str(decision)]=sha(decision)
 diagnostic=B/'Pali10Pt448_existing_complete_real3_v9_original_async_EOS_commit_full_numeric_diagnostic_original_FLOCK_20261117';d=js(diagnostic/'actual_completion.json');assert d['actual_returncode']==d['underlying_verifier_returncode']==0 and d['all_old_reports_job_and_canonical_SHA_unchanged']
 assert sha(diagnostic/'verification_stdout.json')==d['review_SHA256']=='5fbf9e68ee14ba22d244d34749b284044d90e8738f1dc824f68c4b57f7e5cffa';review=js(diagnostic/'verification_stdout.json');assert review['diagnostic_only'] and not review['NEW_real_NPU_inference'] and review['numeric_review_passed']
 assert sha(R/nfs)=='988de3e8679f77f8465f317a22f1fd7de5c8c5319c21af3d00d4f9d6555420ad'
 for f in [diagnostic/'actual_completion.json',diagnostic/'verification_stdout.json']:p['static_evidence_sha256'][str(f)]=sha(f)
 p.update(corrected_numeric_contract_readonly_full3_diagnostic_review=str(diagnostic/'verification_stdout.json'),old_failed_v4_job_not_promoted=True,new_real3_required_with_this_exact_final_source_binding=True)
 pp=R/'configs/protocols/pali10pt448_own_original_vllm_BF16_1024_new_full_native_true3_v10_correct_numeric_contract_20261117.json';create(pp,p)
 cf=R/'configs/PaliGemmaForConditionalGeneration/google--paligemma2-10b-pt-448-own-original-vllm-BF16-1024-full-native-value-true3-v10-correct-numeric-contract-20261117.yaml';cfg['name']=cf.stem;cfg['description']='Own original native vLLM BF16/1024, sampling and batch10 unchanged; only the standard observation worker_extension_cls is added and named RPC is used. New missing full-value numerical3 phase; old accepted compatibility3/B10 preserved, not reclassified as full-numerical acceptance.';cfg['model']['adapter']=adapter+':Pali10Pt448OwnFullValuesNamedRPCV4';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));cfg['baseline']['score']=float(value);create(cf,cfg)
 # The model, data, batch, token budget, sampling and scoring remain the
 # already-accepted own original deployment. Only observation/source binding.
 before=js(R/original['spec']['config']);assert cfg['data']==before['data'] and cfg['algorithm']==before['algorithm'] and cfg['metrics']==before['metrics'] and cfg['output']==before['output'] and cfg['model']['model_path']==before['model']['model_path'] and cfg['model']['kwargs']['protocol']==before['model']['kwargs']['protocol']
 template=js(B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_explicit_CPU_diagnostic_role_real_FLOCK_20261117/job.json');job=copy.deepcopy(template);job.update(model_id=MID,stages=[3],created_at=datetime.now(timezone.utc).isoformat());job['protected_canonical_hashes']={f:sha(f) for f in template['protected_canonical_hashes']};sources.update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)})
 job['spec']=dict(approved=False,config=str(cf.relative_to(R)),verifier=nfs,npu_devices=[4],source_sha256=sources,smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600)
 batch.check_sources(job['spec']);scope(job['spec'],[3],False);entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 for count in [3,200]:_,c=batch.config_for(job['spec'],entry,count);validate_config(c,MID)
 assert shutil.disk_usage(D).free>=12*1024**3
 for n,h in sources.items():
  assert sha(R/n)==h;f=H/'source_original_bytes'/(n+'.txt');f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((R/n).read_bytes())
 create(J/'job.json',job);ready=dict(UTC=datetime.now(timezone.utc).isoformat(),passed_pre_dispatch_source_config_and_old_stage_protection=True,model_id=MID,true3_job=str(J),job_SHA256=sha(J/'job.json'),standalone_YAML=str(cf),standalone_YAML_SHA256=sha(cf),old_compat3_and_B10_preserved=True,original_native_model_math_dtype1024_batch10_prompt_extraction_scoring_identical=True,only_observation_worker_extension_keyword_added=True,no_insecure_callable_serialization_enabled=True,new_phase_missing_full_native_value_collection_only=True,real_NPU_and_independent_full_numeric_acceptance_still_pending=True,not_formal200_not_approval_not_G=True,current_three_D=p['A100_D']);create(H/'readiness.json',ready);print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':main()

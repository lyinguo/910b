"""Own Merve frozen resources/603 source preflight and independent config; no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,struct,zipfile,xml.etree.ElementTree as ET,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='merve/paligemma_vqav2';H=B/'MervePali_own603_original_native_2048_full_numeric_candidate_preflight_no_NN_20261117';J=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import workbook_rows,validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 from universal_eval.model.adapters.merve_pali_own603_original_native_full_values_v1_20261117 import MervePaliOwn603FullNativeValues
 assert not inspect.isabstract(MervePaliOwn603FullNativeValues);H.mkdir(exist_ok=False);assert not J.exists()
 old=B/'MervePaliGemma_A100_original_native_compat3_1322_v4_spawn_20261110';done=js(old/'completion.json');assert done['actual_returncode']==0;a,=done['accepted'];assert sha(a['proof'])==a['proof_sha256']=='b5bd29b39bb3833fb0679dc7f86dd2558f33a581dadcd13eae01fbb7e1d89903';job=js(old/'job.json');cfg=js(R/job['spec']['config']);p=copy.deepcopy(js(old/'protocol.json'));W=Path(p['weight_path']);assert W.is_dir()
 resources={}
 for n,h in p['resource_sha256'].items():assert sha(W/n)==h;resources[n]=h
 assert not (W/'adapter_config.json').exists() and (W/'adapter_config.json.universal_eval.bak').exists();headers={}
 for f in W.glob('*.safetensors'):
  with f.open('rb') as inp:n=struct.unpack('<Q',inp.read(8))[0];block=json.loads(inp.read(n))
  for key,value in block.items():
   if key=='__metadata__':continue
   assert key not in headers and value['dtype']=='BF16';headers[key]=dict(value,shard=f.name)
 assert len(headers)==603;index=js(W/'model.safetensors.index.json');assert set(index['weight_map'])==set(headers) and all(v['shard']==index['weight_map'][k] for k,v in headers.items());put(H/'own603_original_source_headers.json',headers)
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==binding['local_workbook_sha256'];rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['score'] is None and row['task']==cfg['task_type'];ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:d=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 p['A100_D']=dict(current_remote=d,current_local=str(js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D']),preserved_local=str(js(binding['local_A100_reference_snapshot'])['models'][MID]['A100_D']))
 nf='scripts/verify_merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117.py';adapter='universal_eval/model/adapters/merve_pali_own603_original_native_full_values_v1_20261117.py';cn='scripts/continue_merve_pali_own603_after_Granite_publication_release_true3_fresh3_200_20261117.py';names=list(job['spec']['source_sha256'])+[nf,adapter,cn,str(Path(__file__).relative_to(R)),'universal_eval/model/merve_pali_full_native_value_20261117/worker.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_named_RPC.py','universal_eval/model/merve_pali_full_native_value_20261117/__init__.py'];sources={n:sha(R/n) for n in names};oldtrace=Path(js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution']['request_trace'])
 p.update(full_numeric_observer_not_model_math_change=True,observation_only_worker_extension_cls='universal_eval.model.merve_pali_full_native_value_20261117.worker_named_RPC.OwnMerveFullValueWorkerExtension',full_numeric_implementation_sources_SHA256=copy.deepcopy(sources),own_source_headers=str(H/'own603_original_source_headers.json'),own_source_headers_SHA256=sha(H/'own603_original_source_headers.json'),old_compat3_proof=a['proof'],old_compat3_proof_SHA256=a['proof_sha256'],old_original_request_trace=str(oldtrace),old_original_request_trace_SHA256=sha(oldtrace),full_numeric_evidence_job=str(J/'job.json'),new_phase_collects_missing_full_native_values_not_old_compatibility_replay=True,no_A100_online_control_extra_prerequisite=True,original_root_name_or_path=js(W/'config.json').get('_name_or_path'),original_inactive_adapter_backups_not_renamed_or_replaced=True,all_frozen_original_AA_resource_SHA_match=True,intrinsic_RoPE_reconstruction_has_no_checkpoint_or_learned_values=True)
 for f in [H/'own603_original_source_headers.json',Path(a['proof']),old/'completion.json',oldtrace]:p['static_evidence_sha256'][str(f)]=sha(f)
 pp=R/'configs/protocols/merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117.json';put(pp,p);cf=R/'configs/PaliGemmaForConditionalGeneration/merve--paligemma_vqav2-own-original-native-BF16-2048-full-values-v1-20261117.yaml';cfg.update(name=cf.stem,sample_count=200,sample_range='0:199');cfg['model']['adapter']=adapter+':MervePaliOwn603FullNativeValues';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));cfg['baseline']['score']=float(d);put(cf,cfg)
 sources.update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});base=js(B/'Granite41_completed_numeric3_approved_NEWfresh3_fixed200_v4_runner_return_schema_real_FLOCK_20261117/job.json');new=copy.deepcopy(base);new.update(model_id=MID,stages=[3]);new['spec']=dict(approved=False,config=str(cf.relative_to(R)),verifier=nf,npu_devices=[4],source_sha256=sources,smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600);new['protected_canonical_paths_at_actual_dispatch']=list(base['protected_canonical_hashes']);new.pop('protected_canonical_hashes');new['not_dispatchable_until_current_Granite_result_decision_and_new_canonical_binding']=True;batch.check_sources(new['spec']);scope(new['spec'],[3],False)
 entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,MID)
 put(H/'dispatch_template_NOT_actual_job.json',new);v=dict(UTC=datetime.now(timezone.utc).isoformat(),passed_source_config_own_resource_preflight_no_NN=True,model_id=MID,own_source_count=603,old_completed_compat3_not_repeated=True,all_frozen_original_weight_resource_SHA_match=True,own_original_2048_batch10_and_math_unchanged=True,template=str(H/'dispatch_template_NOT_actual_job.json'),template_SHA256=sha(H/'dispatch_template_NOT_actual_job.json'),new_full_numeric3_not_yet_started=True,no_formal200_approval_or_G_claimed=True,current_three_D=p['A100_D']);put(H/'readiness.json',v);print(json.dumps(v,ensure_ascii=False))
if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64,copy
P=Path(__file__).parent
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
mp=json.loads((P/'R_current_Merve_full_value_before_parallel_binding_actual2_20261117.json').read_bytes())
pp=json.loads((P/'R_Prithiv_device_binding_Merve_lossless_diagnostic_actual2_20261117.json').read_bytes())
files={}
def add(n,text):
 ast.parse(text);raw=text.encode();files[n]=dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest());(P/Path(n).name).write_bytes(raw);return n
worker=(P/'A100_alignment_parallel_stage_worker_20261117.py').read_text(encoding='utf-8')
worker=worker.replace("assert _PARALLEL;kernel_lock_owned", "assert _PARALLEL;assert not pathlib.Path(_PARALLEL['manifest']['coordinator_failure_pending']).exists(),'Coordinator failure: retain owned stages and wait for natural exits';kernel_lock_owned")
W=add('scripts/A100_alignment_parallel_stage_worker_v2_20261117.py',worker)
master=(P/'A100_alignment_parallel_coordinator_20261117.py').read_text(encoding='utf-8')
master=master.replace('one_original_global_exclusive_FLOCK=True,','coordinator_failure_pending=str(home/\'coordinator_failure_pending_natural_children.json\'),one_original_global_exclusive_FLOCK=True,')
master=master.replace(" except BaseException:\n  # No cancellation", " except BaseException:\n  put(home/'coordinator_failure_pending_natural_children.json',dict(UTC=now(),error=traceback.format_exc(),no_signals_or_restarts=True))\n  # No cancellation")
M=add('scripts/A100_alignment_parallel_coordinator_v2_20261117.py',master)
rpc_old=mp['sources']['universal_eval/model/merve_pali_full_native_value_20261117/worker_named_RPC.py']['text']
rpc=rpc_old.replace('class OwnMerveFullValueWorkerExtension:','class OwnMerveLosslessValueWorkerExtension:').replace('merve_pali_full_native_value_20261117.worker import','merve_pali_full_native_value_20261117.worker_lossless_v2 import')
assert rpc.replace('OwnMerveLosslessValueWorkerExtension','OwnMerveFullValueWorkerExtension').replace('merve_pali_full_native_value_20261117.worker_lossless_v2 import','merve_pali_full_native_value_20261117.worker import')==rpc_old
RPC=add('universal_eval/model/merve_pali_full_native_value_20261117/worker_named_lossless_RPC_v2.py',rpc)
old_ext='universal_eval.model.merve_pali_full_native_value_20261117.worker_named_RPC.OwnMerveFullValueWorkerExtension';new_ext='universal_eval.model.merve_pali_full_native_value_20261117.worker_named_lossless_RPC_v2.OwnMerveLosslessValueWorkerExtension'
adapter_old=(P/'inspection_merve_pali_own603_original_native_full_values_v1_20261117.py').read_text(encoding='utf-8')
adapter=adapter_old.replace(old_ext,new_ext).replace('class MervePaliOwn603FullNativeValues(', 'class MervePaliOwn603LosslessValues(')
assert adapter.replace(new_ext,old_ext).replace('class MervePaliOwn603LosslessValues(','class MervePaliOwn603FullNativeValues(')==adapter_old
MA=add('universal_eval/model/adapters/merve_pali_own603_original_native_lossless_values_v2_20261117.py',adapter)
nf='''"""All original full numeric checks plus exact storage-sharing provenance."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,zipfile,contextlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from verify_merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117 import verify as original_verify
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def verify(folder,count):
 value=original_verify(folder,count);assert value['passed'] and value['numeric_review_passed'];line=js(folder/'operator_lineage.json')['actual_execution'];p=js(js(folder/'eval_report.json')['pipeline_config']['model']['kwargs']['protocol_file']);root=Path(__file__).resolve().parents[1];external=value['external_evidence_sha256'];reviews=[]
 for name in ['universal_eval/utils/lossless_npz_aliases.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py','__RPC__','__MA__']:
  assert sha(root/name)==p['full_numeric_implementation_sources_SHA256'][name];external[str(root/name)]=sha(root/name)
 assert p['lossless_storage_full_value_preservation_review_SHA256']==sha(p['lossless_storage_full_value_preservation_review']);prior=js(p['lossless_storage_full_value_preservation_review']);assert prior['numeric_review_passed'] and prior['diagnostic_only'];external[p['lossless_storage_full_value_preservation_review']]=sha(p['lossless_storage_full_value_preservation_review'])
 index=js(line['full_numeric_group_index'])
 for item in index['groups']:
  g=js(item['path']);s=g['lossless_shared_tensor_storage'];assert s['schema']=='complete_tensor_occurrences_exact_value_sharing_v1' and s['no_tensor_or_occurrence_or_numeric_check_dropped'] and s['no_precision_or_model_math_or_sampling_change'] and s['every_duplicate_compared_bitwise_to_actual_current_NPU_value'];assert sha(s['storage_source'])==s['storage_source_SHA256']==p['full_numeric_implementation_sources_SHA256']['universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py'];external[s['storage_source']]=s['storage_source_SHA256']
  with zipfile.ZipFile(g['archive']) as z:names={x.filename.removesuffix('.npy') for x in z.infolist()};assert len(names)==s['physical_unique_arrays']
  refs=[]
  def walk(x):
   if isinstance(x,dict):
    if {'array','shape','dtype','SHA256'}<=set(x):refs.append(x['array'])
    for v in x.values():walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
  walk(g);assert set(refs)==names and len(refs)==s['all_actual_tensor_occurrences'];assert len(refs)-len(names)==s['bitwise_duplicate_occurrences'];reviews.append(dict(group_index=g['group_index'],all_occurrences=len(refs),unique_complete_arrays=len(names),all_original_numeric_assertions_still_executed=True,storage=s))
 value.update(lossless_shared_storage_full_value_checks=reviews,no_original_NPU_capture_occurrence_or_numeric_check_dropped=True,new_storage_is_exact_value_sharing_not_pruned_numeric_evidence=True,all_original_numeric_checks_retained=True,external_evidence_sha256=external,independent_verifier_SHA256=sha(__file__))
 return value
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):value=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
'''.replace('__RPC__',RPC).replace('__MA__',MA)
MN=add('scripts/verify_merve_pali_own603_original_native_lossless_full_values_v2_20261117.py',nf)
pa_name='universal_eval/model/adapters/prithiv_gemma4_e4b_own_original_BF16_native_v1_20261116.py';paold=pp['sources'][pa_name]['text'];oldcl='class PrithivGemma4E4BOwnBF16(Gemma4LosslessObserver):';newcl='class PrithivGemma4E4BOwnBF16Card2(Gemma4LosslessObserver):'
ctor="\n    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):\n        assert protocol['model_id']==MID and npu_devices==[2]\n        self.protocol_file=Path(protocol_file);self.protocol_sha256=protocol_sha256;self.devices=npu_devices\n"
pa=paold.replace(oldcl,newcl+ctor).replace('No running processes found in NPU 4','No running processes found in NPU 2').replace("self.device=torch.device('npu:4')","self.device=torch.device('npu:2')").replace("hardware='Ascend910B',devices=[4]","hardware='Ascend910B',devices=[2]")
assert pa.replace(newcl+ctor,oldcl).replace('No running processes found in NPU 2','No running processes found in NPU 4').replace("self.device=torch.device('npu:2')","self.device=torch.device('npu:4')").replace("hardware='Ascend910B',devices=[2]","hardware='Ascend910B',devices=[4]")==paold
PA=add('universal_eval/model/adapters/prithiv_gemma4_e4b_own_original_BF16_card2_affinity_only_v1_20261117.py',pa)
pno='scripts/verify_prithiv_gemma4_e4b_own_original_BF16_native_v2_exact_own_report_path_20261116.py';pnold=pp['sources'][pno]['text'];pn=pnold.replace("rec['actual_NPU_device']=='npu:4'","rec['actual_NPU_device']=='npu:2'").replace("x['Device_id']=='4'","x['Device_id']=='2'");extra="    assert cfg['model']['kwargs']['npu_devices']==line['devices']==[2], 'Exact independent card2 deployment affinity'\n";needle="    assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B'\n";assert pn.count(needle)==1;pn=pn.replace(needle,needle+extra)
assert pn.replace(extra,'').replace("rec['actual_NPU_device']=='npu:2'","rec['actual_NPU_device']=='npu:4'").replace("x['Device_id']=='2'","x['Device_id']=='4'")==pnold
PN=add('scripts/verify_prithiv_gemma4_e4b_own_original_BF16_card2_affinity_only_v1_20261117.py',pn)
review=dict(original_frozen_runner_file_never_changed=True,original_Merve_loader_and_predict_and_named_RPC_math_source_reverse_delta_byte_identical=True,Prithiv_changes_only_real_device_2_constructor_load_occupancy_lineage_and_two_independent_device_guards=True,Prithiv_original_load_own_model_and_predict_and_observer_functions_byte_identical=True,all_original_NPU_math_precision_budget_cache_inputs_sampling_and_numeric_assertions_unchanged=True,coordinator_failure_releases_only_after_all_actually_owned_children_naturally_exit=True,sources={n:v['SHA256'] for n,v in files.items()})
(P/'two_card_model_affinity_lossless_source_review_v2_20261117.json').write_bytes(json.dumps(review,indent=2).encode())
payload=dict(files=files,merve_job=mp['job'],merve_config=mp['config'],merve_protocol=mp['protocol']['value'],merve_completion=mp['completion'],prithiv_job=pp['job'],prithiv_config=pp['config'],prithiv_protocol=pp['protocol']['value'],prithiv_completion=pp['completion'],expected_canonical=mp['current_canonical'],review=review,core_worker=W,core_master=M,MA=MA,MN=MN,PA=PA,PN=PN,RPC=RPC,new_extension=new_ext)
(P/'two_card_model_affinity_lossless_payload_v2_20261117.json').write_bytes(json.dumps(payload,ensure_ascii=False).encode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,ast,subprocess,os
R=Path('/root/projects/comperror');PAYLOAD=__PAYLOAD__
for n,v in PAYLOAD['files'].items():
 p=R/n;raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=subprocess.run(['python3','-B',PAYLOAD['core_worker'],'--self-test'],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_core_selftest_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_review=PAYLOAD['review'],no_production_NN_or_lock_used=True),ensure_ascii=False));raise SystemExit(r.returncode)
'''.replace('__PAYLOAD__',repr(payload));ast.parse(remote);(P/'deploy_test_two_card_model_affinity_lossless_v2_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(review,ensure_ascii=False,indent=2))

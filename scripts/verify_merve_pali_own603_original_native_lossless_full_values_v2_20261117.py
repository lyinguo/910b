"""All original full numeric checks plus exact storage-sharing provenance."""
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
 for name in ['universal_eval/utils/lossless_npz_aliases.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py','universal_eval/model/merve_pali_full_native_value_20261117/worker_named_lossless_RPC_v2.py','universal_eval/model/adapters/merve_pali_own603_original_native_lossless_values_v2_20261117.py']:
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

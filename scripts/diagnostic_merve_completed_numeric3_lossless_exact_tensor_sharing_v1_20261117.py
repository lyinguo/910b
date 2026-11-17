"""Full old numeric3 checks read exact aliases; no new NN or old-job promotion."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,contextlib,traceback,zipfile,shutil
import numpy as np
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.lossless_npz_aliases import copy_with_exact_aliases,AliasArchive
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=B/'Merve_completed_numeric3_lossless_exact_tensor_sharing_full_independent_diagnostic_v1_20261117';J=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117'
def main():
 H.mkdir(exist_ok=False);assert shutil.disk_usage(D).free>7*1024**3;done=js(J/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0;a=done['accepted'][0];assert sha(a['proof'])==a['proof_sha256']=='d8b7425d08dfbc19077f41c83b79dd296671802f9f56f84bc8555e2f8481cdd3';folder=Path(a['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution'];idx=js(line['full_numeric_group_index']);protected={str(folder/n):sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']};protected.update({str(J/'completion.json'):sha(J/'completion.json'),a['proof']:a['proof_sha256']});redirect={};audits=[]
 for group in idx['groups']:
  g=js(group['path']);source=Path(g['archive']);assert sha(source)==g['archive_SHA256'];protected[str(source)]=sha(source);dest=H/source.name;rec=copy_with_exact_aliases(source,dest);aliasfile=dest.with_suffix('.aliases.json');aliasfile.write_bytes(json.dumps(rec,ensure_ascii=False,indent=2).encode());redirect[str(source.resolve())]=(str(dest),rec['aliases'])
  with zipfile.ZipFile(dest) as z:parts=[dict(name=i.filename,uncompressed=i.file_size,compressed=i.compress_size) for i in z.infolist()]
  audits.append(dict(original_archive=str(source),original_archive_SHA256=sha(source),complete_original_occurrences=rec['original_occurrences'],complete_shared_unique_arrays=rec['physical_unique_arrays'],lossless_archive=str(dest),lossless_archive_SHA256=sha(dest),aliasfile=str(aliasfile),aliasfile_SHA256=sha(aliasfile),physical_uncompressed_bytes=sum(p['uncompressed'] for p in parts),complete_parts=parts,all_original_values_bitwise_preserved=True))
 original_np_load=np.load
 def exact_loader(file,*args,**kwargs):
  if isinstance(file,(str,Path)) and str(Path(file).resolve()) in redirect:
   assert not args and kwargs=={'allow_pickle':False};archive,aliases=redirect[str(Path(file).resolve())];return AliasArchive(archive,aliases,original_np_load)
  return original_np_load(file,*args,**kwargs)
 assert sha(R/'scripts/verify_merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117.py')=='d8b149bd2fd062fed6f9c89a9e9345be72c5143759684e9244dd2d91fb9a8816'
 import verify_merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117 as original
 np.load=exact_loader
 try:
  with contextlib.redirect_stdout(sys.stderr):proof=original.verify(folder,3)
 finally:np.load=original_np_load
 assert proof['passed'] and proof['numeric_review_passed'] and proof['actual_native_model_forward_calls']==9 and proof['sample_count']==3
 for p,h in protected.items():assert sha(p)==h,p
 uncompressed=sum(v['physical_uncompressed_bytes'] for v in audits);metadata=sum(Path(v['aliasfile']).stat().st_size for v in audits);need=int((uncompressed+metadata)*200/3*4+6*1024**3);free=shutil.disk_usage(D).free
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=False,diagnostic_only=True,no_NEW_NPU_inference=True,no_old_job_or_proof_promoted_or_rewritten=True,all_original_full_numeric3_assertions_executed_against_lossless_stored_values=True,numeric_review_passed=True,original_full_numeric_proof=proof,lossless_storage_full_array_audits=audits,capacity_review=dict(empirical_estimate_bytes=need,actual_free_bytes=free,passed=need<=free,factor=4,extra_budget_bytes=6*1024**3,all_metadata_overhead_included_conservatively=True,empirical_not_worst_case_guarantee=True,original_3GiB_hard_1GiB_log_halfGiB_precheck_and_batch_count_times2GiB_guards_unchanged=True),all_protected_before_after_SHA=protected,source_sha256={str(Path(__file__)):sha(__file__),str(R/'universal_eval/utils/lossless_npz_aliases.py'):sha(R/'universal_eval/utils/lossless_npz_aliases.py'),str(R/'universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py'):sha(R/'universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py')},no_canonical_G_state_registry_or_source_mutation=True,no_original_array_deleted=True)
 (H/'review.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps(dict(UTC=result['UTC'],actual_numeric_diagnostic_passed=True,old_full_numeric_assertions_all_executed=True,review=str(H/'review.json'),review_SHA256=sha(H/'review.json'),capacity_review=result['capacity_review'],all_original_occurrences_preserved=sum(x['complete_original_occurrences'] for x in audits),physical_unique_arrays=sum(x['complete_shared_unique_arrays'] for x in audits),no_NEW_NPU=True),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():(H/'failure.json').write_bytes(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_original_array_modified=True),ensure_ascii=False,indent=2).encode())
  raise

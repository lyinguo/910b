"""Completed own three samples only: full original NF against exact new storage.

No NPU model run, new qualification, old-source mutation or evidence deletion.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,contextlib,traceback,fcntl,zipfile,shutil,math,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.lossless_complete_alias_npz_fortran_head_20261117 import copy_complete_arrays,CompleteAliasNPZ
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
H=B/'Prithiv_completed_card2_true3_exact_alias_standard_Fortran_full_head_storage_diagnostic_only_20261117';J=B/'PrithivGemma4E4B_existing_model_first_independent_card2_true3_parallel_v1_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def save(p,v):p.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 assert not H.exists() and not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','legacy_controller']]
 H.mkdir();started=datetime.now(timezone.utc).isoformat();save(H/'state.json',dict(phase='copying_complete_old_values_without_NN',started_UTC=started,original_small3_not_replayed=True))
 canonical={str(p):sha(p) for p in [Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]+[D/n for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json']]}
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged();assert shutil.disk_usage(D).free>5*1024**3
  done=js(J/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0;a=done['accepted'][0];assert sha(a['proof'])==a['proof_sha256']=='94f5649a4117467baae93dee177a5b794fc4d171b4ae8d962c9067bfc251f645';folder=Path(a['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution'];rows=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert len(rows)==3
  protected={str(folder/n):sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']};protected.update({str(J/'completion.json'):sha(J/'completion.json'),a['proof']:a['proof_sha256']});redirect={};audits=[]
  for row in rows:
   e=js(row['actual_execution_file']);src=Path(e['archive']);assert sha(src)==e['archive_SHA256'];protected[str(src)]=sha(src);dst=H/src.name;m=copy_complete_arrays(src,dst);redirect[str(src.resolve())]=dst
   with zipfile.ZipFile(dst) as z:
    parts={x.filename:x for x in z.infolist()};activation={n.replace('.','_') for n in e['activation_prefill']};mandatory={owner for name,owner in m['aliases'].items() if name not in activation};extra={owner for name,owner in m['aliases'].items() if name in activation}-mandatory
    first3_only=sum(parts[n+'.npy'].compress_size for n in extra);common=dst.stat().st_size-first3_only
   index=dst.with_suffix('.aliases.json');audits.append(dict(index=e['index'],prompt_length=e['prompt_length'],forwards=e['actual_native_forwards'],original_archive=str(src),original_archive_SHA256=sha(src),original_archive_bytes=src.stat().st_size,lossless_archive=str(dst),lossless_archive_SHA256=sha(dst),lossless_archive_bytes=dst.stat().st_size,complete_alias_index=str(index),complete_alias_index_SHA256=sha(index),complete_metadata_bytes=index.stat().st_size,all_original_occurrences=m['original_occurrences'],physical_unique_arrays=m['physical_unique_arrays'],all_original_complete_values_and_occurrences_preserved=True,first3_only_prefill_payload_bytes=first3_only,mandatory_common_payload_headers_bytes=common))
   save(H/'state.json',dict(phase='copying_complete_old_values_without_NN',completed_storage_copies=len(audits),current_storage_audits=audits,started_UTC=started))
  name='verify_prithiv_gemma4_e4b_own_original_BF16_card2_affinity_only_v1_20261117';source=R/'scripts'/f'{name}.py';assert sha(source)=='699370ccd31a4ffa6d75a0fc8786f88b37a33eaf5d0dbd1bd158af88aaf0894e';original=importlib.import_module(name);reader=original.ReadOnlyNPZ
  def exact_reader(path,*args,**kwargs):
   p=str(Path(path).resolve())
   if p in redirect:
    assert not args and not kwargs
    return CompleteAliasNPZ(redirect[p])
   return reader(path,*args,**kwargs)
  original.ReadOnlyNPZ=exact_reader;save(H/'state.json',dict(phase='independent_full_original_numeric3_against_exact_new_storage_no_NN',started_UTC=started,storage_audits=audits))
  try:
   with contextlib.redirect_stdout(sys.stderr):proof=original.verify(folder,3)
  finally:original.ReadOnlyNPZ=reader
  assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==475
  for path,value in protected.items():assert sha(path)==value
  assert {p:sha(p) for p in canonical}==canonical
  protocol=js(line['protocol_file']);front=[json.loads(x) for x in Path(protocol['own_frontend200']).read_text().splitlines()];lengths=[x['inputs']['input_ids']['shape'][1] for x in front];assert len(lengths)==200
  full3=sum(x['lossless_archive_bytes']+x['complete_metadata_bytes'] for x in audits);base=max(x['mandatory_common_payload_headers_bytes']+x['complete_metadata_bytes'] for x in audits);minN=min(x['prompt_length'] for x in audits);forecast=[math.ceil(base*max(1.,(n/minN)**2)) for n in lengths[3:]];need=4*(2*full3+sum(forecast))+6*1024**3;free=shutil.disk_usage(D).free
  review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=False,diagnostic_only=True,actual_storage_and_original_full_numeric_checks_passed=True,no_NEW_NPU_inference=True,no_old_proof_job_or_source_promoted_or_rewritten=True,all_original_3_complete_values_occurrences_and_assertions_retained=True,original_full_numeric_proof=proof,storage_audits=audits,canonical_SHA256_before_and_after=canonical,all_protected_before_and_after_SHA256=protected,capacity_review=dict(empirical_estimate_bytes=need,actual_free_bytes=free,passed=need<=free,all200_input_lengths=lengths,forecast_mandatory_common_bytes=forecast,original_factor=4,original_extra_budget_bytes=6*1024**3,original2048_budget_and_all_hard_log_precheck_sample_guards_unchanged=True,empirical_not_worst_case_guarantee=True,all_complete_alias_metadata_conservatively_included_for_all_samples=True),no_original_file_modified_or_deleted=True,source_SHA256={str(source):sha(source),str(Path(__file__)):sha(__file__),str(R/'universal_eval/utils/lossless_complete_alias_npz_fortran_head_20261117.py'):sha(R/'universal_eval/utils/lossless_complete_alias_npz_fortran_head_20261117.py')})
  save(H/'review.json',review);save(H/'state.json',dict(phase='completed_diagnostic_no_NN_no_new_qualification',started_UTC=started,completed_UTC=review['UTC'],review_SHA256=sha(H/'review.json'),capacity_review=review['capacity_review']));save(H/'actual_completion.json',dict(actual_returncode=0,started_UTC=started,completed_UTC=review['UTC'],full_original_numeric_assertions_passed=True,review_SHA256=sha(H/'review.json'),no_NEW_NPU=True,canonical_unchanged=True));print(json.dumps(dict(actual_returncode=0,review=str(H/'review.json'),review_SHA256=sha(H/'review.json'),capacity_review=review['capacity_review'],storage_audits=audits,no_NEW_NPU=True),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():save(H/'actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc(),no_NEW_NPU=True,no_original_file_modified=True))
  raise

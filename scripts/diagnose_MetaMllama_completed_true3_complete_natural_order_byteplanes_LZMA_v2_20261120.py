"""New Meta storage diagnostic only; complete every value, no NN or report edits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import contextlib,fcntl,hashlib,json,os,shutil,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MetaMllama_completed_true3_complete_natural_order_byteplanes_LZMA_storage_v2_diagnostic_only_20261120';FORMAL=B/'MetaMllama_completed_genuine3_formal_NF3_exact_original_instruction_no_NN_replay_capacity_serial_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main():
 import numpy as np,torch
 from scripts.egm8b_original_independent_cpu_910b import decode_arrays,tr
 from universal_eval.utils.complete_native_raw_bytes_natural_order_byteplanes_LZMA_v2_20261120 import write_complete_native_raw_npz,verify_complete_native_archive
 from scripts.meta_mllama_complete_shape_storage_capacity_20261120 import estimate
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 H.mkdir(exist_ok=False);torch.set_num_threads(4);assert quiet() and not runner_lock_owners();lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock')
 try:
  assert sha(FORMAL/'formal_acceptance.json')=='e55ce8c8080c1a8696bb58725631c6b8635fb1eb01352fbafebec2f87c8c361c';a=js(FORMAL/'formal_acceptance.json');accepted,=a['accepted'];assert sha(accepted['proof'])=='eb85a1130a28cc99059da75a17b6110d0a3e7fa669dbe8dc51a694c1b6826d48';pf=js(accepted['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['actual_native_forwards']==11
  report=Path(accepted['report_dir']);original5={p.name:sha(p) for p in report.iterdir()};assert original5==a['all_original_five_report_SHA256'];line=js(report/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);assert sha(line['protocol_file'])==line['protocol_sha256'];ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canon={n:sha(n) for n in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256'];traces=[json.loads(s) for s in Path(line['per_sample_evidence']).read_text().splitlines()];assert [r['index'] for r in traces]==[0,1,2]
  reviews=[];parts=[];metadata=0;forwards=[]
  for entry in traces:
   assert shutil.disk_usage(R).free>=max(3*2**30,2*2**30*10+2**30);i=entry['index'];q=Path(entry['group_execution']);g=js(q);assert sha(q)==entry['group_execution_sha256'] and g['indices']==[i] and sha(g['archive'])==g['archive_sha256'];src=Path(g['archive']);dst=H/f'own_complete_native_values_{i:05d}.npz';put('progress.json',dict(UTC=now(),phase='complete_natural_order_byteplanes_LZMA',completed_samples=i,total=3,actual_NEW_model_forwards=0))
   values=decode_arrays(src,g['arrays']);write_complete_native_raw_npz(dst,values);complete=verify_complete_native_archive(dst,values);independent=decode_arrays(dst,g['arrays']);assert set(independent)==set(values)
   for n,t in independent.items():assert tr(t)==g['arrays'][n] and torch.equal(t,values[n]),(i,n)
   with zipfile.ZipFile(dst) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(values);compressed=0
    for zi in z.infolist():
     assert zi.compress_type==zipfile.ZIP_LZMA and zi.filename.endswith('.npy');name=zi.filename[:-4]
     with z.open(zi) as f:
      ver=np.lib.format.read_magic(f);shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if ver==(1,0) else np.lib.format.read_array_header_2_0(f))
     assert dtype==np.dtype('uint8') and list(shape)==complete[name]['stored_complete_uint8_shape'];parts.append(dict(index=i,name=zi.filename,original_tensor_shape=g['arrays'][name]['shape'],original_tensor_dtype=g['arrays'][name]['dtype'],stored_NPY_shape=list(shape),stored_NPY_dtype=dtype.str,fortran_order=fortran,compressed_bytes=zi.compress_size,uncompressed_complete_NPY_bytes=zi.file_size,all_original_raw_bits_SHA256=g['arrays'][name]['sha256']));compressed+=zi.compress_size
   metadata+=dst.stat().st_size-compressed+q.stat().st_size;forwards.append(len(g['steps']));reviews.append(dict(index=i,source_archive=str(src),source_archive_SHA256=g['archive_sha256'],new_complete_archive=str(dst),new_archive_SHA256=sha(dst),old_bytes=src.stat().st_size,new_bytes=dst.stat().st_size,all_names_and_full_values_verified=complete,ordinary_original_independent_decoder_accepts_unchanged=True,logical_native_dtype_shape_and_every_raw_bit_identical=True));put('progress.json',dict(UTC=now(),phase='complete_copy_and_original_decoder_verified',completed_samples=i+1,total=3,actual_NEW_model_forwards=0));del values,independent
  cap=estimate(parts,p['input_records'],forwards,metadata);cap.update(UTC=now(),actual_free_bytes=shutil.disk_usage(R).free,capacity_passed=False);cap['capacity_passed']=cap['actual_free_bytes']>=cap['empirical_estimate_bytes'];put('all200_complete_storage_capacity_review.json',cap)
  assert {p.name:sha(p) for p in report.iterdir()}==original5 and {n:sha(n) for n in canon}==canon and sha(accepted['proof'])==accepted['proof_sha256']
  review=dict(UTC=now(),passed=True,diagnostic_only=True,actual_NEW_model_forwards=0,new_fullNF_replay_performed=False,own_genuine_true3_proof=accepted['proof'],own_genuine_true3_proof_SHA256=accepted['proof_sha256'],all_three_complete_archive_reviews=reviews,old_total_bytes=sum(x['old_bytes'] for x in reviews),new_total_bytes=sum(x['new_bytes'] for x in reviews),all_original_five_and_source_and_canonical_unchanged=True,all_original_keys_full_native_tensor_values_dtype_shape_raw_SHA_and_original_decoder_unchanged=True,no_sampling_alias_approximation_math_weight_budget_or_guard_change=True,stored_NPY_byte_axes_and_Fortran_serialization_only=True,capacity_review=str(H/'all200_complete_storage_capacity_review.json'),capacity_review_SHA256=sha(H/'all200_complete_storage_capacity_review.json'),capacity_passed=cap['capacity_passed'],empirical_estimate_bytes=cap['empirical_estimate_bytes'],no_first200_approval_NEWfresh3_200_or_G=True)
  put('review.json',review);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,diagnostic_only=True,actual_NEW_model_forwards=0,capacity_passed=cap['capacity_passed']));return {k:v for k,v in review.items() if k!='all_three_complete_archive_reviews'}
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:
  with contextlib.redirect_stdout(sys.stderr):r=main()
  print(json.dumps(r,ensure_ascii=False))
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0,no_original_values_or_reports_changed=True))
  raise

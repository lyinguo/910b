"""Read existing complete standard LZMA3; prove every tensor's forecast axes."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,shutil,zipfile,traceback
import numpy as np
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'
OLD=B/'GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_actual_D965_1653_20261118'
S=B/'GLM46Flash_completed_true3_complete_standard_Fortran_LZMA_storage_capacity_diagnostic_only_1714_20261118'
H=B/'GLM46Flash_completed_true3_existing_full_LZMA_all200_component_shape_capacity_review_v3_actual_one_based_20261119'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 from glm46_complete_shape_component_capacity_v3_actual_one_based_20261119 import estimate_complete
 H.mkdir(exist_ok=False);assert sha(S/'review.json')=='a0ec96ab83110aef79d6c5327fb86e8ed05ebf80656dc61271afd8e8bf9e88dc'
 storage=js(S/'review.json');assert storage['all_complete_original_names_shapes_dtypes_every_raw_bit_equal'] and storage['all_original_reports_source_old_proof_and_arrays_unmodified']
 done=js(J/'completion.json');a,=done['accepted'];assert done['actual_returncode']==0 and a['proof_sha256']==sha(a['proof'])=='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950'
 proof=js(a['proof']);assert proof['sample_count']==3 and proof['passed'] and proof['actual_forward_count']==586
 folder=Path(a['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution'];trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert len(trace)==3
 old=js(OLD/'full_shape_complete_numeric_storage_capacity_review.json');parts=[];overhead=0;forwards=[];bound={str(S/'review.json'):sha(S/'review.json'),a['proof']:sha(a['proof'])}
 for row,review_file in zip(trace,storage['complete_per_array_value_reviews']):
  e=js(row['actual_execution_file']);i=e['index'];forwards.append(e['actual_native_forwards']);assert len(e['input_ids'])==old['all200_original_prompt_lengths'][i]
  v=js(review_file);p=Path(v['new_path']);assert sha(p)==v['new_SHA256'];bound[str(p)]=sha(p);bound[review_file]=sha(review_file);count=0
  with zipfile.ZipFile(p) as z:
   assert set(n[:-4] for n in z.namelist())==set(v['all_original_complete_values_SHA256'])
   for info in z.infolist():
    assert info.filename.endswith('.npy')
    with z.open(info) as f:
     version=np.lib.format.read_magic(f)
     shape,order,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
    meta=v['all_original_complete_values_SHA256'][info.filename[:-4]];assert list(shape)==meta['shape'] and dtype.str==meta['dtype']
    parts.append(dict(index=i,name=info.filename,shape=list(shape),dtype=dtype.str,Fortran_order=bool(order),compressed_bytes=info.compress_size,complete_uncompressed_NPY_bytes=info.file_size,archive=str(p),archive_SHA256=v['new_SHA256']));count+=info.compress_size
  overhead+=p.stat().st_size-count+Path(row['actual_execution_file']).stat().st_size
 review=estimate_complete(parts,old['all200_original_prompt_lengths'],old['all200_original_pixel_elements'],forwards,overhead)
 for p,h in bound.items():assert sha(p)==h,p
 review.update(UTC=datetime.now(timezone.utc).isoformat(),passed=True,diagnostic_no_NEW_NPU_or_full_NF=True,actual_free_bytes=shutil.disk_usage(R).free,current_free_fits_estimate=shutil.disk_usage(R).free>=review['empirical_estimate_bytes'],original_global_maximum_ratio_estimate_bytes=old['empirical_estimate_bytes'],existing_standard_LZMA_storage_review_SHA256=sha(S/'review.json'),no_new_matrix_copies_or_model_math_changes=True,all_complete_original_values_still_available_and_SHA_bound=bound,original_true3_proof=a['proof'],original_true3_proof_SHA256=a['proof_sha256'],all_original200_prompt_lengths=old['all200_original_prompt_lengths'],all_original200_pixel_elements=old['all200_original_pixel_elements'],own_original_observed_forwards=forwards,serial_release_and_fresh_current_capacity_checks_required_before_approval=True)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,review_SHA256=sha(H/'review.json'),no_model_inference=True));print(json.dumps({k:v for k,v in review.items() if k not in ['all_complete_ZIP_parts_individually_classified','all_complete_original_values_still_available_and_SHA_bound','all_original200_prompt_lengths','all_original200_pixel_elements']},ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_NN_or_canonical_write=True))
  raise

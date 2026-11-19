"""No NN: prove all old GLM3 raw values survive complete reversible storage.

A concurrently running Huihui's complete reserved budget has priority. This
private CPU copy never owns its runner lock, changes its job, or deletes values.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,os,shutil,traceback,zipfile,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'
OLD=B/'GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_actual_D965_1653_20261118'
CUR=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
H=B/'GLM46Flash_completed3_complete_bitplane_exact_sharing_storage_for_immediate_serial_next_20261119'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def reservation():
 homes=[p for p in (R/'workspace/diagnostics/model_runs').glob('HuihuiGLM46Flash_own_original_BF16_2048_*') if p.name.endswith(('_1565767','_1567344'))]
 actual=sum(p.stat().st_size for h in homes for p in h.rglob('*') if p.is_file())+sum(p.stat().st_size for p in CUR.rglob('*') if p.is_file())
 return max(0,146869927228-actual)
def main():
 from universal_eval.utils.complete_bitplane_shared_npz_20261119 import copy_and_check_complete_values
 from A100_alignment_evidence_runner_20261103 import unchanged
 H.mkdir(exist_ok=False);guard=(H/'diagnostic.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 done=js(J/'completion.json');assert done['actual_returncode']==done['model_returncode']==done['verifier_returncode']==0
 a,=done['accepted'];assert a['proof_sha256']==sha(a['proof'])=='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950'
 proof=js(a['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==586 and proof['actual_NPU_inference']
 folder=Path(a['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];assert len(trace)==3
 oldcap=js(OLD/'full_shape_complete_numeric_storage_capacity_review.json');assert len(oldcap['all200_original_prompt_lengths'])==len(oldcap['all200_original_pixel_elements'])==200
 protected={str(folder/n):sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']};protected.update({str(J/'job.json'):sha(J/'job.json'),str(J/'completion.json'):sha(J/'completion.json'),a['proof']:sha(a['proof'])})
 records=[js(row['actual_execution_file']) for row in trace];raw_bound=0
 for record in records:
  p=Path(record['archive']);assert sha(p)==record['archive_SHA256'];protected[str(p)]=sha(p)
  with zipfile.ZipFile(p) as z:raw_bound+=sum(x.file_size for x in z.infolist())
 free=shutil.disk_usage(R).free;remaining=reservation();needed=math.ceil(raw_bound*1.05)+512*1024**2
 assert free-remaining>=needed,('Keep the live Huihui reserved budget',free,remaining,needed)
 put('initial_private_copy_budget.json',dict(actual_free_bytes=free,live_Huihui_reserved_remaining_bytes=remaining,diagnostic_complete_output_upper_budget=needed,no_original_runner_lock_taken=True,not_a_second_model_or_learned_CPU_run=True))
 results=[]
 for row,record in zip(trace,records):
  source=Path(record['archive']);dest=H/source.name
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='copying_and_independently_decoding_all_original_raw_bits',completed=len(results),next_index=record['index'],no_NEW_NPU_or_full_NF=True))
  rec=copy_and_check_complete_values(source,dest);rec.update(source=str(source),source_SHA256=sha(source),destination=str(dest),destination_SHA256=sha(dest),index=record['index'])
  put('complete_all_array_value_review_%d.json'%record['index'],rec);results.append(rec)
  assert shutil.disk_usage(R).free>=reservation()+512*1024**2,'Live Huihui reserve must remain protected'
 unchanged(protected)
 total=sum(x['archive_bytes']+Path(row['actual_execution_file']).stat().st_size for x,row in zip(results,trace));need=max(math.ceil(total*max(oldcap['ratios'].values())*4)+6*1024**3,int(4.5*1024**3)+20*1024**3);free=shutil.disk_usage(R).free
 review=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,diagnostic_only_no_NEW_model_or_full_NF=True,old_true3_proof=a['proof'],old_true3_proof_SHA256=a['proof_sha256'],all_original_reports_proof_sources_and_arrays_unchanged=True,complete_logical_arrays=sum(x['logical_arrays'] for x in results),complete_physical_full_value_arrays=sum(x['physical_complete_arrays'] for x in results),all_original_names_shapes_dtypes_every_raw_bit_reconstructed=True,original_three_archive_bytes=sum(Path(x['source']).stat().st_size for x in results),new_three_archive_bytes=sum(x['archive_bytes'] for x in results),capacity=dict(empirical_estimate_bytes=need,actual_free_bytes=free,passed=free>=need,all_original200_prompt_pixel_shape_ratios=oldcap['ratios'],original_factor4_plus6GiB_and_hard3_log1_precheckHalf_sample2GiB_times_actual_batch_unchanged=True,empirical_not_worstcase_guarantee=True),complete_array_reviews=[str(H/('complete_all_array_value_review_%d.json'%x['index'])) for x in results],protected_original_SHA256=protected,no_old_true3_replay_no_approval_no_fresh3_no_200_no_G=True)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,diagnostic_only=True,review_SHA256=sha(H/'review.json')));print(json.dumps({k:v for k,v in review.items() if k not in ['protected_original_SHA256']},ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_inference_or_canonical_write=True))
  raise

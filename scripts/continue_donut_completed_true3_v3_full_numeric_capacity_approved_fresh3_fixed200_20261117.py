"""Own passed real3 -> empirical storage review -> first200 measurement.

Keep old failures and sources. No model, precision, generation or score change.
The historical A100 selected score covers8; final original-scope qualification
remains pending even if this fixed200 numerical measurement passes.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,fcntl,os,shutil,subprocess,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='katanaml-org/invoices-donut-model-v1'
J3=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_explicit_CPU_diagnostic_role_real_FLOCK_20261117'
J=B/'DonutInvoice_own_original_FP16_poscap2048_approved_fresh3_fixed200_v4_real_FLOCK_20261117'
H=B/'DonutInvoice_completed_true3_v3_full_numeric_capacity_approved_fresh3_fixed200_v4_continuation_20261117'
NF='scripts/verify_donut_invoice_own_native_fp16_v5_approved_measurement_full_linear_KV_20261117.py'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config,validate_report
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not J.exists() and sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 assert sha(J3/'job.json')=='60d079550f48419df9d05b1c4d60cf1a4ae08203c672be9bac4b7f33c6d9cea0'
 old=js(J3/'job.json');done=js(J3/'completion.json');assert done['actual_returncode']==0;accepted,=done['accepted'];proof=js(accepted['proof']);assert sha(accepted['proof'])==accepted['proof_sha256']=='1bb3b0976f0ac9bae8a130e611cf1922378f97ef84bbc19b89c4d31910b2faa0'
 assert proof['passed'] and proof['numeric_review_passed'] and proof['NEW_real_NPU_inference'] and proof['sample_count']==3 and proof['actual_forward_count']==453 and proof['all484_own_source_actual_NPU_SHA256_verified'] and proof['all_full57566_head_rows_independently_recomputed'] and proof['all_native_self_and_cross_KV_values_losslessly_reconstructed_and_independently_projected']
 cfg=js(R/old['spec']['config']);validate_report(R,accepted['report_dir'],MID,'image-to-text',3,proof);unchanged(old['protected_canonical_hashes']);batch.check_sources(old['spec'])
 for path,h in proof['external_evidence_sha256'].items():assert sha(path)==h,path
 p0=js(cfg['model']['kwargs']['protocol_file']);front=[json.loads(s) for s in Path(p0['own_frontend200']).read_text().splitlines()];assert len(front)==200
 lin=js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution'];trace=[json.loads(s) for s in Path(lin['sample_trace']).read_text().splitlines()];assert len(trace)==3
 # Reuse the established 4x observed full3 / 3 * 200 + 6GiB empirical
 # budget. Own fixed pixels and raw-prefix shape are equal for all200.
 # It is not a 1528-token worst-case guarantee; hard/log/sample guards stay.
 assert len({json.dumps(f['original_FP16_pixels']['shape']) for f in front})==1
 assert len({len(f['original_prompt_decoder_ids']) for f in front})==1
 archives=[];parts=[]
 for e in trace:
  a=Path(e['archive']);assert sha(a)==e['archive_SHA256']
  with zipfile.ZipFile(a) as z:
   pp=[dict(name=i.filename,uncompressed_bytes=i.file_size,compressed_bytes=i.compress_size) for i in z.infolist()];assert pp and all(x['name'].endswith('.npy') for x in pp)
  archives.append(dict(path=str(a),SHA256=sha(a),actual_bytes=a.stat().st_size,full_ZIP_uncompressed_bytes=sum(x['uncompressed_bytes'] for x in pp),actual_forward_count=e['actual_native_forwards']))
  parts.extend(dict(archive=str(a),**x) for x in pp)
 actual3=sum(x['full_ZIP_uncompressed_bytes'] for x in archives);estimate=(actual3*200*4+2)//3+6*1024**3;free=shutil.disk_usage(D).free
 cap=dict(passed=True,UTC=now(),capacity_passed=free>=estimate,model_id=MID,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],all_ZIP_part_count=len(parts),all_ZIP_parts=parts,actual_three_archives=archives,all200_fixed_pixel_shape_and_raw_decoder_prefix_shape_verified=True,empirical_estimate_bytes=estimate,actual_free_bytes=free,formula='ceil(sum(all three full ZIP uncompressed bytes) * 200/3 * 4) + 6GiB',established_factor4_plus6GiB_empirical_policy_not_worst_case_guarantee=True,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,preflight_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3,no_matrix_deleted=True)
 put('complete_ZIP_and_all200_shape_empirical_capacity_review.json',cap);assert cap['capacity_passed'],('Actual project disk below unchanged empirical budget',free,estimate)
 assert batch.device_idle(old['spec']);approval=dict(approved=True,UTC=now(),model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'complete_ZIP_and_all200_shape_empirical_capacity_review.json'),capacity_review_SHA256=sha(H/'complete_ZIP_and_all200_shape_empirical_capacity_review.json'),small3_actual_metrics=proof['metrics'],first200_is_measurement_not_predetermined_5pct_qualification=True,no_small_sample_5pct_extra_gate=True,original_A100_reference_samples=8,original_A1008_vs_fixed200_final_same_range_qualification_pending=True,all_original_model_dtype_math_generation_score_and_guards_preserved=True,old_failed_run_not_accepted_or_rewritten=True,old_passed_true3_not_repeated=True);put('first200_measurement_approval.json',approval)
 p=copy.deepcopy(p0);p['evidence_job']=str(J/'job.json');p['first200_measurement_approval']=str(H/'first200_measurement_approval.json');p['approved_true3_proof']=accepted['proof'];p['approved_true3_proof_SHA256']=accepted['proof_sha256'];p['first200_measurement_approved']=True
 p['implementation_source_SHA256'].update({NF:sha(R/NF),str(Path(__file__).resolve().relative_to(R)):sha(__file__)})
 for path in [Path(accepted['proof']),J3/'completion.json',J3/'job.json',H/'first200_measurement_approval.json',H/'complete_ZIP_and_all200_shape_empirical_capacity_review.json']:p['static_evidence_SHA256'][str(path)]=sha(path)
 for n,h in proof['report_sha256'].items():a=Path(accepted['report_dir'])/n;assert sha(a)==h;p['static_evidence_SHA256'][str(a)]=h
 metadata={'evidence_job','first200_measurement_approval','approved_true3_proof','approved_true3_proof_SHA256','first200_measurement_approved','implementation_source_SHA256','static_evidence_SHA256'}
 assert {k:v for k,v in p.items() if k not in metadata}=={k:v for k,v in p0.items() if k not in metadata}
 put('exact_own_model_runtime_protocol_identity_review.json',dict(passed=True,UTC=now(),all_nonstatic_model_input_dtype_math_generation_score_and_guards_exactly_identical=True,only_approval_stage_source_bindings_and_evidence_job_changed=True))
 pp=R/'configs/protocols/donut_invoice_own_original_auto_FP16_poscap_approved_first200_v4_20261117.json';create(pp,p)
 cf=R/'configs/VisionEncoderDecoderModel/katanaml-org--invoices-donut-model-v1-own-original-auto-FP16-poscap2048-native-v4-approved-first200-20261117.yaml';cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 new=copy.deepcopy(old);new['created_at']=now();new['stages']=[3,200];new['spec'].update(approved=True,config=str(cf.relative_to(R)),verifier=NF,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));new['spec']['source_sha256'].update({NF:sha(R/NF),str(Path(__file__).resolve().relative_to(R)):sha(__file__),str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)})
 assert all(new['spec'][k]==old['spec'][k] for k in ['smoke_timeout_seconds','formal_timeout_seconds','verify_timeout_seconds']);batch.check_sources(new['spec']);scope(new['spec'],[3,200],False)
 entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,MID)
 unchanged(old['protected_canonical_hashes']);create(J/'job.json',new)
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=ENV);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=child.pid));rc=child.wait()
 final=js(J/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_job_completion=final,no_G_or_canonical_write=True));assert rc==final['actual_returncode']==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise

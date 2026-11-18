"""Inspect exact standard-NPY layout on accepted old3; never replay NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,zipfile,math,shutil,fcntl,traceback,importlib,contextlib
import numpy as np
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import copy_and_check_all_values
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from measure_OvisU1_completed_true3_actual_ZIP_component_capacity_no_NN_1916_20261117 import components
J=B/'OvisU1_own_original_BF16_512_true3_v2_actual_empty_precision_guard_real_FLOCK_20261116';H=B/'OvisU1_completed3_complete_standard_LZMA_NPZ_storage_diagnostic_only_20261118'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def main():
 os.chdir('/root');assert not H.exists() and not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];H.mkdir();canonical={str(D/n):sha(D/n) for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','PAUSE','POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']}
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);done=js(J/'completion.json');assert done['actual_returncode']==0;a,=done['accepted'];assert sha(a['proof'])==a['proof_sha256']=='221e4c3060bfce23733937f1679a7f89b0e58f0be148650738b99b0da016f39d';folder=Path(a['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];assert len(trace)==3 and len(front)==200;old={str(folder/n):sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']};old[a['proof']]=sha(a['proof']);audits=[];ratios={};overhead=0;redirect={};hasmask=False;maskitems={}
  raw_control=0
  for control in trace:
   with zipfile.ZipFile(js(control['actual_execution_file'])['archive']) as z:raw_control+=sum(v.file_size for v in z.infolist())
  assert shutil.disk_usage(D).free>3*1024**3+1024**3+512*1024**2+2*raw_control,'Full control-copy reserve; no original evidence deletion'
  for row in trace:
   e=js(row['actual_execution_file']);src=Path(e['archive']);assert e['actual_native_forwards']==512 and sha(src)==e['archive_SHA256'];old[str(src)]=sha(src);dst=H/src.name;put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='copying_only_complete_old_small_sample_values',completed_copies=len(audits),current_sample=e['index'],actual_NEW_NPU_forwards=0));m=copy_and_check_all_values(src,dst);(H/(src.stem+'.complete_value_bit_audit.json')).write_bytes(json.dumps(m,ensure_ascii=False).encode());seen={}
   with zipfile.ZipFile(dst) as z:
    for rec in z.infolist():
     with z.open(rec) as f:ver=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format.read_array_header_1_0(f) if ver==(1,0) else np.lib.format.read_array_header_2_0(f)
     key=rec.filename[:-4];raw=int(np.prod(shape))*dtype.itemsize;seen[key]=(shape,dtype.itemsize);ratios[key]=max(ratios.get(key,0),rec.compress_size/raw)
     if key.startswith('attention_mask_'):hasmask=True;maskitems[key]=dtype.itemsize
    overhead=max(overhead,dst.stat().st_size-sum(x.compress_size for x in z.infolist()))
   expected=components(front[e['index']],512,e['index'],hasmask)
   for key,item in maskitems.items():
    if key in expected:expected[key]=(expected[key][0],item)
   assert expected==seen;redirect[str(src.resolve())]=dst;audits.append(dict(index=e['index'],archive_original=str(src),original_SHA256=sha(src),original_bytes=src.stat().st_size,new_complete_archive=str(dst),new_complete_archive_SHA256=sha(dst),new_bytes=dst.stat().st_size,all_complete_original_arrays_preserved=True,full_raw_bit_audit=str(H/(src.stem+'.complete_value_bit_audit.json'))))
  projected=[]
  for i,f in enumerate(front):
   c=components(f,512,i,hasmask)
   for key,item in maskitems.items():
    if key in c:c[key]=(c[key][0],item)
   assert set(c)<=set(ratios);projected.append(sum(math.ceil(int(np.prod(sh))*item*ratios[n]) for n,(sh,item) in c.items())+overhead)
  need=4*(sum(projected)+sum(x['new_bytes'] for x in audits))+6*1024**3;free=shutil.disk_usage(D).free
  review=dict(UTC=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,passed=False,model_id='AIDC-AI/Ovis-U1-3B',storage_audits=audits,capacity_review=dict(empirical_estimate_bytes=need,actual_free_bytes=free,passed=need<=free,original_factor4_extra6GiB_and_all_runtime_guards_preserved=True,own_original512_and_EOSNone_preserved=True,all200_own_original_shapes_checked=True,all200_projected_complete_archives=projected,empirical_not_worst_case_guarantee=True),storage_format='Standard NPZ/NPY full arrays with ZIP_LZMA; exact full dtype/shape/raw values unchanged',full_original_numeric_recheck=None,no_NEW_NPU=True,no_original_value_report_proof_or_source_mutation=True,all_complete_old_values_and_semantic_occurrences_preserved=True)
  put('storage_review_before_optional_full_numeric_recheck.json',review)
  if need<=free:
   name='verify_ovisu1_own_original_understanding_bf16_512_20261116';module=importlib.import_module(name);original=module.ReadOnlyNPZ
   def redirected(path,*args,**kwargs):return original(redirect.get(str(Path(path).resolve()),path),*args,**kwargs)
   module.ReadOnlyNPZ=redirected;put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='independent_original_full_numeric_checks_against_all_exact_new_small3_copies_no_NN',actual_NEW_NPU_forwards=0))
   try:
    with contextlib.redirect_stdout(sys.stderr):proof=module.verify(folder,3)
   finally:module.ReadOnlyNPZ=original
   assert proof['passed'] and proof['actual_forward_count']==1536;review['full_original_numeric_recheck']=proof
  for n,h in old.items():assert sha(n)==h
  for n,h in canonical.items():assert sha(n)==h
  review['original_reports_values_proofs_and_canonical_SHA256_verified_unchanged']=True;put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,no_NEW_NPU=True,all_complete_values_preserved=True,full_original_numeric_recheck_performed=review['full_original_numeric_recheck'] is not None,capacity_passed=need<=free,review_SHA256=sha(H/'review.json')));print(json.dumps(dict(actual_returncode=0,capacity_review=review['capacity_review'],storage_audits=audits,full_numeric_recheck_performed=review['full_original_numeric_recheck'] is not None,no_NEW_NPU=True),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NEW_NPU=True,old_evidence_unmodified=True))
  raise

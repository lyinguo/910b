import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent
utility='''"""Lossless sharing of identical complete NPY values; never drops an occurrence."""
from pathlib import Path
import hashlib,json
import numpy as np

def array_key(a):
 a=np.ascontiguousarray(a)
 return (a.dtype.str,tuple(a.shape),hashlib.sha256(a.view(np.uint8).reshape(-1).tobytes()).hexdigest())

class AliasArchive:
 def __init__(self,archive,aliases,loader):
  self._z=loader(archive,allow_pickle=False);self.aliases=dict(aliases);self.files=list(aliases)
  assert set(self.aliases.values())==set(self._z.files)
 def __getitem__(self,key):return self._z[self.aliases[key]]
 def __contains__(self,key):return key in self.aliases
 def __iter__(self):return iter(self.files)
 def keys(self):return self.files
 def close(self):self._z.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()

def copy_with_exact_aliases(source,destination):
 source=Path(source);destination=Path(destination);assert source!=destination and not destination.exists()
 unique={};lookup={};aliases={};records={};original_loader=np.load
 with original_loader(source,allow_pickle=False) as z:
  for name in z.files:
   a=z[name];key=array_key(a)
   if key in lookup:
    target=lookup[key];assert np.array_equal(unique[target].view(np.uint8).reshape(-1),np.ascontiguousarray(a).view(np.uint8).reshape(-1))
   else:target=name;lookup[key]=name;unique[name]=a.copy()
   aliases[name]=target;records[name]=dict(dtype=a.dtype.str,shape=list(a.shape),raw_value_SHA256=key[2],nbytes=a.nbytes)
 np.savez_compressed(destination,**unique)
 with original_loader(source,allow_pickle=False) as original,AliasArchive(destination,aliases,original_loader) as decoded:
  assert set(original.files)==set(decoded.files)
  for name in original.files:
   a=original[name];b=decoded[name];assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a.view(np.uint8).reshape(-1),b.view(np.uint8).reshape(-1))
 return dict(all_logical_arrays_bitwise_reconstructed=True,original_occurrences=len(aliases),physical_unique_arrays=len(unique),aliases=aliases,all_original_values=records,logical_nbytes=sum(v['nbytes'] for v in records.values()),unique_nbytes=sum(a.nbytes for a in unique.values()),no_original_file_modified_or_deleted=True)
'''
worker='''"""Only the observer storage changes; every original model method stays inherited."""
from pathlib import Path
import numpy as np
import torch
from universal_eval.model.merve_pali_full_native_value_20261117.worker import OwnWorkerCapture,inspect_own_sources,meta,jwrite,sha

class LosslessSharedOwnWorkerCapture(OwnWorkerCapture):
 def begin(self,group):
  self._same_values={};self._capture_occurrences=0;self._logical_nbytes=0;self._bitwise_reuses=0
  return super().begin(group)
 def save(self,t,label):
  assert torch.is_tensor(t);a=t.detach().cpu().contiguous();m=meta(a);data=a.view(torch.uint16).numpy() if a.dtype==torch.bfloat16 else a.numpy();key=(m['dtype'],tuple(m['shape']),m['SHA256']);self._capture_occurrences+=1;self._logical_nbytes+=data.nbytes
  if key in self._same_values:
   name=self._same_values[key];old=self.arrays[name];assert old.dtype==data.dtype and old.shape==data.shape and np.array_equal(old.view(np.uint8).reshape(-1),data.view(np.uint8).reshape(-1));self._bitwise_reuses+=1
  else:
   name='a'+str(self.seq).zfill(8)+'_'+label.replace('.','_');self.seq+=1;self.arrays[name]=data.copy();self._same_values[key]=name
  return dict(array=name,**m)
 def finish(self):
  storage=dict(schema='complete_tensor_occurrences_exact_value_sharing_v1',all_actual_tensor_occurrences=self._capture_occurrences,physical_unique_arrays=len(self.arrays),logical_unshared_raw_nbytes=self._logical_nbytes,physical_unique_raw_nbytes=sum(a.nbytes for a in self.arrays.values()),every_duplicate_compared_bitwise_to_actual_current_NPU_value=True,bitwise_duplicate_occurrences=self._bitwise_reuses,no_tensor_or_occurrence_or_numeric_check_dropped=True,no_precision_or_model_math_or_sampling_change=True,storage_source=__file__,storage_source_SHA256=sha(__file__))
  result=super().finish();p=Path(result['path']);import json;g=json.loads(p.read_bytes());g['lossless_shared_tensor_storage']=storage;jwrite(p,g);result['SHA256']=sha(p);result['lossless_shared_tensor_storage']=storage;self._same_values={};return result

def install_capture(worker,root,headers,home):
 assert not hasattr(worker,'_own_pali10pt_full_capture');assert len(headers)==603
 inspection=inspect_own_sources(worker,root,headers,home);worker._own_pali10pt_full_capture=LosslessSharedOwnWorkerCapture(worker,root,headers,home)
 return dict(installed=True,own_source_inspection=inspection,actual_model_implementation_unchanged=True,actual_model_forward_not_executed_by_install=True)

def begin_capture(worker,group):
 worker._own_pali10pt_full_capture.begin(group);return dict(started=True,group_index=group)
def finish_capture(worker):return worker._own_pali10pt_full_capture.finish()
'''
diagnostic='''"""Full old numeric3 checks read exact aliases; no new NN or old-job promotion."""
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
'''
payload={}
for n,s in [('universal_eval/utils/lossless_npz_aliases.py',utility),('universal_eval/model/merve_pali_full_native_value_20261117/worker_lossless_v2.py',worker),('scripts/diagnostic_merve_completed_numeric3_lossless_exact_tensor_sharing_v1_20261117.py',diagnostic)]:
 ast.parse(s);(P/Path(n).name).write_bytes(s.encode());payload[n]=dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;I=B/'Merve_lossless_full_numeric3_diagnostic_source_and_actual_launch_v1_20261117';I.mkdir(exist_ok=False)
for n,v in FILES.items():
 p=R/n;raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
source=R/'scripts/diagnostic_merve_completed_numeric3_lossless_exact_tensor_sharing_v1_20261117.py';runner=I/'supervisor.py'
supervisor=chr(10).join(['import sys','sys.dont_write_bytecode=True','from pathlib import Path','from datetime import datetime,timezone','import json,subprocess,os','I=Path('+repr(str(I))+');R=Path('+repr(str(R))+')',"with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',"+repr(str(source))+"],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))","v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,no_NEW_NPU_inference=True);(I/'actual_completion.json').write_bytes(json.dumps(v,indent=2).encode())",'raise SystemExit(p.returncode)','']);ast.parse(supervisor);runner.write_bytes(supervisor.encode())
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(runner)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,diagnostic_only_no_NEW_NPU=True,source_SHA256={n:v['SHA256'] for n,v in FILES.items()},all_original_model_math_reports_and_arrays_untouched=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload));ast.parse(remote);(P/'deploy_run_Merve_lossless_full_numeric3_diagnostic_v1_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:v['SHA256'] for n,v in payload.items()}))

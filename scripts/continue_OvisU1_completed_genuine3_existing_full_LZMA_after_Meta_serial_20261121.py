"""Serial own Ovis continuation: retain genuine3, validate existing complete storage, then fresh3/200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import ast,base64,contextlib,copy,fcntl,hashlib,importlib,json,os,shutil,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='AIDC-AI/Ovis-U1-3B'
SELF='scripts/continue_OvisU1_completed_genuine3_existing_full_LZMA_after_Meta_serial_20261121.py'
OLDMODEL='universal_eval/model/adapters/ovisu1_own_original_understanding_bf16_512_native_v2_actual_empty_precision_guard_20261116.py'
MODEL='universal_eval/model/adapters/ovisu1_own_original_understanding_BF16_512_complete_standard_LZMA_20261121.py'
UTILITY='universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py'
NF='scripts/verify_ovisu1_own_original_understanding_bf16_512_20261116.py'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
J3=B/'OvisU1_own_original_BF16_512_true3_v2_actual_empty_precision_guard_real_FLOCK_20261116'
DIAG=B/'OvisU1_completed3_complete_standard_LZMA_NPZ_storage_diagnostic_only_20261118'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def model_AST_unchanged():
 old=ast.parse((R/OLDMODEL).read_text());new=ast.parse((R/MODEL).read_text())
 def mathematical(t):return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='save'}
 assert mathematical(old)==mathematical(new)
 oldimports=[ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,(ast.Import,ast.ImportFrom))]
 newimports=[ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.ImportFrom) and n.module=='universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118')]
 assert oldimports==newimports
def active_exact_storage_maintenance():
 names={'periodic_batch_910b_exact_storage_maintenance_v3_immediate_budget_20261120.py','discover_exact_byte_identical_completed_project_NPZ_storage_reusable_v2_20261120.py','apply_exact_byte_identical_project_matrix_physical_storage_20261120.py'}
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
   if any(Path(a).name in names for a in args):found.append(int(p.name))
  except FileNotFoundError:pass
 return found
def reference_alias(q,oldproto):
 from batch_910b_support import workbook_rows
 book=Path(oldproto['reference_workbook']);backup=Path(q['old170_exact_book_backup'])
 assert sha(backup)==oldproto['reference_workbook_SHA256']==q['old170_book_SHA256']=='195e9af1a5284c764c795516853256bca54e9880ddd7b388cfefaa9a3c31b496'
 assert sha(book)==q['expected_protected_canonical_hashes'][str(book)]=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
 old,_,_=workbook_rows(backup);new,_,_=workbook_rows(book);assert len(old)==len(new)==700
 assert all(a['row']==b['row'] and a['model_id']==b['model_id'] and a['task']==b['task'] for a,b in zip(old,new))
 assert sum(v['score'] is not None for v in old)==170 and sum(v['score'] is not None for v in new)==183
 assert all(a['score'] is None or Decimal(str(a['score']))==Decimal(str(b['score'])) for a,b in zip(old,new))
 row,=[v for v in new if v['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 import zipfile,xml.etree.ElementTree as ET
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 def dmap(p):
  with zipfile.ZipFile(p) as z:
   root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
   return {c.attrib['r']:ET.tostring(c) for c in root.findall('.//s:c',ns) if c.attrib['r'].startswith('D')}
 assert dmap(backup)==dmap(book)
 with zipfile.ZipFile(book) as z:
  root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));d=root.find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert Decimal(d)==Decimal('.875') and all(Decimal(v)==Decimal('.875') for v in oldproto['current_A100_D_values'].values())
 assert q['fresh_local_book_SHA256']==sha(book)
 return dict(passed=True,old170_exact_book_backup=str(backup),old170_book_SHA256=sha(backup),current183_book=str(book),current183_book_SHA256=sha(book),all700_C_B_row_bindings_and_entire_D_column_unchanged=True,all170_prior_G_values_preserved=True,current_row=row['row'],three_D_values=oldproto['current_A100_D_values'],own_original_G_still_blank=True,only_exact_historical_book_check_redirected_to_immutable_identical_bytes=True)
def storage_acceptance(H,q,alias,parent):
 accepted,=js(J3/'completion.json')['accepted'];assert accepted['sample_count']==3 and sha(accepted['proof'])==accepted['proof_sha256']=='221e4c3060bfce23733937f1679a7f89b0e58f0be148650738b99b0da016f39d'
 oldproof=js(accepted['proof']);assert oldproof['passed'] and oldproof['actual_forward_count']==1536 and oldproof['source_tensor_count']==1060 and oldproof['active_physical_source_tensor_count']==487
 review=js(DIAG/'review.json');assert sha(DIAG/'review.json')==q['storage_review_SHA256'] and review['no_NEW_NPU'] and review['all_complete_old_values_and_semantic_occurrences_preserved'] and review['no_original_value_report_proof_or_source_mutation']
 assert review['full_original_numeric_recheck'] is None and review['capacity_review']['empirical_estimate_bytes']==160797505832
 redir={}
 for r in review['storage_audits']:
  assert r['all_complete_original_arrays_preserved'] and sha(r['archive_original'])==r['original_SHA256'] and sha(r['new_complete_archive'])==r['new_complete_archive_SHA256']
  redir[str(Path(r['archive_original']).resolve())]=Path(r['new_complete_archive'])
 assert len(redir)==3
 if q.get('reuse_completed_storage_acceptance'):
  rec=q['reuse_completed_storage_acceptance'];assert sha(rec['path'])==rec['SHA256'];v=js(rec['path']);assert v['passed'] and v['actual_NEW_model_forwards']==0 and v['original_genuine3_proof_SHA256']==accepted['proof_sha256'] and v['all1060_own_original_source_and_full1536_steps_numerically_verified']
  assert sha(v['complete_storage_full_NF_proof'])==v['complete_storage_full_NF_proof_SHA256'];return v
 module=importlib.import_module('verify_ovisu1_own_original_understanding_bf16_512_20261116')
 original_loader=module.ReadOnlyNPZ;original_check=module.check
 def exact_book_check(p,h=None):
  if str(Path(p))==alias['current183_book'] and h==alias['old170_book_SHA256']:return original_check(alias['old170_exact_book_backup'],h)
  return original_check(p,h)
 def existing_full_LZMA_values(p,*a,**kw):return original_loader(redir.get(str(Path(p).resolve()),p),*a,**kw)
 module.check=exact_book_check;module.ReadOnlyNPZ=existing_full_LZMA_values
 try:
  with (H/'complete_existing_storage_fullNF_stderr.log').open('x') as err,contextlib.redirect_stdout(err),contextlib.redirect_stderr(err):
   pf=module.verify(Path(accepted['report_dir']),3)
 finally:module.ReadOnlyNPZ=original_loader;module.check=original_check
 assert pf['passed'] and pf['actual_forward_count']==1536 and pf['source_tensor_count']==1060 and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed']
 assert pf['report_sha256']==oldproof['report_sha256']
 for p,h in oldproof['external_evidence_sha256'].items():
  if p!=alias['current183_book']:assert pf['external_evidence_sha256'].get(p)==h,(p,'Original full external check not retained')
 pf['all_complete_new_storage_archives_SHA256']={str(p):sha(p) for p in redir.values()}
 put(H/'complete_existing_storage_fullNF_proof.json',pf)
 v=dict(passed=True,UTC=now(),actual_NEW_model_forwards=0,same_original_genuine3_no_model_replay=True,original_genuine3_proof=accepted['proof'],original_genuine3_proof_SHA256=accepted['proof_sha256'],complete_storage_full_NF_proof=str(H/'complete_existing_storage_fullNF_proof.json'),complete_storage_full_NF_proof_SHA256=sha(H/'complete_existing_storage_fullNF_proof.json'),all1060_own_original_source_and_full1536_steps_numerically_verified=True,all_complete_existing_new_storage_archives_SHA256=pf['all_complete_new_storage_archives_SHA256'],original_D_reference_alias_review=alias,original_five_reports_and_all_old_values_source_and_logs_unchanged=True)
 put(H/'formal_existing_storage_acceptance.json',v);return v
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
 config=Path(sys.argv[1]).resolve();assert config.is_relative_to(R/'configs/queues');q=js(config);H=Path(q['home']);J=Path(q['new_job'])
 assert q['model_id']==MID and q['max_concurrent_models']==1 and not H.exists() and not J.exists()
 H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 unchanged({str(R/n):h for n,h in q['exact_source_SHA256'].items()});model_AST_unchanged()
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 prev=Path(q['previous_job'])
 while not (prev/'completion.json').is_file() or not quiet():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_previous_Meta_actual_NN_NF_and_original_EX_release_not_NN',previous_job=str(prev),actual_NEW_model_forwards=0,max_concurrent_models=1,required_bytes=160797505832,actual_free_bytes=shutil.disk_usage(R).free));time.sleep(20)
 prior=js(prev/'completion.json');put(H/'previous_Meta_actual_completion.json',prior)
 unchanged(q['expected_protected_canonical_hashes']);parent=js(J3/'job.json');batch.check_sources(parent['spec'])
 cfg=js(R/parent['spec']['config']);oldproto=js(cfg['model']['kwargs']['protocol_file']);alias=reference_alias(q,oldproto);put(H/'original170_to_current183_exact_reference_review.json',alias)
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='formal_full_numeric_validation_of_existing_complete_LZMA_original3_no_NN_replay',actual_NEW_model_forwards=0,original_genuine3_forward_count=1536))
  storage=storage_acceptance(H,q,alias,parent);unchanged(q['expected_protected_canonical_hashes'])
 if shutil.disk_usage(R).free<160797505832:
  while active_exact_storage_maintenance():
   put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_existing_exact_storage_maintenance_natural_completion_not_NN',maintenance_PIDs=active_exact_storage_maintenance(),actual_NEW_model_forwards=0));time.sleep(20)
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='immediate_authorized_exact_storage_recovery_no_NN',required_bytes=160797505832,actual_free_bytes=shutil.disk_usage(R).free,actual_NEW_model_forwards=0))
  recovery=R/'scripts/maintenance/periodic_batch_910b_exact_storage_maintenance_v3_immediate_budget_20261120.py'
  assert sha(recovery)=='58755caac54d19f323ad06e42ed2ae4f7b1031dfeccc640299effb19856dc16e'
  with (H/'exact_storage_recovery_stdout.log').open('x') as out,(H/'exact_storage_recovery_stderr.log').open('x') as err:
   result=subprocess.run(['python3','-B',str(recovery),'--apply','--required-free-bytes','160797505832'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
  assert result.returncode==0,'Recovery failed; preserve actual logs and use a new private continuation after correction'
 cap=copy.deepcopy(js(DIAG/'review.json')['capacity_review']);cap.update(UTC=now(),passed=shutil.disk_usage(R).free>=160797505832,actual_free_bytes=shutil.disk_usage(R).free,all_complete_existing_standard_LZMA_values_full_NF_accepted=True,formal_storage_acceptance=str(H/'formal_existing_storage_acceptance.json') if not q.get('reuse_completed_storage_acceptance') else q['reuse_completed_storage_acceptance']['path'],formal_storage_acceptance_SHA256=sha(H/'formal_existing_storage_acceptance.json') if not q.get('reuse_completed_storage_acceptance') else q['reuse_completed_storage_acceptance']['SHA256'],no_new_compression_or_original_true3_NN_replay=True)
 put(H/'capacity_review.json',cap);assert cap['passed'],'Actual original project disk below full200 LZMA budget; keep original values and resume this accepted storage proof without replay'
 unchanged(q['expected_protected_canonical_hashes']);assert quiet()
 approval=dict(UTC=now(),approved=True,model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=storage['original_genuine3_proof'],true3_proof_SHA256=storage['original_genuine3_proof_SHA256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),storage_full_NF_acceptance=storage,all_original_math_dtype_512_EOSNone_outer4_and_all_hard_guards_unchanged=True,small3_not_an_additional_5pct_gate=True)
 put(H/'own_first200_measurement_approval.json',approval)
 p2=copy.deepcopy(oldproto);p2['reference_workbook_SHA256']=alias['current183_book_SHA256']
 exact={**q['exact_source_SHA256'],str(config.relative_to(R)):sha(config)}
 p2['implementation_source_SHA256'].update(exact)
 p2['static_evidence_SHA256'].update({str(H/'original170_to_current183_exact_reference_review.json'):sha(H/'original170_to_current183_exact_reference_review.json'),str(H/'capacity_review.json'):sha(H/'capacity_review.json'),str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')})
 p2['complete_lossless_standard_LZMA_storage']=dict(full_original_values_dtype_shape_and_semantic_names_preserved=True,source=UTILITY,SHA256=sha(R/UTILITY),all_nonstorage_model_AST_unchanged=True,formal_acceptance=storage)
 pp=R/q['new_protocol'];cc=R/q['new_config'];create(pp,p2)
 cfg['name']=cc.stem;cfg['model']['adapter']=MODEL+':OvisU1OwnUnderstandingNative';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cc,cfg)
 spec=copy.deepcopy(parent['spec']);spec['source_sha256'].pop(parent['spec']['config'],None);spec['source_sha256'].update(exact)
 spec.update(approved=True,config=str(cc.relative_to(R)),verifier=NF,alignment_approval_review=str(H/'own_first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'own_first200_measurement_approval.json'))
 spec['source_sha256'].update({str(cc.relative_to(R)):sha(cc),str(pp.relative_to(R)):sha(pp),NF:sha(R/NF)})
 validate_config(cfg,MID);entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:
  path,c=batch.config_for(spec,entry,count);validate_config(c,MID)
  if path.resolve().is_relative_to(R/'configs'):spec['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 scope(spec,[3,200],False);batch.check_sources(spec);unchanged(q['expected_protected_canonical_hashes']);assert quiet() and batch.device_idle(spec) and shutil.disk_usage(R).free>=160797505832
 newjob=copy.deepcopy(parent);newjob.update(spec=spec,stages=[3,200],created_at=now(),protected_canonical_hashes=copy.deepcopy(q['expected_protected_canonical_hashes']),private_stage_type='same accepted genuine3, complete standard LZMA storage fullNF, approved NEWfresh3/fixed200 original Ovis serial max1');create(J/'job.json',newjob)
 with (H/'runner_stdout.log').open('x') as out,(H/'runner_stderr.log').open('x') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='approved_NEWfresh3_then_fixed200_original_Ovis_runner_active',job=str(J),job_SHA256=sha(J/'job.json'),runner_PID=child.pid,max_concurrent_models=1,no_G_or_canonical_write=True));rc=child.wait()
 done=js(J/'completion.json');assert rc==done['actual_returncode'];put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,phase='own_Ovis_NEWfresh3_fixed200_natural_completion',job=str(J),completion=done,no_G_or_canonical_write=True));return rc
if __name__=='__main__':
 H=None
 try:sys.exit(main())
 except SystemExit:raise
 except BaseException:
  try:
   q=js(Path(sys.argv[1]));H=Path(q['home'])
   if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_values_reports_sources_configs_logs_retained=True,no_G_or_canonical_write=True))
  finally:raise

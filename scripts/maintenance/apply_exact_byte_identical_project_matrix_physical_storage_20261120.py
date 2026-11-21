"""Merge identical completed NPZ inodes; retain every original path and byte."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,shutil,stat,tempfile,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';ROOT=R/'workspace/diagnostics/model_runs';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,x):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def journal(p,x):
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def samebytes(a,b):
 with a.open('rb') as x,b.open('rb') as y:
  while True:
   p=x.read(8*2**20);q=y.read(8*2**20)
   if p!=q:return False
   if not p:return True
def checked(p,root):
 assert p.suffix=='.npz' and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),str(p)
 st=p.lstat();assert stat.S_ISREG(st.st_mode);return st
def merge(a,b,digest,root,blocked_inodes):
 sa=checked(a,root);sb=checked(b,root);assert (sa.st_dev,sa.st_uid,sa.st_gid,sa.st_mode)==(sb.st_dev,sb.st_uid,sb.st_gid,sb.st_mode)
 assert sa.st_size==sb.st_size and (sa.st_dev,sa.st_ino) not in blocked_inodes and (sb.st_dev,sb.st_ino) not in blocked_inodes
 assert sha(a)==sha(b)==digest and samebytes(a,b)
 if sa.st_ino==sb.st_ino:return dict(path=str(b),already_same_inode=True,reclaimed_allocated_bytes=0,SHA256=digest)
 assert sb.st_nlink==1,'Preserve unseen/previous multi-link aliases'
 old=dict(inode=sb.st_ino,mtime_ns=sb.st_mtime_ns,atime_ns=sb.st_atime_ns,mode=sb.st_mode,uid=sb.st_uid,gid=sb.st_gid,bytes=sb.st_size,allocated_bytes=sb.st_blocks*512)
 tmp=b.with_name('.'+b.name+'.exact-byte-share-'+str(os.getpid())+'.tmp');assert not tmp.exists()
 os.link(a,tmp,follow_symlinks=False)
 try:
  assert tmp.stat().st_ino==sa.st_ino and sha(tmp)==digest
  current=b.lstat();assert (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)==(sb.st_dev,sb.st_ino,sb.st_size,sb.st_mtime_ns)
  os.replace(tmp,b)
 finally:
  if tmp.exists():tmp.unlink()
 assert b.is_file() and a.stat().st_ino==b.stat().st_ino and sha(b)==digest and b.stat().st_size==old['bytes']
 return dict(path=str(b),canonical_path=str(a),SHA256=digest,previous_stat=old,new_inode=b.stat().st_ino,new_mtime_ns=b.stat().st_mtime_ns,reclaimed_allocated_bytes=old['allocated_bytes'],every_original_path_and_complete_file_bytes_preserved=True,metadata_inode_ctime_and_duplicate_mtime_only_may_change=True)
def tests(home):
 passed=[]
 with tempfile.TemporaryDirectory(prefix='exact_storage_tests_',dir=home) as name:
  root=Path(name);a=root/'a.npz';b=root/'b.npz';a.write_bytes(b'complete-original-bytes\0'*100);b.write_bytes(a.read_bytes());h=sha(a)
  result=merge(a,b,h,root,set());assert result['reclaimed_allocated_bytes']>0 and a.exists() and b.exists();passed.append('same complete bytes retain both paths and SHA')
  assert merge(a,b,h,root,set())['already_same_inode'];passed.append('already linked is idempotent')
  c=root/'c.npz';c.write_bytes(b'wrong-and-different')
  try:merge(a,c,h,root,set());raise RuntimeError('different bytes accepted')
  except AssertionError:assert c.read_bytes()==b'wrong-and-different';passed.append('different bytes rejected without mutation')
  c.unlink();c.symlink_to(a)
  try:merge(a,c,h,root,set());raise RuntimeError('symlink accepted')
  except AssertionError:passed.append('symlink rejected')
  c.unlink();c.write_bytes(a.read_bytes())
  try:merge(a,c,h,root,{(a.stat().st_dev,a.stat().st_ino)});raise RuntimeError('open inode accepted')
  except AssertionError:passed.append('active inode rejected')
  try:merge(a,c,h,root/'not-the-root',set());raise RuntimeError('outside scope accepted')
  except AssertionError:passed.append('out of scope rejected')
 return passed
def fd_inodes():
 result=set()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   for q in (p/'fd').iterdir():
    try:st=q.stat();result.add((st.st_dev,st.st_ino))
    except (FileNotFoundError,PermissionError):pass
  except (FileNotFoundError,PermissionError):pass
 return result
def report_stats():
 return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (R/'workspace/reports').rglob('*') if p.is_file()}
def main(plan_path,expected):
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 assert sha(plan_path)==expected;plan=js(plan_path);H=Path(plan['home']);assert H.resolve().is_relative_to(B) and not H.exists();H.mkdir();put(H/'plan.json',plan);test_results=tests(H);put(H/'tests.json',dict(passed=test_results,count=len(test_results)))
 assert sha(plan['discovery_review'])==plan['discovery_review_SHA256'];discovery=js(plan['discovery_review']);assert discovery['passed'] and discovery['read_only'] and discovery['scope']==str(ROOT)
 policy=js(plan['policy_file']);assert sha(plan['policy_file'])==plan['policy_file_SHA256'] and policy['physical_deduplication_keeps_all_original_paths_complete_bytes_and_SHA'] and policy['max_concurrent_models']==1
 assert quiet() and not runner_lock_owners();lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');put(H/'original_lock.json',dict(UTC=now(),PID=os.getpid(),actual_original_EX=runner_lock_owners(),storage_only_no_NN=True))
 before=shutil.disk_usage(R).free;receipts=[];skipped=[]
 try:
  ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canon={p:sha(p) for p in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256'];before_reports=report_stats();opened=fd_inodes();groups=discovery['exact_duplicate_groups']
  planned={(g['SHA256'],g['canonical_path']) for g in groups};assert all((g['SHA256'],g['canonical_path']) in planned for g in plan['groups'])
  all_paths={r['path']:r for g in plan['groups'] for r in g['all_original_paths']};protected_before={}
  # Capture and recheck every affected full file, never infer byte equality from size.
  for path,row in all_paths.items():
   p=Path(path);st=checked(p,ROOT);assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns']);assert sha(p)==row['SHA256'];protected_before[path]=row['SHA256']
  put(H/'all_original_paths_complete_SHA_before.json',protected_before)
  for idx,g in enumerate(plan['groups']):
   a=Path(g['canonical_path']);sa=checked(a,ROOT)
   if (sa.st_dev,sa.st_ino) in opened:skipped.append(dict(group=g['SHA256'],reason='canonical has active FD'));continue
   for row in g['all_original_paths']:
    b=Path(row['path']);sb=checked(b,ROOT)
    if sb.st_ino==sa.st_ino:continue
    if sb.st_nlink!=1 or (sb.st_dev,sb.st_ino) in opened or (sa.st_uid,sa.st_gid,sa.st_mode)!=(sb.st_uid,sb.st_gid,sb.st_mode):skipped.append(dict(path=str(b),reason='existing aliases, active FD or different ownership/mode'));continue
    journal(H/'exact_file_storage_operations.jsonl',dict(UTC=now(),phase='before_atomic_exact_byte_inode_share',path=str(b),canonical_path=str(a),SHA256=g['SHA256'],previous_stat=row))
    rec=merge(a,b,g['SHA256'],ROOT,opened);receipts.append(rec);journal(H/'exact_file_storage_operations.jsonl',dict(UTC=now(),phase='after_full_SHA_verified',receipt=rec))
   put(H/'progress.json',dict(UTC=now(),phase='merging_only_exact_complete_file_duplicates_all_original_paths_retained',groups_completed=idx+1,groups_total=len(plan['groups']),files_shared=len(receipts),physical_bytes_reclaimed=sum(x['reclaimed_allocated_bytes'] for x in receipts),actual_NEW_model_forwards=0))
  after={p:sha(p) for p in all_paths};assert after==protected_before;assert report_stats()==before_reports and {p:sha(p) for p in canon}==canon
  put(H/'all_original_paths_complete_SHA_after.json',after);put(H/'per_file_exact_storage_receipts.json',dict(UTC=now(),files=receipts,skipped=skipped,every_original_matrix_file_and_byte_still_present=True,no_numerical_array_pruned=True,source_config_reports_logs_original_retention_ledger_and_canonical_unchanged=True,previous_duplicate_mtime_metadata_preserved_in_receipts=True))
  result=dict(UTC=now(),actual_returncode=0,files_physically_shared=len(receipts),groups_checked=len(plan['groups']),all_original_matrix_paths_protected=len(all_paths),all_original_complete_file_SHA_before_after_equal=True,all_complete_reports_stat_before_after_equal=True,original_retention_ledger_and_all183_canonical_unchanged=True,no_report_source_config_log_weight_task_data_or_unique_matrix_file_deleted=True,no_numerical_array_pruned=True,physical_bytes_reclaimed=sum(x['reclaimed_allocated_bytes'] for x in receipts),actual_free_before_bytes=before,actual_free_after_bytes=shutil.disk_usage(R).free,tests_passed=len(test_results),max_concurrent_models=1,storage_only_not_NN_or_fullNF=True)
  put(H/'actual_completion.json',result);print(json.dumps(result,ensure_ascii=False))
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main(Path(sys.argv[1]),sys.argv[2])
 except BaseException:
  if len(sys.argv)>2:
   try:
    p=Path(js(sys.argv[1])['home'])
    if p.resolve().is_relative_to(B) and p.exists() and not (p/'actual_completion.json').exists():put(p/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),partial_operation_journal=str(p/'exact_file_storage_operations.jsonl'),all_original_file_payloads_retained=True,no_unique_array_deleted=True))
   except BaseException:pass
  raise

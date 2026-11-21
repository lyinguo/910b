"""Read-only metadata-first discovery; every original matrix path/byte retained."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,json,os,shutil,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=Path(sys.argv[1]).resolve();assert H.is_relative_to(B.resolve())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,x):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 H.mkdir(exist_ok=False);assert not runner_lock_owners()
 before=shutil.disk_usage(R).free;buckets=collections.defaultdict(list);scanned=0
 # Native captured arrays only. No weights, task data, reports, source or logs.
 root=R/'workspace/diagnostics/model_runs'
 for home in sorted(root.iterdir()):
  if not home.is_dir() or home.is_symlink():continue
  for p in home.rglob('*.npz'):
   if p.is_symlink() or not p.is_file():continue
   st=p.stat();scanned+=1
   if st.st_size<1024*1024 or st.st_dev!=root.stat().st_dev:continue
   buckets[st.st_size].append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,mtime_ns=st.st_mtime_ns,mode=st.st_mode))
 candidates=[v for v in buckets.values() if len({x['inode'] for x in v})>1];groups=[];hashed=0;hashed_bytes=0
 for num,rows in enumerate(sorted(candidates,key=lambda rows:-rows[0]['bytes'])):
  byhash=collections.defaultdict(list);cache={}
  for row in rows:
   p=Path(row['path']);st=p.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns']);key=(st.st_dev,st.st_ino)
   if key not in cache:cache[key]=sha(p);hashed+=1;hashed_bytes+=st.st_size
   row['SHA256']=cache[key];byhash[row['SHA256']].append(row)
  for digest,rs in byhash.items():
   inodes={x['inode'] for x in rs}
   if len(inodes)>1:
    representatives={}
    for x in rs:representatives.setdefault(x['inode'],x)
    # Estimate only fully single-link duplicate inodes; never assume unseen links.
    canonical=sorted(rs,key=lambda x:(-x['nlink'],x['path']))[0]
    reclaim=sum(x['allocated_bytes'] for ino,x in representatives.items() if ino!=canonical['inode'] and x['nlink']==1)
    groups.append(dict(SHA256=digest,bytes=rs[0]['bytes'],canonical_path=canonical['path'],all_original_paths=rs,complete_unique_physical_copies=len(inodes),potential_allocated_bytes=reclaim))
  if num%10==0:put('progress.json',dict(UTC=now(),phase='hashing_only_equal_size_complete_NPZ_candidates',size_buckets_done=num+1,size_buckets_total=len(candidates),NPZ_metadata_scanned=scanned,complete_files_hashed=hashed,complete_bytes_hashed=hashed_bytes,exact_duplicate_groups=len(groups),potential_bytes=sum(x['potential_allocated_bytes'] for x in groups),no_deletion_or_matrix_change=True))
 result=dict(UTC=now(),passed=True,read_only=True,scope=str(root),NPZ_metadata_scanned=scanned,size_buckets_hashed=len(candidates),complete_files_hashed=hashed,complete_bytes_hashed=hashed_bytes,exact_duplicate_groups=groups,potential_allocated_bytes=sum(x['potential_allocated_bytes'] for x in groups),free_before_bytes=before,free_after_bytes=shutil.disk_usage(R).free,no_deletion_or_source_report_config_log_ledger_change=True,all_original_matrix_paths_bytes_and_SHA_retained=True)
 put('review.json',result);put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_deletion=True,potential_allocated_bytes=result['potential_allocated_bytes']));print(json.dumps({k:v for k,v in result.items() if k!='exact_duplicate_groups'},ensure_ascii=False))
if __name__=='__main__':main()

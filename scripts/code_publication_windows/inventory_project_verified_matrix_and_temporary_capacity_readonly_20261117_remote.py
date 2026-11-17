import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
import os,json,stat,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
dev=R.stat().st_dev;roots=[R/'workspace/diagnostics/model_runs',D,R/'.tmp',R/'tmp',R/'.cache',R/'workspace/.tmp'];groups={};exts=defaultdict(lambda:dict(files=0,logical=0,allocated=0));large=[];errors=[];visited=set()
for root in roots:
 if not root.exists():continue
 stack=[root]
 while stack:
  p=stack.pop()
  try:entries=list(os.scandir(p))
  except OSError as e:errors.append(dict(path=str(p),error=repr(e)));continue
  for e in entries:
   try:
    s=e.stat(follow_symlinks=False);path=Path(e.path)
    if stat.S_ISDIR(s.st_mode):
     if s.st_dev==dev:stack.append(path)
     continue
    if not stat.S_ISREG(s.st_mode) or s.st_dev!=dev:continue
    ident=(s.st_dev,s.st_ino)
    if ident in visited:continue
    visited.add(ident);suffix=path.suffix.lower() or '[none]';a=getattr(s,'st_blocks',0)*512;ex=exts[suffix];ex['files']+=1;ex['logical']+=s.st_size;ex['allocated']+=a
    rel=path.relative_to(root).parts
    if root.name=='model_runs':owner=root/rel[0]
    elif root==D and len(rel)>2 and rel[0]=='A100_relative5pct_alignment_20261103':owner=root/rel[0]/rel[1]/rel[2]
    else:owner=root/(rel[0] if len(rel)>1 else '[root-files]')
    k=str(owner);g=groups.setdefault(k,dict(owner=k,root=str(root),files=0,total_logical=0,total_allocated=0,matrix_files=0,matrix_logical=0,matrix_allocated=0,temporary_files=0,temporary_allocated=0,largest_matrix_files=[]))
    g['files']+=1;g['total_logical']+=s.st_size;g['total_allocated']+=a
    matrix=suffix in ['.npz','.npy'];temp=suffix in ['.tmp','.part','.temp']
    if matrix:
     g['matrix_files']+=1;g['matrix_logical']+=s.st_size;g['matrix_allocated']+=a
     record=dict(path=str(path),bytes=s.st_size,allocated=a,nlink=s.st_nlink,dev=s.st_dev,ino=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
     g['largest_matrix_files'].append(record);g['largest_matrix_files']=sorted(g['largest_matrix_files'],key=lambda x:x['allocated'],reverse=True)[:4]
    if temp:g['temporary_files']+=1;g['temporary_allocated']+=a
    if s.st_size>=128*1024**2 and not matrix:large.append(dict(path=str(path),bytes=s.st_size,allocated=a,suffix=suffix,nlink=s.st_nlink))
   except OSError as e:errors.append(dict(path=e.path,error=repr(e)))
du=subprocess.run(['du','-x','-B1','--max-depth=2',str(R)],capture_output=True,text=True,timeout=45)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,project_disk_device=dev,project_du=dict(returncode=du.returncode,stdout=du.stdout,stderr=du.stderr),file_extension_totals=dict(sorted(exts.items(),key=lambda x:x[1]['allocated'],reverse=True)),matrix_or_temporary_owner_groups=sorted([g for g in groups.values() if g['matrix_files'] or g['temporary_files']],key=lambda x:x['matrix_allocated']+x['temporary_allocated'],reverse=True),largest_other_files=sorted(large,key=lambda x:x['allocated'],reverse=True)[:80],metadata_errors=errors,current_project_processes=project_processes(),original_kernel_FLOCK=runner_lock_owners(),no_array_content_read_or_file_mutated=True),ensure_ascii=False))

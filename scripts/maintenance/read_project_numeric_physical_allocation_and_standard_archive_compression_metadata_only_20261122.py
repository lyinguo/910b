"""Read physical allocation and ZIP central directories only; do not edit, accept or replay values."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,stat,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';S=R/'workspace/diagnostics/model_runs'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
assert not runner_lock_owners()
assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
rows=[];by_suffix={};seen=set()
for parent,dirs,files in os.walk(S,followlinks=False):
    dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
    for n in files:
        p=Path(parent)/n
        if p.suffix not in ['.npz','.npy']:continue
        st=p.lstat()
        if not stat.S_ISREG(st.st_mode) or (st.st_dev,st.st_ino) in seen:continue
        seen.add((st.st_dev,st.st_ino));allocated=st.st_blocks*512
        row=dict(path=str(p),logical_bytes=st.st_size,allocated_bytes=allocated,nlink=st.st_nlink,mtime_ns=st.st_mtime_ns,dev=st.st_dev,inode=st.st_ino)
        rows.append(row);q=by_suffix.setdefault(p.suffix,dict(files=0,logical_bytes=0,allocated_bytes=0));q['files']+=1;q['logical_bytes']+=st.st_size;q['allocated_bytes']+=allocated
selected=[]
for suffix in ['.npy','.npz']:
    chosen=sorted([z for z in rows if z['path'].endswith(suffix)],key=lambda z:-z['allocated_bytes'])[:60]
    for z in chosen:
        if suffix=='.npz':
            try:
                with zipfile.ZipFile(z['path']) as f:
                    methods={}
                    for m in f.infolist():
                        x=methods.setdefault(str(m.compress_type),dict(members=0,raw_bytes=0,compressed_bytes=0));x['members']+=1;x['raw_bytes']+=m.file_size;x['compressed_bytes']+=m.compress_size
                    z=dict(z,complete_ZIP_central_directory_compression_metadata=methods)
            except (OSError,zipfile.BadZipFile) as e:z=dict(z,read_error=repr(e))
        selected.append(z)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),scope=str(S),unique_inode_numeric_allocation=by_suffix,largest_complete_numeric_files_metadata=selected,original_disk_free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,no_array_data_read_or_file_change=True,no_deletion_or_format_change_or_qualification_change=True),ensure_ascii=False))

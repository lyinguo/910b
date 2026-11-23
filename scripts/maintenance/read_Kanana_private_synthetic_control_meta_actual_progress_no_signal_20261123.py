import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
R=Path('/root/projects/comperror');H=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Kanana_own_original_real_RegStage_private_full_synthetic_control_and739_meta_only_v1_20261123'
records={}
if H.is_dir():
    for p in H.iterdir():
        if p.is_file() and p.stat().st_size<2*1024**2:
            b=p.read_bytes()
            try:v=json.loads(b)
            except ValueError:v=b.decode('utf8',errors='replace')
            records[str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_value=v)
procs=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        a=(p/'cmdline').read_bytes().decode().split('\0');cwd=os.readlink(p/'cwd')
        if 'python' not in a[0]:continue
        if cwd!=str(R) and not any('Kanana_private' in x for x in a):continue
        st=(p/'status').read_text();io=(p/'io').read_text();fds={}
        for fd in (p/'fd').iterdir():
            try:t=os.readlink(fd)
            except OSError:continue
            if 'Kanana' in t or fd.name in ['0','1','2']:fds[fd.name]=t
        procs.append(dict(PID=int(p.name),argv=a,cwd=cwd,status=st,io=io,FD=fds))
    except (OSError,IndexError):continue
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),complete_private_control_meta_records=records,actual_processes=procs,no_signal_or_mutation=True),ensure_ascii=False))

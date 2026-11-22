import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,collections
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),read_only=True,no_weight_or_dataset_or_old_matrix_scan=True,retention_policy={},directories=[])
for p in D.glob('*RETENTION*POLICY*.json'):
 raw=p.read_bytes();out['retention_policy'][str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
for name in ['export_only_prof_dir','.tmp','tmp','.cache']:
 home=R/name
 if not home.is_dir() or home.is_symlink():continue
 entries=[];total=0;extensions=collections.Counter()
 for child in home.iterdir():
  if child.is_symlink():continue
  size=0;files=0;exts=collections.Counter()
  iterator=[(str(child.parent),[],[child.name])] if child.is_file() else os.walk(child,followlinks=False)
  for base,dirs,names in iterator:
   dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
   for filename in names:
    p=Path(base)/filename
    if p.is_symlink() or not p.is_file():continue
    st=p.stat();size+=st.st_size;files+=1;exts[p.suffix or '<none>']+=st.st_size
  entries.append(dict(path=str(child),bytes=size,files=files,extension_bytes=dict(exts)));total+=size;extensions.update(exts)
 out['directories'].append(dict(path=str(home),total_bytes=total,extension_bytes=dict(extensions),entries=sorted(entries,key=lambda x:x['bytes'],reverse=True)))
print(json.dumps(out,ensure_ascii=False))

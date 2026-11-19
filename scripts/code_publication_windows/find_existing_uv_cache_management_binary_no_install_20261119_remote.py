import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,stat
roots=['/root/.local','/root/.cache/uv','/root/projects/LYG','/usr/local/python3.12.16','/opt','/root/.venvs','/root/.cargo']
found=[];count=0;errs=[]
for root in roots:
 if not Path(root).is_dir():continue
 for parent,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if d not in ['node_modules','.git','__pycache__'] and not (Path(parent)/d).is_symlink()]
  count+=1
  for n in files:
   if n not in ['uv','uvx']:continue
   p=Path(parent)/n
   try:
    if p.is_file():found.append(dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,executable=os.access(p,os.X_OK)))
   except OSError as e:errs.append(str(e))
print(json.dumps(dict(existing_binary_candidates=found,directories_checked=count,errors=errs,no_install_or_download=True),ensure_ascii=False))

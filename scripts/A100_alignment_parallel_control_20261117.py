"""Pure parallel admission rules plus real process/kernel-lock isolation checks."""
from pathlib import Path
import os,json,hashlib,fcntl,time
GiB=1024**3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def can_admit(free,active,budget,max_parallel):
 assert isinstance(free,int) and free>=0 and isinstance(budget,int) and budget>0
 assert all(isinstance(v,int) and v>0 for v in active.values())
 return len(active)<max_parallel and sum(active.values())+budget<=free
def validate_cards(rows):
 cards=[x['card'] for x in rows];assert cards and len(cards)==len(set(cards)) and all(isinstance(x,int) and 1<=x<=15 for x in cards);return cards
def proc_start(pid):return int((Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19])
def real_exclusive_owner(pid,path):
 st=Path(path).stat();found=[]
 for fd in (Path('/proc')/str(pid)/'fd').iterdir():
  try:
   s=fd.stat();text=(fd.parent.parent/'fdinfo'/fd.name).read_text()
  except OSError:continue
  if (s.st_dev,s.st_ino)==(st.st_dev,st.st_ino) and 'FLOCK' in text and 'WRITE' in text:found.append(str(fd))
 assert found;return found

"""Safety checks for evidence-only original-eager maintenance, without score writes."""
import pathlib,json,os,hashlib,datetime,fcntl,re
R=pathlib.Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
CORE={'scripts/batch_910b.py':'ccc1da1c4bf9ca397df1a3dc95e1fbd7c2d5e2a270c5649f2596e867951063dc','scripts/batch_910b_support.py':'56885858c2ad5acb6929daf93d092187ea6dda0ec562f88650b316f7ce86b7a9'}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,value):
 p=pathlib.Path(p);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def proc(pid,root=pathlib.Path('/proc')):
 p=root/str(pid);st=(p/'stat').read_text().split();a=(p/'cmdline').read_bytes().decode().rstrip(chr(0)).split(chr(0));return {'PID':int(pid),'PPID':int(st[3]),'state':st[2],'start_ticks':int(st[21]),'argv':a,'cwd':os.readlink(p/'cwd')}
def children(pid,root=pathlib.Path('/proc')):
 out=[]
 for p in root.glob('[0-9]*'):
  try:v=proc(int(p.name),root)
  except (OSError,UnicodeError,ValueError):continue
  if v['PPID']==pid and v['state']!='Z':out.append(v)
 return out
def original_identity(value,expected_pid,expected_start=None):
 assert value['PID']==expected_pid and value['state']!='Z' and re.fullmatch(r'python3(?:\.\d+)?',pathlib.Path(value['argv'][0]).name) and value['argv'][1:]==['scripts/batch_910b.py','--run'] and value['cwd']==str(R),'Original controller identity mismatch'
 if expected_start is not None:assert value['start_ticks']==expected_start,'Controller PID was reused'
def idle_contract(state,registry):
 assert state.get('current') is None and state.get('phase')=='waiting_for_adaptation','Not a safe idle boundary'
 assert not any(e.get('child_pid') for e in state['models'].values()),'Canonical entry retains a model child'
 assert not any(registry.get(m,{}).get('approved') for m,e in state['models'].items() if e['status'] not in ['done','skipped_no_complete_weights','skipped_no_dataset']),'A normal approved task is pending'
def semantic_state_sha(state):
 x=dict(state);x.pop('updated_at',None);return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def kernel_lock_owned(pid,path):
 path=pathlib.Path(path);wanted=os.stat(path);matches=[]
 for q in (pathlib.Path('/proc')/str(pid)/'fd').iterdir():
  try:st=os.stat(q);info=q.with_name(q.name).parent.parent/'fdinfo'/q.name;text=info.read_text()
  except OSError:continue
  if (st.st_dev,st.st_ino)==(wanted.st_dev,wanted.st_ino) and 'FLOCK' in text and 'WRITE' in text:matches.append({'fd':q.name,'fdinfo':text})
 assert matches,'Expected real kernel FLOCK is not owned';return matches
def guard_hashes(expected):
 for name,digest in expected.items():assert sha(name)==digest,'Protected file changed concurrently: '+name
def create_pause(path,value):
 path=pathlib.Path(path);data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return hashlib.sha256(data).hexdigest()
def archive_owned_pause(path,expected_sha,dest,owner):
 path=pathlib.Path(path);assert sha(path)==expected_sha and json.loads(path.read_text())['owner']==owner,'Refusing to archive another owner pause';dest=pathlib.Path(dest);assert not dest.exists();os.replace(path,dest)
def core_unchanged():
 for rel,digest in CORE.items():assert sha(R/rel)==digest,'Trusted core changed: '+rel

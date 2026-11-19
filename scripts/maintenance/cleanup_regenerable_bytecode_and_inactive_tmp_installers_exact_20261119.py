"""Exact removal of regenerable bytecode and inactive temporary wheel installers."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,stat,re,json,hashlib,zipfile,shutil,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'derived_bytecode_and_inactive_tmp_installers_final_serial_capacity_gap_20261119';CACHE=Path('/root/.cache/pycache')
WHEELS=[Path('/tmp/mooncake_transfer_engine-0.3.13-cp312-cp312-manylinux_2_35_x86_64.whl'),Path('/tmp/flash_linear_attention_npu_a2-26.9.1-py3-none-manylinux_2_34_x86_64.whl')]
sys.path[:0]=[str(R),str(R/'scripts')]
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def refs(targets):
 found=[];targets=set(map(str,targets))
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   argv=(proc/'cmdline').read_bytes().decode(errors='replace');env=(proc/'environ').read_bytes().split(b'\0');paths=[v.decode(errors='replace') for v in env if v.startswith((b'PYTHONPATH=',b'PATH='))]
   for target in targets:
    if target in argv or any(target in x for x in paths):found.append(dict(PID=int(proc.name),path=target,reason='actual_argv_or_import_path'))
   for fd in (proc/'fd').iterdir():
    try:
     n=os.readlink(fd)
     if n in targets:found.append(dict(PID=int(proc.name),path=n,reason='actual_open_FD'))
    except OSError:pass
   for line in (proc/'maps').read_text(errors='replace').splitlines():
    n=line.split()[-1] if line.split() else ''
    if n in targets:found.append(dict(PID=int(proc.name),path=n,reason='actual_mmap'))
  except OSError:pass
 return found
def scope(p):
 assert p.resolve()==p and not p.is_symlink()
 for q in p.parents:assert not q.is_symlink()
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
 if p in WHEELS:
  assert zipfile.is_zipfile(p)
  with zipfile.ZipFile(p) as z:assert any(n.endswith('.dist-info/METADATA') for n in z.namelist())
  return s,None
 assert p.is_relative_to(CACHE) and p.suffix=='.pyc'
 rel=p.relative_to(CACHE).as_posix();m=re.fullmatch(r'(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc',rel);assert m
 source=Path('/'+m[1]+'.py');assert source.is_file() and source.suffix=='.py'
 return s,source
def guard():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 return runner_lock_owners()
def plan():
 assert not guard();H.mkdir(exist_ok=False);rows=[]
 for parent,dirs,names in os.walk(CACHE,followlinks=False):
  dirs[:]=[x for x in dirs if not (Path(parent)/x).is_symlink()]
  for n in names:
   p=Path(parent)/n
   try:s,source=scope(p)
   except (AssertionError,OSError):continue
   rows.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino,SHA256=sha(p),source=str(source),source_SHA256=sha(source),type='regenerable_bytecode_original_py_source_kept'))
 for p in WHEELS:
  s,_=scope(p);rows.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino,SHA256=sha(p),type='inactive_temporary_installer_not_installed_package'))
 hits=refs([r['path'] for r in rows]);blocked={r['path'] for r in hits};rows=[r for r in rows if r['path'] not in blocked]
 assert rows and all(Path(r['path']).is_relative_to(CACHE) or Path(r['path']) in WHEELS for r in rows)
 current=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')['canonical_SHA256'];canonical={n:sha(n) for n in current};assert {n for n in current if canonical[n]!=current[n]}=={str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')}
 assert canonical[str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')]=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 p=dict(UTC=datetime.now(timezone.utc).isoformat(),user_authorization='Latest direct user asks to remove unnecessary temporary files while all reports/source/config/logs/needed numerical examples remain',files=rows,live_references_preserved=hits,canonical_SHA256=canonical,remove_count=len(rows),allocated_bytes=sum(r['allocated_bytes'] for r in rows),free_before_bytes=shutil.disk_usage(R).free,original_source_and_installed_package_files_kept=True,no_openwebui_specific_cache_or_NPU0_service_changes=True,no_model_evidence_or_control_arrays_selected=True)
 put(H/'plan.json',p);return dict(plan_SHA256=sha(H/'plan.json'),remove_count=p['remove_count'],allocated_bytes=p['allocated_bytes'],temporary_wheel_count=sum(r['type'].startswith('inactive_') for r in rows),free_before_bytes=p['free_before_bytes'],projected_free_after_bytes=p['free_before_bytes']+p['allocated_bytes'])
def apply(reviewed):
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 assert not guard();assert sha(H/'plan.json')==reviewed;p=js(H/'plan.json');assert not refs([r['path'] for r in p['files']])
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert all(x['PID']==os.getpid() for x in runner_lock_owners())
  for r in p['files']:
   s,source=scope(Path(r['path']));assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(r['device'],r['inode'],r['mtime_ns'],r['bytes']) and sha(r['path'])==r['SHA256']
   if source is not None:assert str(source)==r['source'] and sha(source)==r['source_SHA256']
  assert all(sha(n)==h for n,h in p['canonical_SHA256'].items());before=shutil.disk_usage(R).free
  for r in p['files']:Path(r['path']).unlink()
  assert all(sha(n)==h for n,h in p['canonical_SHA256'].items())
  a=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,plan_SHA256=reviewed,deleted_count=len(p['files']),deleted_allocated_bytes=p['allocated_bytes'],free_before_bytes=before,free_after_bytes=shutil.disk_usage(R).free,all_original_sources_and_installed_packages_unchanged=True,all_model_reports_configs_logs_examples_untouched=True,numeric_retention_ledger_unchanged=True,NPU0_services_and_openwebui_specific_caches_untouched=True)
  put(H/'actual_completion.json',a)
 return dict(actual=a,actual_SHA256=sha(H/'actual_completion.json'))

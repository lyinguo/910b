"""Use the existing package manager's supported offline unused-cache operation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,shutil,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'supported_uv_offline_unused_package_cache_prune_final_serial_gap_20261119';sys.path[:0]=[str(R),str(R/'scripts')]
UV=Path('/root/.venvs/uv-bootstrap-910b/bin/uv');CACHE=Path('/root/.cache/uv')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def identity(pid):
 p=Path('/proc')/str(pid)
 return dict(PID=pid,argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),start_ticks=(p/'stat').read_text().split()[21],exe=os.readlink(p/'exe'))
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 assert sha(UV)=='4b5e59bc557ff2cd70934beb59d0eaa92820702853d49f0066b50aa11a8f0c52'
 assert CACHE.resolve()==CACHE and not CACHE.is_symlink();children=[p.name for p in CACHE.iterdir()];assert not any(n.startswith(('environments-','archive-v0-env')) for n in children),'Preserve cached runtime environments'
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');assert not any(part in argv for part in [str(UV)+' ',str(UV.parent/'uvx')+' ']),('Existing package manager active',proc.name)
   assert str(CACHE) not in (proc/'maps').read_text(errors='replace'),('Cache-backed runtime module active',proc.name)
  except OSError:pass
 H.mkdir(exist_ok=False);pointer=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canonical={n:sha(n) for n in pointer['canonical_SHA256']};assert {n for n in canonical if canonical[n]!=pointer['canonical_SHA256'][n]}=={str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')}
 assert canonical[str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')]=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 services={str(pid):identity(pid) for pid in [608610,609296,609952] if (Path('/proc')/str(pid)).exists()}
 command=[str(UV),'cache','prune','--ci','--cache-dir',str(CACHE),'--offline','--no-python-downloads','--no-config']
 before=shutil.disk_usage(R).free;put(H/'reviewed_exact_scope.json',dict(command=command,uv_SHA256=sha(UV),cache_root_children=children,source='https://docs.astral.sh/uv/concepts/cache/#clearing-the-cache',preserve_built_wheels_and_installed_environments=True,no_force=True,no_network_or_install=True,canonical_SHA256=canonical,external_service_identities=services))
 env=dict(os.environ,UV_OFFLINE='1',UV_PYTHON_DOWNLOADS='never',UV_LOCK_TIMEOUT='1',PYTHONDONTWRITEBYTECODE='1');env.pop('UV_NO_CACHE',None)
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock')
  p=subprocess.run(command,capture_output=True,text=True,timeout=120,env=env)
  (H/'uv_stdout.log').write_text(p.stdout,encoding='utf8');(H/'uv_stderr.log').write_text(p.stderr,encoding='utf8')
  assert all(sha(n)==h for n,h in canonical.items())
  assert {pid:identity(int(pid)) for pid in services}==services
  actual=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,command=command,stdout=p.stdout,stderr=p.stderr,free_before_bytes=before,free_after_bytes=shutil.disk_usage(R).free,canonical_SHA256_unchanged=True,installed_packages_and_external_services_not_modified=True,NPU0_service_identity_unchanged=True,no_download_or_install=True,numeric_evidence_and_retention_ledger_unchanged=True,not_model_NN_or_fullNF=True)
  put(H/'actual_completion.json',actual);assert p.returncode==0,actual
 return dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'))
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))

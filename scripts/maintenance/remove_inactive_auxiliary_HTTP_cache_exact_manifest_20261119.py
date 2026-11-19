"""Consume the user's optional-cache pause policy and remove only exact inactive HTTP cache files."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,shutil,stat
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
POLICY=R/'configs/retention/auxiliary_cache_pause_user_20261119.json'
ALLOWED=(Path('/root/.cache/pip/http-v2'),Path('/root/.cache/pip/selfcheck'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def ensure_scope(p):
 assert p.is_absolute() and any(p.is_relative_to(root) and p!=root for root in ALLOWED),str(p)
 assert p.resolve()==p,str(p)
 for parent in [p,*p.parents]:assert not parent.is_symlink(),str(parent)
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1,str(p)
 return s
def live_cache_refs():
 hits=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if any(str(root) in argv for root in ALLOWED):hits.append(dict(PID=int(proc.name),reason='cache root in process arguments'))
   for fd in (proc/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if any(target.startswith(str(root)+'/') for root in ALLOWED):hits.append(dict(PID=int(proc.name),reason='cache open FD',path=target))
    except OSError:pass
  except OSError:pass
 return hits
def checks(policy):
 assert policy['optional_auxiliary_disk_cache_paused'] is True
 assert policy['exact_inactive_cache_roots_allowed_for_this_action']==list(map(str,ALLOWED))
 for k in ['preserve_model_KV_attention_cache_and_all_original_math_runtime_semantics','preserve_complete_reports_sources_configs_logs_all_small_failed_unverified_values_and_required_examples','never_drop_host_page_cache_or_modify_other_services','never_delete_model_assets_installed_package_backing_uv_archives_or_other_disks','no_permission_inferred_for_reducing_full_numerical_evidence_or_original_capacity_guards']:assert policy[k] is True,k
 for k,v in policy['owned_maintenance_environment'].items():assert k in ['PYTHONDONTWRITEBYTECODE','PIP_NO_CACHE_DIR','UV_NO_CACHE'] and v=='1';os.environ[k]=v
 assert not live_cache_refs(),'An HTTP cache file is open; do not delete it'
def run(inventory_path,inventory_sha256,home,apply=False,reviewed_plan_sha256=None):
 sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 policy=js(POLICY);checks(policy);home=Path(home);assert home.resolve().is_relative_to(D)
 assert sha(inventory_path)==inventory_sha256;source=js(inventory_path);rows=source['package_HTTP_cache_files'];assert rows
 ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canonical=ptr['canonical_SHA256'];assert all(sha(p)==v for p,v in canonical.items())
 actual=[]
 for row in rows:
  p=Path(row['path']);s=ensure_scope(p);assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'] and sha(p)==row['sha256'],str(p)
  actual.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,sha256=row['sha256'],device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
 nn=[p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 assert not nn and not runner_lock_owners(),'Actual model/verification/original runner lock active'
 manifest=dict(schema='exact_inactive_package_HTTP_cache_removal_not_model_evidence',source_inventory=str(inventory_path),source_inventory_SHA256=inventory_sha256,policy=str(POLICY),policy_SHA256=sha(POLICY),files=actual,canonical_SHA256=canonical,allocated_bytes=sum(v['allocated_bytes'] for v in actual),logical_bytes=sum(v['bytes'] for v in actual),original_model_evidence_deletion_count=0)
 if not apply:
  home.mkdir(exist_ok=False);put(home/'exact_manifest_plan.json',manifest)
  return dict(phase='reviewable_exact_manifest_only',plan=str(home/'exact_manifest_plan.json'),plan_SHA256=sha(home/'exact_manifest_plan.json'),files=len(actual),allocated_bytes=manifest['allocated_bytes'],canonical_unchanged=True,environment_applied_to_this_owned_maintenance=dict(policy['owned_maintenance_environment']))
 plan=home/'exact_manifest_plan.json';assert sha(plan)==reviewed_plan_sha256;assert js(plan)==manifest
 before=shutil.disk_usage(R).free;deleted=[]
 # Full-file stat/SHA validation is completed before deleting any cache entry.
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  owners=runner_lock_owners();assert any(int(v.get('PID',v.get('pid',-1)))==os.getpid() for v in owners),owners
  checks(policy)
  for row in actual:
   s=ensure_scope(Path(row['path']));assert (s.st_dev,s.st_ino,s.st_mtime_ns)==(row['device'],row['inode'],row['mtime_ns']) and sha(row['path'])==row['sha256']
  assert all(sha(p)==v for p,v in canonical.items())
  for row in actual:
   Path(row['path']).unlink();deleted.append(row)
  assert all(not Path(v['path']).exists() for v in deleted)
  assert all(sha(p)==v for p,v in canonical.items())
  receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,plan_SHA256=reviewed_plan_sha256,deleted_HTTP_cache_files=deleted,deleted_count=len(deleted),deleted_allocated_bytes=sum(x['allocated_bytes'] for x in deleted),free_bytes_before=before,free_bytes_after=shutil.disk_usage(R).free,canonical_SHA256=canonical,canonical_unchanged=True,deleted_model_evidence_files=0,original_numerical_retention_ledger_unchanged=True,environment_applied_to_owned_maintenance=dict(policy['owned_maintenance_environment']),host_page_cache_and_external_services_untouched=True)
  put(home/'actual_cache_cleanup_receipt.json',receipt)
 return dict(passed=True,receipt=str(home/'actual_cache_cleanup_receipt.json'),receipt_SHA256=sha(home/'actual_cache_cleanup_receipt.json'),deleted_count=receipt['deleted_count'],deleted_allocated_bytes=receipt['deleted_allocated_bytes'],free_bytes_before=before,free_bytes_after=receipt['free_bytes_after'],canonical_unchanged=True,model_evidence_files_deleted=0)

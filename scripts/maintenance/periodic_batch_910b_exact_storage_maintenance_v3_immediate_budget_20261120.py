"""Actual periodic policy consumer; defer while NN/NF/EX is active."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
POLICY=R/'configs/maintenance/batch_910b_immediate_disk_recovery_cleanup_v3_20261120.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,x):p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def main():
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 now=datetime.now(timezone.utc);H=B/('periodic_exact_storage_policy_check_'+now.strftime('%Y%m%d_%H%M%S_%f'));H.mkdir(exist_ok=False);cfg=js(POLICY);assert cfg['max_concurrent_models']==1 and cfg['physical_deduplication_keeps_all_original_paths_complete_bytes_and_SHA']
 for name,digest in cfg['exact_source_SHA256'].items():assert sha(R/name)==digest
 required=int(sys.argv[sys.argv.index('--required-free-bytes')+1]) if '--required-free-bytes' in sys.argv else 0;assert 0<=required<=32*2**40;target_free=max(cfg['proactive_free_target_bytes'],required)
 free=shutil.disk_usage(R).free;idle=quiet();base=dict(UTC=now.isoformat(),policy_file=str(POLICY),policy_file_SHA256=sha(POLICY),free_bytes=free,target_free_bytes=target_free,original_NN_NF_and_FLOCK_all_released=idle,actual_NEW_model_forwards=0,no_source_model_cache_math_precision_or_generation_change=True)
 if not idle or '--apply' not in sys.argv or free>=target_free:
  base.update(actual_returncode=0,phase='deferred_active_NN_NF_or_EX' if not idle else 'check_only_or_sufficient_free_space_no_delay_next_model',no_array_or_file_change=True);put(H/'actual_completion.json',base);print(json.dumps(base,ensure_ascii=False));return
 discovery=H/'discovery';argv=[sys.executable,'-B',str(R/cfg['discovery_source']),str(discovery)]
 with (H/'discovery_stdout.log').open('xb') as out,(H/'discovery_stderr.log').open('xb') as err:proc=subprocess.run(argv,cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
 assert proc.returncode==0;review=js(discovery/'review.json');assert review['passed'] and review['read_only']
 target=max(0,target_free-shutil.disk_usage(R).free);groups=[];reclaim=0
 for group in sorted(review['exact_duplicate_groups'],key=lambda x:-x['potential_allocated_bytes']):
  if group['potential_allocated_bytes']<=0:continue
  groups.append(group);reclaim+=group['potential_allocated_bytes']
  if reclaim>=target:break
 if not groups or not quiet():
  base.update(actual_returncode=0,phase='no_duplicate_or_formal_work_started_defer',no_array_or_file_change=True);put(H/'actual_completion.json',base);print(json.dumps(base,ensure_ascii=False));return
 plan=dict(home=str(H/'exact_storage_application'),discovery_review=str(discovery/'review.json'),discovery_review_SHA256=sha(discovery/'review.json'),policy_file=str(POLICY),policy_file_SHA256=sha(POLICY),groups=groups,expected_reclaimed_allocated_bytes=reclaim,all_original_paths_and_SHA_retained=True)
 put(H/'plan.json',plan);argv=[sys.executable,'-B',str(R/cfg['application_source']),str(H/'plan.json'),sha(H/'plan.json')]
 with (H/'application_stdout.log').open('xb') as out,(H/'application_stderr.log').open('xb') as err:proc=subprocess.run(argv,cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
 result=js(H/'exact_storage_application/actual_completion.json');assert proc.returncode==result['actual_returncode']==0;base.update(actual_returncode=0,phase='exact_physical_storage_completed_all_original_payloads_retained',result=result);put(H/'actual_completion.json',base);print(json.dumps(base,ensure_ascii=False))
if __name__=='__main__':main()

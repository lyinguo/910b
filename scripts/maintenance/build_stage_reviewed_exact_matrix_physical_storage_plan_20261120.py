"""Exact selected complete-byte sharing; user-authorized periodic maintenance."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
record=json.loads((V/'R_exact_NPZ_duplicate_storage_discovery_actual2_20261120.json').read_text(encoding='utf-8'));source_path,entry=next((p,x) for p,x in record['files'].items() if Path(p).name=='review.json');review=entry['value'];assert review['passed'] and review['no_deletion_or_source_report_config_log_ledger_change']
selected=[];target=60*2**30;amount=0
for g in sorted(review['exact_duplicate_groups'],key=lambda x:-x['potential_allocated_bytes']):
 if g['potential_allocated_bytes']<=0:continue
 selected.append(g);amount+=g['potential_allocated_bytes']
 if amount>=target:break
assert amount>=target
policy=dict(version=1,user_instruction='最好是将，没有用的中间结果定期进行清理，适配的代码，和我的需要的结果肯定是必须的！其他的这个相关项目的资料，都不重要！',scope=R+'/workspace/diagnostics/model_runs',preserve=['adaptation source and configuration','complete eval reports and every sample input/output/raw/ref/error/token','logs','all required full numerical examples','original model weights and task inputs','all still-needed failed or unverified diagnosis'],max_concurrent_models=1,scheduled_check_minutes=300,opportunity_check_after_complete_acceptance_and_lock_release=True,physical_deduplication_keeps_all_original_paths_complete_bytes_and_SHA=True,storage_method='Same-volume hard links only after complete SHA256 plus full byte comparison; every original path remains a regular complete file',atomic_replace_single_link_duplicate_inodes_only=True,requires_original_global_EX_and_no_model_NN_or_NF=True,active_FD_or_mode_owner_mismatch_skip=True,no_runtime_model_KV_prefix_cache_math_dtype_or_budget_change=True,no_new_numerical_pruning_or_original_retention_ledger_change=True,all_shared_completed_archives_are_immutable_write_new_private_paths_only=True,unused_diagnostic_copy_cleanup_requires_original_values_preserved_and_no_real_consumer=True)
policyrel='configs/maintenance/batch_910b_periodic_exact_storage_retention_20261120.json';policyraw=(json.dumps(policy,ensure_ascii=False,indent=2)+'\n').encode();policysha=hashlib.sha256(policyraw).hexdigest();(V/Path(policyrel).name).write_bytes(policyraw)
n='apply_exact_byte_identical_project_matrix_physical_storage_20261120.py';raw=(V/n).read_bytes();ast.parse(raw);code_sha=hashlib.sha256(raw).hexdigest();codepath=R+'/scripts/maintenance/'+n
home=B+'/user_authorized_periodic_cleanup_exact_matrix_bytes_all_paths_retained_first_serial_recovery_20261120'
plan=dict(home=home,discovery_review=source_path,discovery_review_SHA256=entry['SHA256'],policy_file=R+'/'+policyrel,policy_file_SHA256=policysha,application_source=codepath,application_source_SHA256=code_sha,groups=selected,expected_reclaimed_allocated_bytes=amount,group_count=len(selected),all_original_paths_retained=sum(len(g['all_original_paths']) for g in selected),reclaim_target_bytes=target,no_unique_file_or_array_payload_removed=True,all_paths_same_SHA_after_required=True,all_complete_reports_stat_and_canonical_unchanged_required=True,user_authorization_verbatim=policy['user_instruction'])
planraw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();plansha=hashlib.sha256(planraw).hexdigest();planpath=B+'/user_periodic_exact_byte_matrix_storage_reviewed_plan_20261120.json';(V/'user_periodic_exact_byte_matrix_storage_reviewed_plan_20261120.json').write_bytes(planraw)
items={codepath:dict(SHA256=code_sha,base64=base64.b64encode(raw).decode()),R+'/'+policyrel:dict(SHA256=policysha,base64=base64.b64encode(policyraw).decode()),planpath:dict(SHA256=plansha,base64=base64.b64encode(planraw).decode())}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');items=json.loads(base64.b64decode('PAYLOAD'))
for n,x in items.items():
 p=Path(n);raw=base64.b64decode(x['base64']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];assert not p.exists(),n
for n,x in items.items():
 p=Path(n);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(x['base64']))
L=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/user_periodic_exact_matrix_storage_application_launch_20261120';L.mkdir(exist_ok=False);argv=[sys.executable,'-B','CODE','PLAN','PLANSHA']
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:child=subprocess.Popen(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
x=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,argv=argv,exact_source_SHA256={n:v['SHA256'] for n,v in items.items()},all_original_matrix_paths_and_bytes_retained=True,no_model_NN_or_replay=True);(L/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(x,ensure_ascii=False))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode()).replace("'CODE'",repr(codepath)).replace("'PLANSHA'",repr(plansha)).replace("'PLAN'",repr(planpath))
ast.parse(remote);(V/'stage_reviewed_exact_matrix_physical_storage_plan_20261120_remote.py').write_text(remote,encoding='utf-8',newline='\n');print(json.dumps(dict(plan_SHA256=plansha,groups=len(selected),paths=plan['all_original_paths_retained'],expected_reclaimed_allocated_bytes=amount,application_SHA256=code_sha,policy_SHA256=policysha),ensure_ascii=False))

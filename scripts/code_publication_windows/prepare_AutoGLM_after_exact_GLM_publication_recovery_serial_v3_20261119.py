"""Private serial next-model routing after the preserved publication-only failure."""
from pathlib import Path
import ast, base64, hashlib, json
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
oldprep='prepare_autoglm_phone9b_own_original_BF16_2048_true3_v2_fixed_input_set_20261119.py'
newprep='prepare_autoglm_phone9b_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_20261119.py'
s=(P/oldprep).read_text(encoding='utf8');assert sha(s.encode())=='39e860c15b1a2b4f48eee4cada8997f579bfec08d7ff7afecca3c9052e3c2bcd'
start=s.index(" headers=js(ready['own_source_headers']);oldready=")
end=s.index(" alias_path=",start)
v1=(P/'prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py').read_text(encoding='utf8')
manifest=v1[v1.index(" headers=js(ready['own_source_headers']);manifest={}"):v1.index(" create(H/'own704_active_source_manifest.json'")]
s=s[:start]+manifest+s[end:]
s=s.replace(oldprep,newprep).replace('native_source_readiness_v2_fixed_input_set_20261119','native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261119').replace('true3_v2_fixed_input_set_real_FLOCK_20261119','true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119').replace('native-v2-fixed-input-set-20261119.yaml','native-v3-actual-published-GLM-fixed-input-set-20261119.yaml').replace('20261119_native_v2_fixed_input_set.json','20261119_native_v3_actual_published_GLM_fixed_input_set.json')
s=s.replace(",oldready/'actual_failure.json',oldready/'own704_active_source_manifest.json'",'')
assert 'oldready' not in s
ast.parse(s)
oldqueue='watch_GLM46_then_AutoGLM_preNN_fixed_input_set_recovery_serial_v2_20261119.py'
newqueue='watch_GLM46_exact_qualified181_dual_recovery_then_own_AutoGLM_serial_v3_20261119.py'
oldcfg='after_GLM46_AutoGLM_v1_preNN_config_refusal_own_fixed_input_set_serial_v2_20261119.json'
newcfg='after_GLM46_exact_qualified181_dual_recovery_own_AutoGLM_fixed_input_set_serial_v3_20261119.json'
q=(P/oldqueue).read_text(encoding='utf8');assert sha(q.encode())=='306ca92dc7806961f8ec12e7928220b0e393ff2cf7497016360d3686775c6492'
q=q.replace(oldqueue,newqueue).replace(oldcfg,newcfg).replace('after_GLM46_AutoGLM_v1_preNN_config_refusal_own_fixed_input_set_serial_v2_20261119','after_GLM46_exact_qualified181_dual_recovery_own_AutoGLM_fixed_input_set_serial_v3_20261119').replace('ZaiGLM46Flash_verified181_publication_review_inputs_v2_20261119','ZaiGLM46Flash_verified181_publication_review_inputs_v3_exact_verifier_20261119').replace('native_source_readiness_v2_fixed_input_set_20261119','native_source_readiness_v3_actual_published_GLM_fixed_input_set_20261119').replace('true3_v2_fixed_input_set_real_FLOCK_20261119','true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119').replace('serial-v2-fixed-input-set-20261119.yaml','serial-v3-actual-published-GLM-fixed-input-set-20261119.yaml').replace('serial_v2_fixed_input_set_20261119.json','serial_v3_actual_published_GLM_fixed_input_set_20261119.json').replace('serial_v2_fixed_input_set_real_FLOCK_20261119','serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119')
a=q.index(' cfg=js(R/CFG);',q.index('def main():'))
b=q.index(' validate_report(R,formal',a)
gate=''' cfg=js(R/CFG);assert cfg['max_concurrent_models']==1 and cfg['qualified_GLM_actual_dual_recovery_is_only_dispatch_gate'];assert sha(cfg['own_input_alias_proof'])==cfg['own_input_alias_proof_SHA256']
 unchanged({str(R/n):h for n,h in cfg['own_next_source_SHA256'].items()});assert not H.exists() and not READY.exists() and not J3.exists();H.mkdir();guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old1=B/'after_GLM46_actual200_qualified_dual_commit_or_preserved_gap_AutoGLMPhone_serial_20261119';old2=B/'after_GLM46_AutoGLM_v1_preNN_config_refusal_own_fixed_input_set_serial_v2_20261119'
 assert js(old1/'actual_completion.json')['actual_returncode']==js(old2/'actual_completion.json')['actual_returncode']==1
 assert 'publication_actual_completion.json' in js(old1/'actual_completion.json')['error'] and 'Own next readiness failed before NN' in js(old2/'actual_completion.json')['error']
 assert not (B/'AutoGLMPhone9B_own_original_BF16_2048_true3_v1_real_FLOCK_20261119').exists() and not (B/'AutoGLMPhone9B_own_original_BF16_2048_true3_v2_fixed_input_set_real_FLOCK_20261119').exists()
 while True:
  if (PUBRUN/'publication_actual_completion.json').is_file():
   pr=js(PUBRUN/'publication_actual_completion.json');assert pr['actual_returncode']==0 and pr['source_SHA256']==cfg['qualified_GLM_publisher_SHA256']
   dual=js(PUB/'dual_publication_completed.json');audit=js(PUB/'full_target_dual_publication_audit.json');receipt=js(PUB/'publication_receipt.json');assert dual['completed'] and audit['passed'] and receipt['completed']
   done=js(JGLM/'completion.json');formal,=[v for v in done['accepted'] if v['sample_count']==200];assert done['actual_returncode']==0 and sha(formal['proof'])==formal['proof_sha256']=='cf8d72c9296330ed55e520356c82a8225f0448724c11f405cb640e0858fe62a0';pf=js(formal['proof']);passed=actual_full200_qualification(pf);assert passed
   target,=receipt['targets'];assert target['model_id']=='zai-org/GLM-4.6V-Flash' and target['formal_proof_sha256']==formal['proof_sha256'] and audit['qualified_delivered']==67 and audit['numeric_within5pct']==166
   assert sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')==dual['local_and_server_workbook_sha256']==audit['local_and_server_book_SHA256']
   if quiet():break
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_actual_qualified_GLM181_dual_recovery_completion_and_all_NN_NF_EX_release_no_NN',no_runner_EX_in_waiter=True,no_NN_or_NF_in_waiter=True));time.sleep(3)
 put(H/'exact_prior_publication_only_failure_recovery_scope.json',dict(UTC=now(),passed=True,old_queue_failures_preserved=[str(old1/'actual_completion.json'),str(old2/'actual_completion.json')],actual_qualified_publication_completion=str(PUBRUN/'publication_actual_completion.json'),actual_qualified_publication_completion_SHA256=sha(PUBRUN/'publication_actual_completion.json'),no_prior_AutoGLM_job_or_NN=True,original_input200_proof_reused_without_pixel_token_or_NN_replay=True,no_old_source_job_signal_or_completion_edits=True))
'''
q=q[:a]+gate+q[b:]
ast.parse(q)
oldtree=ast.parse((P/oldqueue).read_bytes());newtree=ast.parse(q)
for name in ['quiet','actual_full200_qualification']:
 get=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name))
 assert get(oldtree)==get(newtree),name
# The capacity/approval/runner loop differs only in fresh private output paths.
oldfn=ast.unparse(next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='continue_own_passed3_full_shape_capacity_first200'))
newfn=ast.unparse(next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='continue_own_passed3_full_shape_capacity_first200'))
assert newfn.replace('serial-v3-actual-published-GLM-fixed-input-set-20261119.yaml','serial-v2-fixed-input-set-20261119.yaml').replace('serial_v3_actual_published_GLM_fixed_input_set_20261119.json','serial_v2_fixed_input_set_20261119.json').replace('serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119','serial_v2_fixed_input_set_real_FLOCK_20261119')==oldfn
cfg=json.loads((P/oldcfg).read_bytes());cfg['qualified_GLM_publisher_SHA256']='dc4160569fe1d8b694da5d0f0d01c112a81579ab6e01ccd0586f0db8e229954a';cfg['own_next_readiness_preparer']='scripts/'+newprep;cfg['own_next_source_SHA256'].pop('scripts/'+oldprep);cfg['own_next_source_SHA256']['scripts/'+newprep]=sha(s.encode());cfg.pop('known_v1_fixed_set_validation_failure_before_any_NN_is_only_recovery_scope');cfg['qualified_GLM_actual_dual_recovery_is_only_dispatch_gate']=True
assets={'scripts/'+newprep:s.encode(),'scripts/'+newqueue:q.encode(),'configs/queues/'+newcfg:(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode()}
for n,raw in assets.items():
 p=P/Path(n).name;assert not p.exists();p.write_bytes(raw)
payload={n:dict(base64=base64.b64encode(v).decode(),SHA256=sha(v)) for n,v in assets.items()}
script='DATA='+repr(payload)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for n,v in DATA.items():
 raw=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];p=R/n
 if n.endswith('.py'):ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
L=B/'AutoGLM_exact_GLM181_publication_recovery_serial_v3_actual_launch_20261119';L.mkdir(exist_ok=False)
source='scripts/watch_GLM46_exact_qualified181_dual_recovery_then_own_AutoGLM_serial_v3_20261119.py'
with (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(R/source)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=source,source_SHA256=DATA[source]['SHA256'],waiting_not_claimed_NN=True,max_concurrent_models=1,all_old_failures_sources_and_logs_preserved=True);(L/'background_launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(v,ensure_ascii=False))
'''
ast.parse(script);lp=P/'launch_AutoGLM_after_exact_GLM_publication_recovery_serial_v3_20261119_remote.py';assert not lp.exists();lp.write_text(script,encoding='utf8',newline='\n')
r=dict(passed=True,own_new_static704_source_hashing_not_replaying_NN_or_input200=True,all_numeric_NF_and_model_math_sources_unchanged=True,capacity_and_fullNN_NF_loops_AST_identical_except_fresh_private_paths=True,requires_real_GLMDual_commit_and_all_locks_released=True,max_models=1,assets=[dict(relative=n,SHA256=sha(v)) for n,v in assets.items()]);(P/'AutoGLM_after_exact_GLM_publication_recovery_serial_v3_scope_tests_20261119.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(r))

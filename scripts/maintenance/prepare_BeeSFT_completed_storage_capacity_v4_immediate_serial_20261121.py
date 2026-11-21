"""Create a private capacity waiter and repair only the original-proof approval binding."""
from pathlib import Path
import ast,hashlib,json,base64
V=Path(__file__).resolve().parent
old=V/'continue_BeeSFT_completed_full_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='73a4a90df70cc2ba3b28d2f3678b2e736a4f7465f158cbb7ce62d39be42e8bc1'
s=old.read_text('utf8')
oldcfg='configs/queues/BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121.json'
newcfg='configs/queues/BeeSFT_completed_storage_NF_capacity_v4_wait_actual_capacity_serial_20261121.json'
newsrc='scripts/continue_BeeSFT_completed_storage_NF_wait_actual_capacity_v4_serial_20261121.py'
assert s.count(oldcfg)==2
s=s.replace(oldcfg,newcfg)
assert s.count("true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256']")==1
s=s.replace("true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256']","true3_proof=original['proof'],true3_proof_SHA256=original['proof_sha256']")
wait=r'''
def wait_for_actual_original_capacity():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged
 q=js(R/'CONFIG_PATH')
 W=Path(q['wait_home']);assert not W.exists();W.mkdir()
 lock=(W/'private_waiter.guard').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source={str(R/p):h for p,h in q['source_sha256'].items()}
 canonical=q['canonical_after_exact117_cleanup_SHA256'];unchanged(canonical)
 old=Path(q['completed_capacity_home']);cap=js(old/'capacity_review.json')
 assert cap['empirical_estimate_bytes']==q['required_free_bytes']==192555984252
 assert cap['all200_own_original_shapes_checked'] and cap['original_factor4_plus6GiB_and_hard3GiB_log1GiB_precheckHalfGiB_sample2GiB_outer10_guards_preserved']
 pf=Path(q['reuse_completed_storage_home'])/'complete_storage_fullNF_proof.json'
 assert sha(pf)=='30927e8537358e03b28aa9bee0fa8679de2c073d84f43847685ecee80261161a'
 cleanup=Path(q['cleanup_actual']);assert sha(cleanup)=='c2ed3e30c9f71b4695ea3dfa35630834f4b9752fc58e13f84c1d64a900c63d3c'
 a=js(cleanup);assert a['actual_returncode']==0 and a['removed_files']==117 and a['all_protected_SHA256_unchanged'] and a['canonical_scores_both_workbooks_state_registry_binding_unchanged']
 assert sha(cleanup.parent/'ledger_before.original')=='6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c'
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd'
 last=None
 while True:
  unchanged(source);unchanged(canonical)
  assert not Path(q['new_job']).exists() and not Path(q['home']).exists()
  owners=runner_lock_owners();procs=project_processes()
  active=[p for p in procs if any(x.endswith('/run.py') or 'verify_' in x or 'A100_alignment_evidence_runner_' in x for x in p.get('argv',[]))]
  free=shutil.disk_usage(R).free
  ready=free>=q['required_free_bytes'] and not owners and not active
  put(W/'state.json',dict(UTC=now(),PID=os.getpid(),phase='capacity_wait_only_no_NN_no_fullNF_no_original_runner_lock',required_free_bytes=q['required_free_bytes'],actual_free_bytes=free,missing_bytes=max(0,q['required_free_bytes']-free),original_FLOCK=owners,actual_other_NN_NF=active,ready=ready,max_concurrent_models=1,no_original_three_or_storage_NF_replay=True,no_expansion_other_disk_cleanup_or_guard_reduction=True))
  if ready:
   put(W/'actual_capacity_recovered.json',dict(UTC=now(),actual_free_bytes=free,required_free_bytes=q['required_free_bytes'],passed=True,all_NN_NF_original_FLOCK_released=True))
   rc=main()
   put(W/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,continuation_home=q['home'],job=q['new_job'],no_G_or_canonical_write=True))
   return rc
  time.sleep(20)
'''.replace('CONFIG_PATH',newcfg)
s=s.replace("if __name__=='__main__':",wait+"\nif __name__=='__main__':")
s=s.replace('try:sys.exit(main())','try:sys.exit(wait_for_actual_original_capacity())')
old_error="if H.exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_reports_values_sources_configs_logs_preserved=True,no_G_or_canonical_write=True))"
new_error=old_error+"\n   W=Path(q['wait_home'])\n   if W.exists() and not (W/'actual_completion.json').exists():put(W/'actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_evidence_preserved=True,no_G_or_canonical_write=True))"
assert s.count(old_error)==1
s=s.replace(old_error,new_error)
assert 'true3_proof=a[' not in s
tree=ast.parse(s)
def fn(text,name):
 return ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)
assert fn(s,'reference_alias')==fn(old.read_text('utf8'),'reference_alias')
assert fn(s,'estimate')==fn(old.read_text('utf8'),'estimate')
compile(s,newsrc,'exec')
(V/Path(newsrc).name).write_text(s,encoding='utf8',newline='\n')
q=json.loads((V/Path(oldcfg).name).read_bytes())
q['config']='/root/projects/comperror/'+newcfg
B='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/'
q['home']=B+'BeeSFT_completed_storage_NF_capacity_v4_actual_recovery_approved_serial_20261121'
q['wait_home']=B+'BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121'
q['new_job']=B+'BeeSFT_completed_storage_NF_capacity_v4_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261121'
q['new_protocol']='configs/protocols/bee8bsft_completed_storage_NF_capacity_v4_actual_recovered_serial_20261121.json'
q['new_config']='configs/BeeForConditionalGeneration/Open-Bee--Bee-8B-SFT-own-original-BF16-4096-complete-LZMA-capacity-v4-serial-20261121.yaml'
q['source_sha256'][newsrc]=hashlib.sha256(s.encode()).hexdigest()
q['required_free_bytes']=192555984252
q['completed_capacity_home']=B+'BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121'
q['cleanup_actual']=B+'verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121/actual_completion.json'
prior=json.loads((V/'Meta117_v3_reviewed_plan_full.json').read_bytes())
q['canonical_after_exact117_cleanup_SHA256']=dict(prior['canonical_SHA256'])
for p in list(q['canonical_after_exact117_cleanup_SHA256']):
 if p.endswith('/A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'):q['canonical_after_exact117_cleanup_SHA256'][p]='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd'
(V/Path(newcfg).name).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
records=[]
for rel in [newsrc,newcfg]:
 b=(V/Path(rel).name).read_bytes();records.append(dict(path=rel,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
deploy='DATA='+repr(records)+r'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,subprocess,os
R=Path('/root/projects/comperror')
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
q=json.loads((R/DATA[1]['path']).read_bytes());W=Path(q['wait_home']);assert not W.exists()
out=Path(str(W)+'_launcher_stdout.log');err=Path(str(W)+'_launcher_stderr.log')
with out.open('x') as o,err.open('x') as e:
 p=subprocess.Popen(['python3','-B',str(R/DATA[0]['path'])],cwd=R,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
print(json.dumps(dict(launched=True,PID=p.pid,source=DATA[0]['path'],source_SHA256=DATA[0]['SHA256'],config=DATA[1]['path'],config_SHA256=DATA[1]['SHA256'],wait_home=str(W),not_formal_NN_or_NF=True,max1=True),ensure_ascii=False))
'''
(V/'launch_BeeSFT_completed_storage_capacity_v4_immediate_serial_20261121_remote.py').write_text(deploy,encoding='utf8',newline='\n')
print(json.dumps(dict(prepared=True,source=records[0]['SHA256'],config=records[1]['SHA256'],checks=['reference_alias_AST_exact','all200_component_estimate_AST_exact','original_proof_unbound_name_repaired','no_model_math_dtype_budget_change','no_NN_or_fullNF_replay','source_compile']),ensure_ascii=False))

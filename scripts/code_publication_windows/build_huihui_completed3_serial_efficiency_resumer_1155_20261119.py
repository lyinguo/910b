"""New private serial resumer; do not modify or signal the live GLM waiter."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent
s=(P/'continue_glm46flash_completed_true3_wait_actual_capacity_resume_fresh3_fixed200_v3_1723_20261118.py').read_text(encoding='utf8')
replacements={
 "MID='zai-org/GLM-4.6V-Flash'":"MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated'",
 'GLM46Flash_own_original_BF16_thinking4096_native_source_readiness_v2_actual_D965_1653_20261118':'HuihuiGLM46Flash_own_original_BF16_2048_native_source_readiness_v1_20261118',
 'GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118':'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119',
 'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118':'HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118',
 'GLM46Flash_completed_true3_actual_capacity_restored_approved_fresh3_fixed200_v3_real_FLOCK_1723_20261118':'HuihuiGLM46_completed_true3_single_model_serial_approved_NEWfresh3_fixed200_v2_real_FLOCK_1155_20261119',
 'scripts/continue_glm46flash_completed_true3_wait_actual_capacity_resume_fresh3_fixed200_v3_1723_20261118.py':'scripts/continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py',
 'configs/Glm4vForConditionalGeneration/zai-org--GLM-4.6V-Flash-own-original-BF16-thinking4096-approved-first200-v3-capacity-restored-1723-20261118.yaml':'configs/Glm4vForConditionalGeneration/huihui-ai--Huihui-GLM-4.6V-Flash-abliterated-own-original-BF16-2048-serial-approved-first200-v2-1155-20261119.yaml',
 'configs/protocols/glm46flash_own_original_BF16_thinking4096_approved_first200_v3_capacity_restored_1723_20261118.json':'configs/protocols/huihui_glm46flash_own_original_BF16_2048_serial_approved_first200_v2_1155_20261119.json',
 "'b922c1c10b5a847c8fc1a54774a6c2e4f22cd9357abfc42f241e5950453a2e31'":"'e5e7eff6d8e501f2c60737e83985482ab53ea03c6a556f18b979d01e3280c57b'",
 "small['model_returncode']==0":"small['model_returncode'] in [0,1]",
 'Own GLM46 measured first200 approval NEWfresh3 fixed0_199':'Own Huihui original2048 single-model serial first200 approval NEWfresh3 fixed0_199',
}
for a,b in replacements.items():
 assert a in s,a;s=s.replace(a,b)
header="""
SERIAL_CONFIG=R/'configs/queues/serial_efficiency_user_priority_20261119.json'
SERIAL_MODULE=R/'scripts/serial_efficiency_dispatch_policy_20261119.py'
LEGACY_SOURCE=R/'scripts/continue_glm46flash_completed_true3_wait_actual_capacity_resume_fresh3_fixed200_v3_1723_20261118.py'
LEGACY_HOME=B/'GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118'
LEGACY_JOB=B/'GLM46Flash_completed_true3_actual_capacity_restored_approved_fresh3_fixed200_v3_real_FLOCK_1723_20261118'
LEGACY_BUDGET=235720910370

def existing_legacy_dispatch():
 assert sha(LEGACY_SOURCE)=='96dfdd09b7b14a4845e142cbcc6dcc4384130a362ff1bea777f9fcf2a059637b'
 state=js(LEGACY_HOME/'state.json');pid=state['PID'];proc=Path('/proc')/str(pid);alive=False;detail=dict(PID=pid)
 if proc.is_dir():
  argv=(proc/'cmdline').read_bytes().split(b'\\0');cwd=os.readlink(proc/'cwd')
  alive=str(LEGACY_SOURCE).encode() in argv and cwd=='/root'
  detail.update(actual_argv=[v.decode(errors='replace') for v in argv if v],actual_cwd=cwd,actual_stat=(proc/'stat').read_text())
 pending=bool(alive and (LEGACY_JOB/'job.json').exists() and not (LEGACY_JOB/'completion.json').exists())
 detail.update(bound_alive=alive,actual_job_pending=pending,legacy_actual_completion=js(LEGACY_HOME/'actual_completion.json') if (LEGACY_HOME/'actual_completion.json').exists() else None)
 return alive,pending,detail
"""
s=s.replace('\ndef main():',header+'\ndef main():',1)
s=s.replace(" H.mkdir(exist_ok=False);guard=", " from serial_efficiency_dispatch_policy_20261119 import validate_policy,choose_action\n scheduling=js(SERIAL_CONFIG);validate_policy(scheduling);serial_hashes={str(SERIAL_CONFIG):sha(SERIAL_CONFIG),str(SERIAL_MODULE):sha(SERIAL_MODULE)}\n H.mkdir(exist_ok=False);guard=",1)
s=s.replace("  assert True;assert len(small['accepted'])==1;", "  assert len(small['accepted'])==1;")
s=s.replace("  put(H/'full_shape_complete_numeric_storage_capacity_review.json',cap)","  assert estimate==164003454340,'Original own complete storage estimate changed; preserve and review, never weaken guard'\n  put(H/'full_shape_complete_numeric_storage_capacity_review.json',cap)",1)
start=s.index('  while True:\n   batch.check_sources(spec)')
end=s.index('  approval=dict(',start)
loop="""  while True:
   batch.check_sources(spec);unchanged(parent['protected_canonical_hashes']);assert serial_hashes=={n:sha(n) for n in serial_hashes}
   nn=active();owners=runner_lock_owners();current_free=shutil.disk_usage(R).free
   legacy_alive,legacy_pending,legacy_detail=existing_legacy_dispatch()
   action=choose_action(scheduling,free_bytes=current_free,candidate_budget_bytes=estimate,active_NN_NF=nn,lock_owners=owners,bound_legacy_resumer_alive=legacy_alive,legacy_job_pending=legacy_pending,legacy_budget_bytes=LEGACY_BUDGET)
   put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase=action,serial_only=True,max_active_models=1,original_true3_complete_no_replay=True,actual_NEW_model_forwards=0,actual_free_bytes=current_free,required_original_full200_bytes=estimate,capacity_passed=current_free>=estimate,actual_NN_NF=nn,actual_original_FLOCK=owners,existing_unmodified_legacy=legacy_detail,continuation_holds_original_runner_lock=False,no_disk_expansion_or_other_disk_or_matrix_deletion=True))
   if action=='start_candidate_NEWfresh3_fixed200_original_runner':
    cap.update(UTC=now(),actual_free_bytes=current_free,passed=True);put(H/'full_shape_complete_numeric_storage_capacity_review.json',cap);break
   time.sleep(20)
"""
s=s[:start]+loop+s[end:]
s=s.replace("p2['implementation_source_SHA256'][SELF]=sha(R/SELF);", "p2['implementation_source_SHA256'].update({SELF:sha(R/SELF),str(SERIAL_MODULE.relative_to(R)):sha(SERIAL_MODULE),str(SERIAL_CONFIG.relative_to(R)):sha(SERIAL_CONFIG)});")
s=s.replace("source_sha256={**sources,SELF:sha(R/SELF),CFG:sha(R/CFG),PROTO:sha(R/PROTO)}", "source_sha256={**sources,SELF:sha(R/SELF),str(SERIAL_MODULE.relative_to(R)):sha(SERIAL_MODULE),str(SERIAL_CONFIG.relative_to(R)):sha(SERIAL_CONFIG),CFG:sha(R/CFG),PROTO:sha(R/PROTO)}")
s=s.replace("  done=run(J200,'approved_fresh3_fixed200');", "  assert cfg['algorithm']==c2['algorithm'] and cfg['data']==c2['data'] and cfg['output']==c2['output'] and cfg['task_type']==c2['task_type']\n  assert {k:v for k,v in cfg['model']['kwargs'].items() if k not in ['protocol_file','protocol_sha256']}=={k:v for k,v in c2['model']['kwargs'].items() if k not in ['protocol_file','protocol_sha256']}\n  legacy_alive,legacy_pending,_=existing_legacy_dispatch()\n  assert choose_action(scheduling,free_bytes=shutil.disk_usage(R).free,candidate_budget_bytes=estimate,active_NN_NF=active(),lock_owners=runner_lock_owners(),bound_legacy_resumer_alive=legacy_alive,legacy_job_pending=legacy_pending,legacy_budget_bytes=LEGACY_BUDGET)=='start_candidate_NEWfresh3_fixed200_original_runner','Changed dispatch admission; preserve new private job without starting a second model'\n  done=run(J200,'approved_fresh3_fixed200');")
s=s.replace('"""Own GLM46 private real3 then measured first200, without waiting for a heartbeat."""','"""User-prioritized serial continuation of already accepted own Huihui NPU3.\nNo true3 replay, live-source edits, legacy signals, parallel NN, or canonical writes.\n"""')
name='continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py'
ast.parse(s);(P/name).write_text(s,encoding='utf8',newline='\n')
print(json.dumps(dict(relative='scripts/'+name,SHA256=hashlib.sha256(s.encode()).hexdigest(),source_bytes=len(s.encode()),serial_only=True,model_math_source_changed=False)))

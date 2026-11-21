"""Read-only event gate: current own Single200, conditional dual audit, then next resources.

This waiter owns no runner lock, performs no learned or numerical inference,
never writes a G cell, and never repeats a completed model stage.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import fcntl,hashlib,json,os,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
CFG=R/'configs/queues/after_AutoGLMSingle_actual_fixed200_conditional_dual_then_original_resources_serial_20261120.json'
H=B/'after_AutoGLMSingle_actual_fixed200_conditional_dual_then_original_resources_serial_20261120'
sys.path[:0]=[str(R),str(R/'scripts')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from batch_910b_support import validate_report
 from eager_corrective_control_safety_2203 import core_unchanged
 cfg=js(CFG);assert cfg['max_concurrent_models']==1
 J=Path(cfg['current_own_job']);assert sha(J/'job.json')==cfg['current_own_job_SHA256']
 job=js(J/'job.json');assert job['model_id']=='zai-org/AutoGLM-Phone-9B' and job['stages']==[3,200] and not job['write_workbook']
 H.mkdir(exist_ok=False);guard=(H/'private_event.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 oldphase=None
 def status(phase,**extra):
  nonlocal oldphase
  if oldphase!=phase:
   put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase=phase,no_NN_or_NF_in_waiter=True,no_original_runner_EX_in_waiter=True,**extra));oldphase=phase
 while not (J/'completion.json').exists():
  status('waiting_same_own_fixed200_actual_model_and_fullNF_completion_no_NN_here');time.sleep(3)
 done=js(J/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0 and [e['sample_count'] for e in done['accepted']]==[3,200]
 formal=done['accepted'][-1];assert sha(formal['proof'])==formal['proof_sha256'];pf=js(formal['proof'])
 assert pf['passed'] and pf['sample_count']==200 and pf['actual_NPU_inference'] and pf['NEW_real_NPU_inference'] and not pf['historical_record_reuse']
 assert pf['independent_verifier_SHA256']==cfg['own_verifier_SHA256'] and pf['source_tensor_count']==704 and pf['all_own_source_actual_NPU_SHA256_verified'] and pf['all_native_KV_values_losslessly_reconstructed'] and pf['all_full_inputs_tokens_raw_references_errors_preserved']
 assert validate_report(R,formal['report_dir'],job['model_id'],'image-text-to-text',200,pf)==formal['score']
 errs=pf['relative_error_to_current_A100_D'];assert set(errs)=={'server','local','preserved_original'}
 qualified=all(Decimal('0')<=Decimal(str(e))<=Decimal('.05') for e in errs.values())
 assert pf['formal_alignment_within_5pct'] is qualified and pf['original_protocol_final_qualification_complete'] is qualified
 controller=Path(cfg['current_own_controller'])
 while not (controller/'actual_completion.json').exists():
  status('waiting_actual_current_own_controller_exit_after_fullNF');time.sleep(3)
 assert js(controller/'actual_completion.json')['actual_returncode']==0
 if qualified:
  pub=Path(cfg['qualified183_publication'])
  while not (pub/'dual_publication_completed.json').exists() or not (pub/'full_target_dual_publication_audit.json').exists():
   status('waiting_own_qualified200_required_actual_both_books_and_final_audit_no_new_NN',formal_proof_SHA256=formal['proof_sha256']);time.sleep(3)
  dual=js(pub/'dual_publication_completed.json');audit=js(pub/'full_target_dual_publication_audit.json')
  assert dual['completed'] and audit['passed'] and audit['qualified_delivered']==69 and audit['numeric_within5pct']==168
  reg=js(D/'registry.json');assert reg[job['model_id']]['accepted_formal200_proof_sha256']==formal['proof_sha256']
 else:
  assert done['model_returncode']==1 and js(D/'state.json')['models'][job['model_id']]['score'] is None and job['model_id'] not in js(D/'registry.json')
 roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
 while runner_lock_owners() or any(p['PID']!=os.getpid() and p['role'] in roles for p in project_processes()):
  status('waiting_all_actual_model_NF_original_runner_lock_release_no_NN_here');time.sleep(3)
 core_unchanged();selector=R/cfg['selector_source'];assert sha(selector)==cfg['selector_SHA256']=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494'
 result=subprocess.run(['python3','-B',str(selector)],cwd='/root',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
 (H/'original_selector_stdout.json').write_bytes(result.stdout);(H/'original_selector_stderr.log').write_bytes(result.stderr)
 assert result.returncode==0,result.stderr.decode(errors='replace')
 selected=json.loads(result.stdout)
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,actual_dispatch=selected.get('dispatch'),actual_original_resource_rows=selected,preceding_full200_proof_SHA256=formal['proof_sha256'],preceding_original_protocol_qualified=qualified,qualified_requires_real_dual_audit=qualified,no_G_or_canonical_write_by_waiter=True,no_stage_replayed=True,max_concurrent_models=1,NOTIFY_file_is_not_chat_notification=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_replay_or_G_or_canonical_write=True,chat_notification_not_claimed=True))
  raise

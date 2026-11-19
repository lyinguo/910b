"""One serial event continuation after actual accepted AutoGLM182 delivery."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,time,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'after_AutoGLM182_actual_dual_audit_release_original_resources_serial_20261120';PUB=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);guard=(H/'followup.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  if (PUB/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json').exists():
   actual=js(PUB/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json');audit=js(PUB/'full_target_dual_publication_audit.json');dual=js(PUB/'dual_publication_completed.json');assert actual['actual_returncode']==0 and audit['passed'] and dual['completed'] and audit['qualified_delivered']==68
   if not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]:break
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='waiting_actual_AutoGLM182_dual_final_audit_and_all_NN_NF_EX_release_no_NN_in_waiter',no_runner_lock_in_waiter=True));time.sleep(3)
 cfg=js(R/'configs/queues/existing_original_resources_immediately_after_Prithiv_completed_card2_20261117.json');selector=R/cfg['selector_source'];assert sha(selector)==cfg['selector_SHA256']=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494'
 r=subprocess.run(['python3','-B',str(selector)],cwd='/root',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE);(H/'original_selector_stdout.json').write_bytes(r.stdout);(H/'original_selector_stderr.log').write_bytes(r.stderr);assert r.returncode==0,r.stderr.decode(errors='replace')
 selected=json.loads(r.stdout);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,selector_actual_returncode=0,actual_dispatch=selected.get('dispatch'),actual_original_resource_rows=selected,no_stage_replayed=True,no_G_or_canonical_write=True,max_concurrent_models=1,NOTIFY_file_is_not_chat_notification=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_replay_or_canonical_write=True))
  raise

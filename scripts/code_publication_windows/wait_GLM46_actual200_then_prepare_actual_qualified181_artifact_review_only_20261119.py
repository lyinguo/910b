"""Keep this active review attached to the current genuine200, no periodic wait.

No production workbook writes. Stops at actual authored PNG review, baseline gap,
or real error so the operator can inspect the actual evidence immediately.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,traceback
P=Path(__file__).resolve().parent;H=P/'GLM46_live_actual200_to_qualified181_artifact_review_driver_20261119';H.mkdir(exist_ok=False)
PY=sys.executable;NODE=Path('C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
def now():return datetime.now(timezone.utc).isoformat()
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run(args,base):
 with (H/(base+'.stdout.log')).open('xb') as out,(H/(base+'.stderr.log')).open('xb') as err:
  result=subprocess.run(args,cwd=P,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 assert result.returncode==0,(base,result.returncode,str(H/(base+'.stderr.log')))
def main():
 sequence=0
 while True:
  dest=H/'actual_probe_latest.json';args=[PY,'-X','utf8','-B',str(P/'remote_strict_config_copy_0721_20261114.py'),'probe_GLM46_serial_actual200_and_dual_commit_gate_for_local_artifact_review_20261119_remote.py',str(dest)]
  result=subprocess.run(args,cwd=P,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  if result.returncode:
   (H/('probe_transport_failure_'+str(sequence)+'.stdout.log')).write_bytes(result.stdout);(H/('probe_transport_failure_'+str(sequence)+'.stderr.log')).write_bytes(result.stderr);raise RuntimeError('Critical strict SSH/probe check failed; formal status cannot be confirmed')
  x=json.loads(dest.read_bytes());sequence+=1
  with (H/'all_actual_probe_lines.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  print(json.dumps({k:x[k] for k in ['UTC','sample_count','committed_actual_forwards','free_bytes']}|dict(full200_complete=x['full200_proof_summary'] is not None),ensure_ascii=False),flush=True)
  assert not x['failure_records'],('Actual model failure record',x['failure_records'])
  done=x['job_completion'];control=x['controller_completion']
  if done and not done.get('partial_live_JSON_not_a_completed_result'):
   assert done['actual_returncode']==0,('Actual GLM runner failed',done)
  if control and not control.get('partial_live_JSON_not_a_completed_result'):
   assert control['actual_returncode']==0,('Actual GLM controller failed',control)
  if x['full200_proof_summary'] and done and control and not x['original_FLOCK'] and not any(str(p) in x['actual_procs'] for p in [1573127,1573205,1574589]):break
  time.sleep(25)
 pf=x['full200_proof_summary'];assert pf['passed'] and pf['sample_count']==200
 run([PY,'-X','utf8','-B',str(P/'remote_strict_config_copy_0721_20261114.py'),'read_ZaiGLM46_completed200_exact_publication_inputs_no_live_assumptions_20261119_remote.py',str(P/'R_ZaiGLM46_completed200_exact_publication_inputs_20261119.json')],'actual_completed200_full_inputs_read')
 if not pf['original_protocol_final_qualification_complete']:
  value=dict(UTC=now(),actual_returncode=0,phase='full200_numerically_verified_baselinegap_G_preserved_no_publication',full200=pf,next_own_serial_queue_already_event_bound=True,no_production_workbook_write=True);put('actual_completion.json',value);print(json.dumps(value,ensure_ascii=False),flush=True);return
 assert pf['formal_alignment_within_5pct']
 run([PY,'-X','utf8','-B',str(P/'prepare_ZaiGLM46_verified181_completed200_actual_score_local_plan_20261119.py')],'actual_qualified200_local_plan')
 target=P/'ZaiGLM46Flash_local_result_sync_verified181_20261119'
 run([str(NODE),str(P/'zai_glm46_workbook_cell_verified181_20261119.mjs'),str(target/'plan.json'),'author'],'artifact_actual_score_author_and_render')
 assert (target/'after_697.png').is_file() and (target/'artifact_authored_G_values.json').is_file()
 value=dict(UTC=now(),actual_returncode=0,phase='actual_qualified200_artifact_authored_ready_for_actual_PNG_view_before_production_write',actual_after_PNG=str(target/'after_697.png'),actual_after_PNG_SHA256=hashlib.sha256((target/'after_697.png').read_bytes()).hexdigest(),full200=pf,no_production_workbook_write=True);put('actual_completion.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_production_workbook_write_by_this_driver=True));raise

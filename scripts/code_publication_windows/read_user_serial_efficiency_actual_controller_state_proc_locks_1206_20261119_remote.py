import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
pointer=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canonical={p:sha(p) for p in pointer['canonical_SHA256']};assert canonical==pointer['canonical_SHA256']
homes=dict(Huihui_serial=B/'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119',existing_zai_legacy=B/'GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118')
controllers={}
for name,home in homes.items():
 state=js(home/'state.json');pid=state['PID'];proc=Path('/proc')/str(pid)
 record=dict(state=state,state_SHA256=sha(home/'state.json'),actual_completion=js(home/'actual_completion.json') if (home/'actual_completion.json').exists() else None,proc_exists=proc.is_dir())
 if proc.is_dir():record['actual_proc']=dict(stat=(proc/'stat').read_text(),argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(proc/'cwd'),io=(proc/'io').read_text())
 controllers[name]=record
L=B/'Huihui_serial_efficiency_user_decision_policy_and_waiter_launch_1158_20261119'
launch=js(L/'launch.json');source_bindings={item['relative']:sha(R/item['relative']) for item in launch['primary_sources']};assert source_bindings=={item['relative']:item['SHA256'] for item in launch['primary_sources']}
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
procs=project_processes();owners=runner_lock_owners();smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert smi.returncode==0
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),controllers=controllers,launch=launch,serial_policy=js(R/'configs/queues/serial_efficiency_user_priority_20261119.json'),serial_scope_test=js(L/'isolated_serial_scope_tests_stdout.log'),source_bindings=source_bindings,pointer=pointer,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),canonical_SHA256=canonical,free_bytes=shutil.disk_usage(R).free,actual_project_processes=procs,actual_NN_NF=[p for p in procs if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']],actual_original_FLOCK=owners,npu_smi=smi.stdout,new_Huihui_J200_exists=(B/'HuihuiGLM46_completed_true3_single_model_serial_approved_NEWfresh3_fixed200_v2_real_FLOCK_1155_20261119/job.json').exists(),no_replay_or_source_mutation=True),ensure_ascii=False))

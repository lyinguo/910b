import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except Exception:return b.decode('utf8',errors='replace')
M=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
m=subprocess.run(['python3','-B',str(M)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert m.returncode==0,m.stderr
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_XtunerLlava3_private_IModel_v2_fullNF_serial_current_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
H=B/'LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122';READY=B/'LlavaMore_own_original_FP16_letter_source_readiness_v1_20261122';J=B/'LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122';I=B/'LlavaMore_own_original_probability_after_Xtuner_serial_v1_controller_launcher_20261122'
current=json.loads(r.stdout);owners=[]
for owner in current['original_FLOCK']:
 proc=Path('/proc')/str(owner['PID'])
 try:owners.append(dict(owner=owner,argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x],cwd=os.readlink(proc/'cwd'),stat=(proc/'stat').read_text(),status=(proc/'status').read_text(),io=(proc/'io').read_text(),fd={f.name:os.readlink(f) for f in (proc/'fd').iterdir()}))
 except FileNotFoundError:owners.append(dict(owner=owner,natural_exit_during_read=True))
pointer=read(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')
out=dict(UTC=datetime.now(timezone.utc).isoformat(),current_Xtuner_same_serial_actual=current,actual_original_lock_owner_proc=owners,LlavaMore_new_source_ready_and_serial_waiter={str(p):read(p) for p in [H/'state.json',H/'actual_completion.json',H/'readiness_stderr.log',READY/'readiness.json',READY/'actual_completion.json',READY/'actual_failure.json',J/'state.json',J/'completion.json',I/'launch.json',I/'stderr.log']},complete_monitor_latest_checkpoint={str(p):read(p) for p in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json']},pointer=pointer,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),closing_SHA256=sha(Path(pointer['handoff_directory'])/'closing_addendum.json'),all_current_canonical_fresh_SHA256={n:sha(n) for n in pointer['canonical_SHA256']})
print(json.dumps(out,ensure_ascii=False))

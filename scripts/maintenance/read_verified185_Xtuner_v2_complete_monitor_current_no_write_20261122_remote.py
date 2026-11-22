import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
M=R/'scripts/A100_alignment_incremental_monitor_20261103.py';assert sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d';m=subprocess.run(['python3','-B',str(M)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert m.returncode==0,m.stderr
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_XtunerLlava3_private_IModel_v2_fullNF_serial_current_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
j=json.loads(r.stdout);j['complete_monitor_latest_checkpoint']={str(p):json.loads(p.read_bytes()) for p in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json']};j['pointer']=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());j['pointer_SHA256']=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');j['closing_SHA256']=sha(Path(j['pointer']['handoff_directory'])/'closing_addendum.json');j['BeeSFT_v5_actual_state']=json.loads((B/'BeeSFT_completed_storage_NF_capacity_v5_verified185_actual_original_disk_waiter_serial_20261122/state.json').read_bytes());print(json.dumps(j,ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119'
s=json.loads((J/'state.json').read_bytes());pid=s.get('child_PID');p=Path('/proc')/str(pid);v=dict(PID=pid,exists=p.exists())
if p.exists():
 v.update(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text(),wchan=(p/'wchan').read_text())
 fds=[]
 for fd in (p/'fd').iterdir():
  try:
   n=os.readlink(fd)
   if any(x in n for x in ['safetensors','model-','GLM','sample_','/mnt/afs','/dev/davinci']):fds.append(n)
  except OSError:pass
 v['relevant_actual_FDs']=fds
card=subprocess.run(['npu-smi','info'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage=s['phase'],actual_process=v,npu_smi_returncode=card.returncode,npu_smi=card.stdout.decode(errors='replace'),npu_smi_stderr=card.stderr.decode(errors='replace'),no_signal_no_restart_no_model_or_config_change=True),ensure_ascii=False))

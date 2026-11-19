import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
canonical={p:sha(p) for p in pointer['canonical_SHA256']};assert canonical==pointer['canonical_SHA256']
stages={};arrays=[]
for name in ['Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_1106_20261119','Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_v2_json_sets_1116_20261119','Moondream3_own_actual_NPU_loading_all662_runtime_buffers_no_forward_1120_20261119']:
 home=B/name;records={}
 for p in sorted(home.iterdir()):
  if p.is_file() and p.suffix=='.json':records[p.name]=dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size,value=json.loads(p.read_bytes()))
  elif p.is_file() and p.suffix=='.npz':arrays.append(dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size,role='All original runtime mask/RoPE/zero-KV values from no-forward loading audit'))
 stages[str(home)]=records
sources=[]
for name in ['diagnostic_moondream3_own_original_HF_actual_loading_precision_no_forward_1106_20261119.py','diagnostic_moondream3_own_original_HF_actual_loading_precision_no_forward_v2_json_sets_1116_20261119.py','diagnostic_moondream3_own_actual_NPU_loading_runtime_buffers_no_forward_1120_20261119.py']:
 p=R/'scripts'/name;sources.append(dict(relative=p.relative_to(R).as_posix(),SHA256=sha(p),bytes=p.stat().st_size))
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert smi.returncode==0
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pointer=pointer,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),canonical_SHA256=canonical,stage_metadata=stages,all_new_numeric_control_files=arrays,new_source_inventory=sources,actual_project_processes=project_processes(),actual_original_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free,npu_smi=smi.stdout,npu_smi_stderr=smi.stderr),ensure_ascii=False))

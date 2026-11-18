import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')];os.chdir('/root')
J=B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118'
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
files={}
for name in ['state.json','completion.json','fresh3/accepted.json','fresh3/verification_stdout.json','fresh3/verification_stderr.log','fresh3/console.log']:
 p=J/name
 if not p.exists():continue
 raw=p.read_bytes()
 if name in ['state.json','completion.json','fresh3/accepted.json']:
  try:value=json.loads(raw)
  except ValueError:value=dict(nonfinal_content_preserved=True,raw_text=raw.decode(errors='replace'))
 else:value=raw.decode('utf-8',errors='replace')[-22000:]
 files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=value)
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,actual_project_processes=project_processes(),actual_original_FLOCK=runner_lock_owners(),npu_smi=smi.stdout,free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=B/'Pali10Pt448_new_full_native_numeric_true3_v10_actual_supervision_20261117';J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117'
v=dict(UTC=datetime.now(timezone.utc).isoformat(),procs=project_processes(),FLOCK=runner_lock_owners(),free=shutil.disk_usage(D).free,files={})
for p in [H/'state.json',H/'actual_completion.json',H/'runner_stdout.log',J/'state.json',J/'completion.json',J/'job.json',J/'fresh3/console.log',J/'fresh3/verification_stderr.log']:
 if not p.exists():continue
 raw=p.read_bytes();entry=dict(SHA256=sha(p),bytes=len(raw))
 if p.suffix=='.json':entry['value']=json.loads(raw)
 else:entry['tail']=raw.decode('utf8',errors='replace').splitlines()[-15:]
 v['files'][str(p)]=entry
for p in [R/'universal_eval/model/pali10pt448_full_native_value_20261117/worker_v2_ABI.py',R/'universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117.py']:
 v['files'][str(p)]=dict(SHA256=sha(p),text=p.read_text())
old=R/'workspace/diagnostics/model_runs/pali10_pt448_native_vllm_a5ix9omo/full_native_group_00.npz'
with zipfile.ZipFile(old) as z:v['old_full3_zip']=dict(path=str(old),bytes=old.stat().st_size,all_members=[dict(name=i.filename,uncompressed=i.file_size,compressed=i.compress_size) for i in z.infolist()],no_arrays_recomputed=True)
oldjob=json.loads((B/'Pali10Pt448_A100_original_native_vllm_compat3_1322_v2_20261110/job.json').read_bytes());cf=R/oldjob['spec']['config'];v['old_own_original_config']=dict(path=str(cf),value=json.loads(cf.read_bytes()))
for p in (B/'Pali10Pt448_A100_original_native_vllm_compat3_1322_v2_20261110/original_source_evidence').glob('*fixed200*'):
 if p.stat().st_size<3000000:v['files'][str(p)]=dict(SHA256=sha(p),text=p.read_text())
n=subprocess.run(['npu-smi','info'],text=True,capture_output=True);v['npu_smi']=dict(returncode=n.returncode,stdout=n.stdout,stderr=n.stderr)
print(json.dumps(v,ensure_ascii=False))

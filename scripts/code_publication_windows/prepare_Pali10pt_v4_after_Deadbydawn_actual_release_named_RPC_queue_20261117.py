import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib,base64
P=Path(__file__).parent
source=(P/'continue_Pali10pt_missing_full_numeric_after_current_Donut_real_release_20261117.py').read_text(encoding='utf8')
source=source.replace('actual current Donut','actual current Deadbydawn')
source=source.replace('Pali10Pt448_after_current_Donut_real_lock_then_missing_full_native_value_true3_continuation_20261117','Pali10Pt448_v4_named_RPC_after_current_Deadbydawn_real_lock_then_missing_full_native_value_true3_continuation_20261117')
source=source.replace('DonutInvoice_own_original_FP16_poscap2048_approved_fresh3_fixed200_v4_real_FLOCK_20261117','DeadbydawnGemma4E2B_completed_true3_419_capacity_recovered_approved_fresh3_fixed200_v3_real_FLOCK_20261117')
source=source.replace('Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v3_real_FLOCK_20261117','Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v4_named_worker_RPC_real_FLOCK_20261117')
source=source.replace('9fb844c8f05543fb081c911ae458c3bc02edd98b071688f40802ef16eae4a75e','9ba4f5d991a0b5d3c8cec15b36c7685818d78a8e0e8f0bec835c1c01c74da9c8').replace('1ac3e1549c22e4d364c22288f9b85e613fbc7523725ff9a4684fafa4b91d65fe','b0f36991481658c97bb479ceba2b8ce7958e03978578484742613f0d0377fde9')
source=source.replace('waiting_for_actual_Donut_NN_NF_runner_completion_and_real_FLOCK_release','waiting_for_actual_Deadbydawn_NN_NF_runner_completion_and_real_FLOCK_release').replace('NOTIFY_old_Donut_error_preserved_continue_other_ready_model.json','NOTIFY_old_Deadbydawn_error_preserved_continue_other_ready_model.json')
source=source.replace("unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert batch.device_idle(job['spec'])","unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert batch.device_idle(job['spec']);import shutil;assert shutil.disk_usage(D).free>=12*1024**3")
source=source.replace("phase='running_new_missing_full_native_numeric_three_samples'","phase='running_v4_named_RPC_new_missing_full_native_numeric_three_samples'")
ast.parse(source)
name='scripts/continue_Pali10pt_v4_named_worker_RPC_after_current_Deadbydawn_real_release_20261117.py'
(P/Path(name).name).write_bytes(source.encode())
digest=hashlib.sha256(source.encode()).hexdigest()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';name=__NAME__;raw=base64.b64decode(__PAYLOAD__);assert hashlib.sha256(raw).hexdigest()==__SHA__;ast.parse(raw);f=R/name
if f.exists():assert f.read_bytes()==raw
else:f.write_bytes(raw)
ready=json.loads((B/'Pali10Pt448_named_worker_RPC_v4_source_unit_and_readiness_no_NN_actual_launch_20261117/actual_completion.json').read_bytes());assert ready['actual_returncode']==0 and all(x['actual_returncode']==0 for x in ready['results'])
I=B/'Pali10Pt448_v4_named_worker_RPC_after_Deadbydawn_immediate_queue_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(f)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=str(f),source_SHA256=__SHA__,waiting_controller_started_not_NPU_model_started=True,upstream_current_Deadbydawn_not_modified=True,new_named_RPC_missing_numeric3_only_not_old_compatibility_replay=True,no_G_or_canonical_write=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__NAME__',repr(name)).replace('__PAYLOAD__',repr(base64.b64encode(source.encode()).decode())).replace('__SHA__',repr(digest))
ast.parse(remote)
(P/'launch_Pali10pt_v4_after_Deadbydawn_actual_release_named_RPC_queue_20261117_remote.py').write_bytes(remote.encode())
print(json.dumps(dict(source=name,SHA256=digest)))

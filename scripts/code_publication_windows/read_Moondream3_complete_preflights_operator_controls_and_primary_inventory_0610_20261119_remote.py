"""Exact current-stage source/metadata/protection readback, never model inference."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ptrraw=(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes();ptr=json.loads(ptrraw)
o=dict(UTC=datetime.now(timezone.utc).isoformat(),pointer=ptr,pointer_SHA256=hashlib.sha256(ptrraw).hexdigest(),canonical_SHA256={p:sha(p) for p in ptr['canonical_SHA256']},new_source_inventory=[],stage_metadata={},numeric_control_files={},actual_project_processes=project_processes(),actual_original_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free)
assert o['canonical_SHA256']==ptr['canonical_SHA256']
sources=[p for p in (R/'universal_eval/model/moondream3_own_original_source_20261119').iterdir() if p.suffix=='.py']
sources += [R/'scripts'/n for n in ['prepare_moondream3_own_original_source_index_meta_preflight_0610_20261119.py','prepare_moondream3_own_source_native_BF16_pixels_frontend200_v2_0610_20261119.py','diagnostic_moondream3_own_flex_decode_operator_original_lock_no_model_0610_20261119.py','diagnostic_moondream3_original_causal_flex_NPU_interface_v2_full_mask_control_0610_20261119.py']]
sources += [R/'universal_eval/model/adapters/moondream3_own_original_causal_flex_NPU_interface_v1_0610_20261119.py']
for p in sorted(sources):o['new_source_inventory'].append(dict(path=str(p),relative=str(p.relative_to(R)),SHA256=sha(p),bytes=p.stat().st_size))
homes=['Moondream3_own_original_source_index_meta_preflight_0610_20261119','Moondream3_own_original_source_native_BF16_pixel_frontend200_v2_0610_20261119','Moondream3_own_original_flex_decode_operator_diagnostic_no_model_real_FLOCK_0610_20261119','Moondream3_original_causal_flex_NPU_interface_v2_full_mask_operator_control_0610_20261119']
for name in homes:
 h=B/name;metadata={}
 for p in sorted(h.iterdir()):
  if p.suffix in ['.json','.jsonl']:
   b=p.read_bytes();entry=dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b))
   if p.suffix=='.json':entry['value']=json.loads(b)
   else:entry.update(lines=len(b.splitlines()),all_rows_complete=b.endswith(b'\n'))
   metadata[p.name]=entry
  elif p.suffix=='.npz':o['numeric_control_files'][str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p),all_original_numeric_control_values_retained=True)
 o['stage_metadata'][name]=metadata
n=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);o['npu_smi']=dict(returncode=n.returncode,stdout=n.stdout,stderr=n.stderr)
o['resource_waiter_state']=json.loads((B/'GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118/state.json').read_bytes())
print(json.dumps(o,ensure_ascii=False))

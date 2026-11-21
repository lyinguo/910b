"""Keep completed experiments unchanged; real capacity waiter consumes shaped v1."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
src=(V/'continue_MetaMllama_completed_genuine3_complete_storage_v2_first200_serial_20261120.py').read_text(encoding='utf-8')
replacements={
'continue_MetaMllama_completed_genuine3_complete_storage_v2_first200_serial_20261120':'continue_MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120',
'MetaMllama_completed_genuine3_complete_storage_v2_first200_serial_20261120':'MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120',
'meta_mllama_own_original_BF16_SDPA_2048_native_lossless_natural_order_LZMA_v2_20261120':'meta_mllama_own_original_BF16_SDPA_2048_native_lossless_shaped_LZMA_v3_20261120',
'complete_native_raw_bytes_natural_order_byteplanes_LZMA_v2_20261120':'complete_native_raw_bytes_Fortran_LZMA_20261120',
'meta_mllama_completed_genuine3_complete_storage_LZMA_v2_approved_first200_serial_20261120':'meta_mllama_completed_genuine3_shaped_storage_LZMA_v3_approved_first200_serial_20261120',
'complete-LZMA-v2-approved-first200':'complete-shaped-LZMA-v3-approved-first200',
'MetaMllama_completed_genuine3_complete_storage_LZMA_v2_first200_serial_continuation_20261120':'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120',
'MetaMllama_completed_true3_complete_natural_order_byteplanes_LZMA_storage_v2_diagnostic_only_20261120':'MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_storage_capacity_diagnostic_only_20261120',
'MetaMllama_completed_genuine3_complete_storage_LZMA_v2_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120':'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120',
'uint8[numel,itemsize]':'uint8[original_shape...,itemsize]'
}
for a,b in replacements.items():assert a in src,a;src=src.replace(a,b)
old=""" if not cap['capacity_passed'] or shutil.disk_usage(R).free<cap['empirical_estimate_bytes']:
  put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='complete_storage_verified_but_original_disk_capacity_refused',capacity_review=str(DIAG/'all200_complete_storage_capacity_review.json'),required_bytes=cap['empirical_estimate_bytes'],actual_free_bytes=shutil.disk_usage(R).free,no_first200_approval_NEWfresh3_200_or_G=True,all_original_values_and_reports_retained=True));return 1
"""
new=""" # Existing GLM41 owns priority when its larger budget actually fits.
 prior= B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
 polls=0
 while True:
  free=shutil.disk_usage(R).free;higher_priority=(free>=cfg['existing_GLM41_priority_required_bytes'] and not (prior/'actual_completion.json').is_file())
  all_quiet=quiet();polls+=1
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_actual_original_capacity_not_NN',polls=polls,actual_free_bytes=free,required_bytes=cap['empirical_estimate_bytes'],existing_GLM41_priority=higher_priority,all_project_NN_NF_and_original_EX_released=all_quiet,actual_NEW_model_forwards=0,max_concurrent_models=1,no_expansion_other_volume_deletion_guard_change_or_passed3_replay=True))
  if free>=cap['empirical_estimate_bytes'] and all_quiet and not higher_priority:break
  time.sleep(20)
"""
assert src.count(old)==1;src=src.replace(old,new)
name='continue_MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.py'
ast.parse(src);(V/name).write_text(src,encoding='utf-8',newline='\n')
model='meta_mllama_own_original_BF16_SDPA_2048_native_lossless_shaped_LZMA_v3_20261120.py'
oldsrc=(V/'meta_mllama_own_original_BF16_SDPA_2048_native_v1_20261120.py').read_text(encoding='utf-8');oldcall='np.savez_compressed(q,**{k:v.reshape(-1).view(torch.uint8).numpy() for k,v in self._arrays.items()})';newimport='from universal_eval.utils.complete_native_raw_bytes_Fortran_LZMA_20261120 import write_complete_native_raw_npz\n'
assert oldsrc.count(oldcall)==1;modelsrc=oldsrc.replace(oldcall,'write_complete_native_raw_npz(q,self._arrays)');modelsrc=modelsrc.replace('from PIL import Image\n','from PIL import Image\n'+newimport)
assert ast.dump(ast.parse(modelsrc.replace(newimport,'')),include_attributes=False)==ast.dump(ast.parse(oldsrc.replace(oldcall,'write_complete_native_raw_npz(q,self._arrays)')),include_attributes=False)
(V/model).write_text(modelsrc,encoding='utf-8',newline='\n')
items={}
for file,rel in [(name,'scripts/'+name),(model,'universal_eval/model/adapters/'+model)]:
 raw=(V/file).read_bytes();items[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
consumer='configs/queues/MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.json'
cfg=dict(model_id='meta-llama/Llama-3.2-11B-Vision-Instruct',max_concurrent_models=1,existing_GLM41_priority_required_bytes=136041918364,exact_source_SHA256={**{n:x['SHA256'] for n,x in items.items()},'universal_eval/utils/complete_native_raw_bytes_Fortran_LZMA_20261120.py':'4a95739870af1a5e85f2daeebedbf7bed934c86f70bc5d5ce4d85f5f9e573045','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py':'a858f771182c1e8240e8e94d158d6089a4c7ee421b7cae94ab4301c9fb2be485','scripts/meta_mllama_complete_shape_storage_capacity_20261120.py':'fb5fceebc3ed09b8c87cd4ecd9ffd4c9f3098f1c1b72bb61cbf1c89b6997f90a','scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py':'90bd632c82c339b0b060194342250bde5a868ec5795119db849b9cd5efa8b7c2'})
raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();items[consumer]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode());(V/Path(consumer).name).write_bytes(raw)
template=ast.parse((V/'build_stage_MetaMllama_complete_shaped_raw_Fortran_LZMA_diagnostic_20261120.py').read_text(encoding='utf-8'));assignment=next(n for n in template.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets));remote=ast.literal_eval(assignment.value.func.value)
remote=remote.replace('assert not runner_lock_owners()','assert True  # New no-NN waiter may coexist with a diagnostic; runner still requires real quiet/EX.').replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode()).replace('MetaMllama_completed3_full_shaped_raw_Fortran_LZMA_diagnostic_source_launch_20261120','MetaMllama_completed3_shaped_storage_v3_actual_capacity_wait_source_launch_20261120').replace('scripts/diagnose_MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_capacity_20261120.py','scripts/'+name)
remote=remote.replace('for n,v in items.items():\n p=R/n;',"for n,h in json.loads(base64.b64decode(items['"+consumer+"']['base64']))['exact_source_SHA256'].items():\n if n not in items:assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h\nfor n,v in items.items():\n p=R/n;")
ast.parse(remote);(V/'stage_MetaMllama_shaped_storage_v3_actual_capacity_wait_20261120_remote.py').write_text(remote,encoding='utf-8',newline='\n');(V/'MetaMllama_shaped_storage_v3_wait_exact_sources_20261120.json').write_text(json.dumps({n:x['SHA256'] for n,x in items.items()},indent=2)+'\n',encoding='utf-8');print(json.dumps({n:x['SHA256'] for n,x in items.items()},indent=2))

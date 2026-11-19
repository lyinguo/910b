"""Close exact official readback with actual completed no-forward stages."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
archive=read('R_archive_verified180_Moondream_loading_KV_v3_actual_1145_20261119.json');base=read('archive_verified180_Moondream_loading_KV_round_plan_1138_20261119.json');fresh=read('R_Moondream3_own_actual_loading_runtime_KV_candidate_inventory_1135_20261119.json');official=read('official910b_Moondream1106_exact_readback_20261119.json')
R=base['project_root'];D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D+'/code_publication_source_layout_20261114'
J=B+'/HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118';HC=B+'/HuihuiGLM46Flash_own_original_BF16_2048_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261118'
states={'last_formal_Huihui/completion.json':J+'/completion.json','last_formal_Huihui/capacity_actual_completion.json':HC+'/actual_completion.json','GLM_original_resource_waiter/state.json':B+'/GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118/state.json'}
for home,values in fresh['stage_metadata'].items():
 for name,v in values.items():states[Path(home).name+'/'+name]=v['path']
names=['official910b_Moondream1106_update_tool_result_20261119.json','official910b_Moondream1106_exact_readback_20261119.json','automation910b_after_Moondream1106_20261119.toml','automation910b_after_Moondream1106_20261119_prompt.txt','prompt_official910b_Moondream_loading_KV_current_1106_20261119.txt','build_official910b_Moondream_actual_loading_KV_short_reference_1148_20261119.py','verify_official910b_Moondream1106_exact_readback_20261119.py',Path(__file__).name]
packets=[]
for name in names:
 b=(P/name).read_bytes();packets.append(dict(name=name,bytes=len(b),SHA256=sha(b),base64=base64.b64encode(b).decode()))
codes=[]
for name in ['build_official910b_Moondream_actual_loading_KV_short_reference_1148_20261119.py','verify_official910b_Moondream1106_exact_readback_20261119.py',Path(__file__).name]:
 code=(P/name).read_bytes();ast.parse(code);codes.append(dict(relative='scripts/code_publication_windows/'+name,SHA256=sha(code),base64=base64.b64encode(code).decode(),role='Actual existing official heartbeat prompt/readback/source-stage closure; preserves name/thread/ACTIVE/300min/notification and canonical, no model math'))
plan=dict(handoff_directory=archive['handoff_directory'],expected_pointer_SHA256=archive['pointer_SHA256'],canonical_SHA256=base['canonical_SHA256'],note_name='official_readback_Moondream_all662_actual_NPU_BF16_loading_KV_candidate_no_model_NN_1106_20261119',local_original_packets=packets,official_readback=official,additional_existing_primary_sources=[{k:v for k,v in item.items() if k!='base64'} for item in codes],process_keyword='Moondream3',explicit_additional_PIDs=[1530341],current_stage_files=states,completed_owned_job_key='last_formal_Huihui/completion.json',completed_owned_capacity_key='last_formal_Huihui/capacity_actual_completion.json',new_pending=C+'/pending_after_daily_20261119_verified180_Moondream_actual_NPU_loading_KV_candidate_official_current_all_primary_v35.json',reviewed_monitor_inspection_context='Fresh complete frozen monitor source/inspection/new-log errors empty and pending log bytes zero. Own CPU load v1 JSON-set failure retained; new v2 and own NPU all662 load audits passed with zero learned forwards. Candidate KV storage local test and hook are not model NN or fullNF, not formal recovery. Original wrong 7/8 unit assertion and archive copy-path / bounded-payload guard failures retained; canonical/source original objects unchanged. Concurrent selector saw only the known monitor reader process; unknown new errors not exempted.')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'official_Moondream1106_loading_KV_source_stage_closure_plan_20261119.json').write_bytes(raw)
data=dict(primary_sources=codes,plan_SHA256=sha(raw),plan_base64=base64.b64encode(raw).decode())
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for item in DATA['primary_sources']:
 code=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(code).hexdigest()==item['SHA256'];ast.parse(code);p=R/item['relative'];assert not p.exists();p.write_bytes(code)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw);H=Path(plan['handoff_directory']);p=H/'official_current_Moondream_loading_KV_source_stage_closure_plan_original.json';assert not p.exists();p.write_bytes(raw);sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_completed_owned_capacity_v5_20261118');print(json.dumps(m.execute(p,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_close_official910b_Moondream_loading_KV_current_1152_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),new_primary_sources=len(codes),new_pending=plan['new_pending']),ensure_ascii=False))

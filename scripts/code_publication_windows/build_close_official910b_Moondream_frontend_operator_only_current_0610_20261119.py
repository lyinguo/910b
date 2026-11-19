"""Archive official readback and exact completed original/diagnostic stages."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
archive=read('R_archive_verified180_Moondream_current_round_actual_0610_20261119.json');base=read('archive_verified180_Moondream_current_round_plan_0610_20261119.json');fresh=read('R_Moondream3_current_preflight_operator_primary_inventory_actual_0610_20261119.json');official=read('official910b_Moondream0610_exact_readback_20261119.json')
R=base['project_root'];D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D+'/code_publication_source_layout_20261114'
J=B+'/HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118';HC=B+'/HuihuiGLM46Flash_own_original_BF16_2048_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261118'
states={'last_formal_Huihui/completion.json':J+'/completion.json','last_formal_Huihui/capacity_actual_completion.json':HC+'/actual_completion.json','GLM_original_resource_waiter/state.json':B+'/GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118/state.json'}
for home,values in fresh['stage_metadata'].items():
 for name,v in values.items():
  if name in ['actual_completion.json','review.json','actual_original_HF_class_meta_precision_and_aliases.json']:states[home+'/'+name]=v['path']
names=['official910b_Moondream0610_update_tool_result_20261119.json','official910b_Moondream0610_exact_readback_20261119.json','automation910b_after_Moondream0610_20261119.toml','automation910b_after_Moondream0610_20261119_prompt.txt','prompt_official910b_Moondream_current_0610_20261119.txt',Path(__file__).name]
packets=[]
for name in names:
 b=(P/name).read_bytes();packets.append(dict(name=name,bytes=len(b),SHA256=sha(b),base64=base64.b64encode(b).decode()))
code=Path(__file__).read_bytes();ast.parse(code);relative='scripts/code_publication_windows/'+Path(__file__).name
plan=dict(handoff_directory=archive['handoff_directory'],expected_pointer_SHA256=archive['pointer_SHA256'],canonical_SHA256=base['canonical_SHA256'],note_name='official_readback_Moondream_frontend200_operator5_no_model_NN_current_0610_20261119',local_original_packets=packets,official_readback=official,additional_existing_primary_sources=[dict(relative=relative,SHA256=sha(code),role='Actual official preserved-setting readback and source/current diagnostic-stage closure, no model or canonical writes')],process_keyword='Moondream3',explicit_additional_PIDs=[1530341],current_stage_files=states,completed_owned_job_key='last_formal_Huihui/completion.json',completed_owned_capacity_key='last_formal_Huihui/capacity_actual_completion.json',new_pending=C+'/pending_after_daily_20261119_verified180_Moondream_frontend_operator_original_failure_official_current_all_primary_v33.json',reviewed_monitor_inspection_context='Current full monitor zero source/inspection/log errors. Original unsupported NPU Flex diagnostic failure retained, later private operator control is diagnostic only with zero model forwards; all old formal failures retained. No operator/meta/frontend activity counts as model NN/fullNF or formal recovery.')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'official_Moondream0610_source_and_stage_closure_plan_20261119.json').write_bytes(raw)
data=dict(source_relative=relative,source_SHA256=sha(code),source_base64=base64.b64encode(code).decode(),plan_SHA256=sha(raw),plan_base64=base64.b64encode(raw).decode())
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror');code=base64.b64decode(DATA['source_base64'],validate=True);assert hashlib.sha256(code).hexdigest()==DATA['source_SHA256'];ast.parse(code);p=R/DATA['source_relative'];assert not p.exists();p.write_bytes(code)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw);H=Path(plan['handoff_directory']);p=H/'official_current_Moondream_source_stage_closure_plan_original.json';assert not p.exists();p.write_bytes(raw);sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_completed_owned_capacity_v5_20261118');print(json.dumps(m.execute(p,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_close_official910b_Moondream_frontend_operator_current_0610_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),source_SHA256=sha(code),new_pending=plan['new_pending']),ensure_ascii=False))

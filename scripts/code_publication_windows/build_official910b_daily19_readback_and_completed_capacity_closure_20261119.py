"""Bind the official readback to existing completed-stage proof and fresh real processes."""
from pathlib import Path
import ast, base64, hashlib, json
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda n:json.loads((P/n).read_bytes())
archive=read('R_archive_daily19_verified180_complete_current_actual_20261119.json')
base=read('archive_daily19_verified180_complete_current_plan_20261119.json')
readback=read('official910b_daily19_exact_readback_20261119.json')
assert readback['passed'] and readback['before_preserved_fields']==readback['after_preserved_fields']
H=archive['handoff_directory'];R=base['project_root'];D=R+'/workspace/diagnostics/batch_910b';C=D+'/code_publication_source_layout_20261114';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
packets=[]
for name in ['official910b_daily19_update_tool_result_20261119.json','official910b_daily19_exact_readback_20261119.json',
             'automation910b_after_daily19_preserved.toml','automation910b_after_daily19_preserved_prompt.txt',
             'prompt_official910b_daily19_current_20261119.txt',Path(__file__).name]:
    raw=(P/name).read_bytes();packets.append(dict(name=name,bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
J=B+'/HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118'
HC=B+'/HuihuiGLM46Flash_own_original_BF16_2048_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261118'
wait=B+'/GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118'
states={'Huihui_J/completion.json':J+'/completion.json','Huihui_J/state.json':J+'/state.json',
        'Huihui_J/full_NF_stdout.json':J+'/fresh3/verification_stdout.json', 'Huihui_J/full_NF_stderr.log':J+'/fresh3/verification_stderr.log',
        'Huihui_cap/actual_completion.json':HC+'/actual_completion.json','Huihui_cap/complete_capacity_review.json':HC+'/full_shape_complete_numeric_storage_capacity_review.json',
        'GLM_original_resource_waiter/state.json':wait+'/state.json','GLM_original_resource_waiter/actual_completion.json':wait+'/actual_completion.json'}
code=Path(__file__).read_bytes();ast.parse(code);relative='scripts/code_publication_windows/'+Path(__file__).name
plan=dict(handoff_directory=H,expected_pointer_SHA256=archive['pointer_SHA256'],canonical_SHA256=base['canonical_SHA256'],
    note_name='official_readback_Day19_push_completed_Huihui_fullNF0_capacity_current_20261119',local_original_packets=packets,
    official_readback=readback,additional_existing_primary_sources=[dict(relative=relative,SHA256=sha(code),role='Actual official Day19 preserved-setting readback and completed-stage/proc/source closure; no model or canonical writes')],
    process_keyword='GLM46Flash',explicit_additional_PIDs=[1530341],current_stage_files=states,
    completed_owned_job_key='Huihui_J/completion.json',completed_owned_capacity_key='Huihui_cap/actual_completion.json',
    new_pending=C+'/pending_after_daily_20261119_verified180_code_pushed_official_current_capacity_all_primary_v31.json',
    reviewed_monitor_inspection_context='The specific prior 20:46 runner cwd=/root filter gap is preserved and closed by genuine same-job completion/NF0 and later zero-error monitors. Day19 full monitor source_errors/inspection_errors/newlog all empty. Unknown new errors are not exempted.')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'official_daily19_completed_capacity_closure_plan_20261119.json').write_bytes(raw)
data=dict(code_relative=relative,code_SHA256=sha(code),code_base64=base64.b64encode(code).decode(),plan_SHA256=sha(raw),plan_base64=base64.b64encode(raw).decode())
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror');code=base64.b64decode(DATA['code_base64'],validate=True);assert hashlib.sha256(code).hexdigest()==DATA['code_SHA256'];ast.parse(code);p=R/DATA['code_relative'];assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(code)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw);H=Path(plan['handoff_directory']);p=H/'official_Day19_completed_capacity_closure_plan_original.json';assert not p.exists();p.write_bytes(raw)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_completed_owned_capacity_v5_20261118')
print(json.dumps(m.execute(p,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_official_daily19_completed_capacity_closure_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),source_SHA256=sha(code),new_pending=plan['new_pending']),ensure_ascii=False))

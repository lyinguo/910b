"""Prepare an explicit final source/readback evidence closure; no model change."""
from pathlib import Path
import ast
import base64
import hashlib
import json

P=Path('D:/code-lv/.tmp/batch_910b_20261030')
R='/root/projects/comperror'
D=R+'/workspace/diagnostics/batch_910b'
C=D+'/code_publication_source_layout_20261114'
initial=json.loads((P/'R_heartbeat0020_verified180_Day18_full_handoff_archive_actual_20261118.json').read_bytes())
readback_raw=(P/'automation910b_verified180_Day18_official_readback_review_20261118.json').read_bytes()
readback=json.loads(readback_raw)
assert readback['passed'] and readback['all_non_prompt_updated_at_fields_preserved']
selector_raw=(P/'R_heartbeat0020_frozen_original_resource_selector_current_20261118.json').read_bytes()
selector=json.loads(selector_raw)
assert selector['dispatch'] is None and not selector['current_actual_project_processes'] and not selector['kernel_FLOCK_owners']

raw=(P/'finalize_alignment_handoff_official_readback_20261117.py').read_text(encoding='utf-8')
raw=raw.replace('official_readback_source_closure_current_actual_20261117','official_readback_source_closure_current_actual_20261118')
raw=raw.replace("        official_update_readback=plan['official_readback'],", "        official_update_readback=plan['official_readback'],\n        frozen_original_resource_selector_current=plan['frozen_original_resource_selector_current'],\n        no_matrix_deletion_this_cycle=True,\n        Day18_already_pushed_commit='02df1887796f49c830c63db01f4dcdb2bb8e4298',")
utility=P/'finalize_alignment_handoff_official_readback_20261118.py'
assert not utility.exists()
ast.parse(raw)
utility.write_bytes(raw.encode('utf-8'))

def row(relative,path,role):
 data=Path(path).read_bytes();ast.parse(data.decode('utf-8-sig'))
 return dict(relative=relative,SHA256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode(),role=role)
sources=[row('scripts/maintenance/'+utility.name,utility,'Official tool exact readback and full primary source closure, not evaluation code')]
for filename in [
 'prepare_official_verified180_Day18_short_handoff_update_20261118.py',
 'verify_automation910b_Day18_official_readback_all_fields_20261118.py',
 'deploy_execute_verified180_Day18_handoff_builder_20261118.py',
 'check_frozen_original_resource_selector_after_Phi35_inputs_20261118_remote.py',
 'build_verified180_Day18_official_final_closure_20261118.py',
]: sources.append(row('scripts/code_publication_windows/'+filename,P/filename,'Executed source/readback/strict resource selector maintenance helper; no canonical evaluation write'))

artifacts=[]
names=[
 'R_heartbeat0020_verified180_Day18_full_handoff_archive_actual_20261118.json',
 'R_heartbeat0020_verified180_Day18_full_handoff_archive_actual_20261118.json.stderr',
 'automation910b_verified180_Day18_official_update_args_20261118.json',
 'automation910b_verified180_Day18_official_tool_receipt_20261118.json',
 'automation910b_verified180_Day18_official_readback_review_20261118.json',
 'automation910b_after_Day18_verified_official_update_20261118.toml',
 'automation910b_after_Day18_verified_official_update_20261118.prompt.txt',
 'R_heartbeat0020_frozen_original_resource_selector_current_20261118.json',
 'R_heartbeat0020_frozen_original_resource_selector_current_20261118.json.stderr',
 'verified180_Day18_handoff_prepared_metadata_20261118.json',
 'deploy_execute_verified180_Day18_handoff_builder_20261118.py',
]
for name in names:
 path=P/name;data=path.read_bytes()
 artifacts.append(dict(path=str(path),archive_name=name,SHA256=hashlib.sha256(data).hexdigest(),bytes=len(data),base64=base64.b64encode(data).decode()))
plan=dict(project_root=R,handoff_directory=initial['handoff_directory'],pointer_SHA256=initial['pointer_SHA256'],new_primary_sources=sources,
 local_original_artifacts=artifacts,official_readback=readback,official_readback_SHA256=hashlib.sha256(readback_raw).hexdigest(),frozen_original_resource_selector_current=selector,
 new_pending=C+'/pending_after_daily_20261118_verified180_Phi35_Ovis_official_full_source_readback_closure_v24.json')
data=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode()
plan_path=P/'verified180_Day18_official_final_closure_explicit_inputs_20261118.json'
plan_path.write_bytes(data)
payload='PLAN='+repr(base64.b64encode(data).decode())+'\nUTILITY='+repr(base64.b64encode(utility.read_bytes()).decode())+'\n'+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,json,runpy
data=base64.b64decode(PLAN,validate=True)
assert hashlib.sha256(data).hexdigest()=='PLAN_SHA'
p=json.loads(data);src=Path(p['project_root'])/'scripts/maintenance/finalize_alignment_handoff_official_readback_20261118.py'
source=base64.b64decode(UTILITY,validate=True)
if src.exists():assert src.read_bytes()==source
else:src.write_bytes(source)
path=Path(p['handoff_directory'])/'official_readback_final_closure_plan.json'
assert not path.exists();path.write_bytes(data)
sys.argv=[str(src),'--plan',str(path),'--plan-sha256','PLAN_SHA']
runpy.run_path(str(src),run_name='__main__')
'''.replace('PLAN_SHA',hashlib.sha256(data).hexdigest())
target=P/'execute_verified180_Day18_official_final_source_closure_exact_payload_20261118_remote.py'
target.write_bytes(payload.encode('utf-8'))
print(json.dumps(dict(plan_SHA256=hashlib.sha256(data).hexdigest(),new_primary_sources=len(sources),actual_original_packets=len(artifacts),payload_bytes=target.stat().st_size)))

from pathlib import Path
import base64
import hashlib
import json

P=Path('D:/code-lv/.tmp/batch_910b_20261030')
meta=json.loads((P/'verified180_Day18_handoff_prepared_metadata_20261118.json').read_bytes())
plan=(P/'verified180_Day18_Phi35_inputs_Ovis_capacity_handoff_plan_20261118.json').read_bytes()
assert hashlib.sha256(plan).hexdigest()==meta['SHA256']
source=(P/'archive_alignment_reviewed_daily_transition_primary_handoff_20261118.py').read_bytes()
code='PLAN_RAW='+repr(base64.b64encode(plan).decode())+'\nSOURCE_RAW='+repr(base64.b64encode(source).decode())+'\n'+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,json,runpy
raw=base64.b64decode(PLAN_RAW,validate=True)
assert hashlib.sha256(raw).hexdigest()=='PLAN_SHA'
plan=json.loads(raw)
home=Path(plan['handoff_directory']);home.mkdir(parents=True,exist_ok=False)
source=base64.b64decode(SOURCE_RAW,validate=True)
src=Path(plan['project_root'])/'scripts/maintenance/archive_alignment_reviewed_daily_transition_primary_handoff_20261118.py'
if src.exists():assert src.read_bytes()==source
else:src.write_bytes(source)
pf=home/'archive_reviewed_plan.json';pf.write_bytes(raw)
sys.argv=[str(src),'--plan',str(pf),'--plan-sha256','PLAN_SHA']
runpy.run_path(str(src),run_name='__main__')
'''.replace('PLAN_SHA',meta['SHA256'])
out=P/'execute_verified180_Day18_handoff_snapshot_exact_plan_20261118_remote.py'
out.write_bytes(code.encode('utf-8'))
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,scope='Payload receipt only; executing exact reviewed source snapshot utility, no evaluation or workbook change')))

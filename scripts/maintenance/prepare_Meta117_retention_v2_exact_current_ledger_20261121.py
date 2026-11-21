"""Correct only the unstarted retention config's stale pre-existing ledger pin."""
from pathlib import Path
import json,hashlib,base64,ast
V=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
old='retain_verified184_Meta_fullNF4_correct_nonexample117_Bee_capacity_20261121.py';new='retain_verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_20261121.py'
cfgold='verified184_Meta_fullNF4_correct_nonexample117_user_Bee_capacity_20261121.json';cfgnew='verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_20261121.json'
s=(V/old).read_text(encoding='utf8').replace('verified184_Meta_fullNF4_correct_nonexample117_user_Bee_capacity_recovery_20261121','verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_recovery_20261121').replace(cfgold,cfgnew);ast.parse(s);(V/new).write_text(s,encoding='utf8',newline='\n')
q=json.loads((V/cfgold).read_bytes());p='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json';assert q['canonical_SHA256'][p]=='4ba1a4c2a3171fa17b1e4fa9e997addf6d40e2210eaebf92705d84b712f728d3';q['canonical_SHA256'][p]=q['ledger_before_SHA256'];assert q['canonical_SHA256'][p]=='6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c';(V/cfgnew).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
records=[dict(path='scripts/maintenance/'+new,SHA256=sha((V/new).read_bytes()),base64=base64.b64encode((V/new).read_bytes()).decode()),dict(path='configs/retention/'+cfgnew,SHA256=sha((V/cfgnew).read_bytes()),base64=base64.b64encode((V/cfgnew).read_bytes()).decode())]
t='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,base64,json,importlib.util
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not p.exists()
 with p.open('xb') as f:f.write(b)
p=R/DATA[0]['path'];sp=importlib.util.spec_from_file_location('Meta117_v2_retention_plan',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.H.mkdir(exist_ok=False);print(json.dumps(m.plan(),ensure_ascii=False))
'''
ast.parse(t);(V/'plan_Meta117_retention_v2_exact_current_ledger_20261121_remote.py').write_text(t,encoding='utf8',newline='\n');print(json.dumps({'prepared':True,'source_SHA256':records[0]['SHA256'],'config_SHA256':records[1]['SHA256'],'only_exact_already_authorized_ledger_pin_updated':True,'no_deletion_yet':True}))

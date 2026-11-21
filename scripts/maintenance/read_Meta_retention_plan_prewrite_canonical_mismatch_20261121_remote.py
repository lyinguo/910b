import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');p=R/'configs/retention/verified184_Meta_fullNF4_correct_nonexample117_user_Bee_capacity_20261121.json';q=json.loads(p.read_bytes());bad=[]
for n,h in q['canonical_SHA256'].items():
 a=hashlib.sha256(Path(n).read_bytes()).hexdigest()
 if a!=h:bad.append({'path':n,'expected':h,'actual':a})
print(json.dumps({'differences':bad,'no_deletion':True}))
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,importlib.util
R=Path('/root/projects/comperror')
sys.path[:0]=[str(R),str(R/'scripts')]
p=R/'scripts/maintenance/retain_verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_20261121.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='3e3b844efcc251e0881ac3661c8dfedced0ca2bba36edc49e90b91eb9aef6dc4'
sp=importlib.util.spec_from_file_location('apply_Meta117_v3_reviewed',p)
m=importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
reviewed='a5a2f5e6564db2d3d1ead96867cf0b17ff5b2e0e760ce45c57469aa503d10c1f'
assert hashlib.sha256((m.H/'plan.json').read_bytes()).hexdigest()==reviewed
plan=json.loads((m.H/'plan.json').read_bytes())
assert plan['remove_count']==117 and plan['remove_bytes']==15312491122 and plan['protected_count']==10790
assert len(plan['selection']['incorrect'])==59 and len(plan['selection']['keep'])==83 and not plan['foreign_references_preserved']
print(json.dumps(m.apply(reviewed),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
P=Path(__file__).resolve().parent;r=json.loads((P/'R_Huihui_reviewed_measurement_retention_plan_20261119.json').read_bytes())
assert r['remove_count']==105 and r['remove_bytes']==10617515541 and r['preserved_wrong_count']==61 and r['preserved_numeric_cases']==95 and r['foreign_files']==0
text='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from maintenance import retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119 as m
assert m.sha(R/'scripts/maintenance/retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py')=='5120db9f35af929f22b01610b7d27353322cdac901e9acc479292c006af9387b'
print(json.dumps(m.apply(REVIEWED_SHA),ensure_ascii=False))
'''.replace('REVIEWED_SHA',repr(r['plan_SHA256']))
(P/'apply_Huihui_exact_reviewed_correct_nonexample_retention_20261119_remote.py').write_text(text,encoding='utf8')
print(json.dumps(dict(plan_SHA256=r['plan_SHA256'],reviewed_correct_only_count=105,preserved_full_wrong_cases=61,ready=True)))

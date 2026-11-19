import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from maintenance import retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119 as m
assert m.sha(R/'scripts/maintenance/retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py')=='5120db9f35af929f22b01610b7d27353322cdac901e9acc479292c006af9387b'
print(json.dumps(m.apply('714b85a5ca1cd82a4d8d96d4831cec892417c9713fac1f2caa13e1402e0bac7d'),ensure_ascii=False))

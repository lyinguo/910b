import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');names=['universal_eval/model/prithiv_gemma4_e4b_own_20261116/observer.py','scripts/verify_prithiv_gemma4_e4b_own_original_BF16_native_v2_exact_own_report_path_20261116.py','scripts/eager_corrective_control_safety_2203.py'];o={}
for n in names:
 p=R/n;o[n]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(o,ensure_ascii=False))

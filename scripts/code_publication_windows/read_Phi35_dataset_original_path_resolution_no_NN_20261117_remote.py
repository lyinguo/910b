import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');f=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';row=json.loads(f.read_text().splitlines()[0]);o={'original_first_dataset_row':row,'loader_sources':{}}
for n in ['universal_eval/data/loader.py','universal_eval/data/preprocessor.py','universal_eval/algorithm/inference.py']:
 p=R/n;raw=p.read_bytes();o['loader_sources'][n]={'SHA256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
print(json.dumps(o,ensure_ascii=False))

import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
R = Path('/root/projects/comperror')
files = sorted(str(p) for root in ['scripts','universal_eval','configs'] for p in (R/root).rglob('*') if p.is_file() and 'ovisu1' in p.name.lower().replace('_','').replace('-',''))
print(json.dumps({'paths': files, 'read_only_filename_inventory': True}, ensure_ascii=False))

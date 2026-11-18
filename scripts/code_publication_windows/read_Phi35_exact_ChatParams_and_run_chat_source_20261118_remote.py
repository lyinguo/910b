import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
p=Path('/vllm-workspace/vllm/vllm/renderers/params.py')
s=p.read_text()
a=Path('/vllm-workspace/vllm/vllm/entrypoints/offline_utils.py').read_text()
i=a.index('    def _run_chat(')
print(json.dumps(dict(params_source=s,params_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),run_chat_source=a[i:i+10000])))
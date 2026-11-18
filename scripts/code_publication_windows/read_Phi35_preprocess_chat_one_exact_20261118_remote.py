import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
s=Path('/vllm-workspace/vllm/vllm/entrypoints/offline_utils.py').read_text(); i=s.index('    def _preprocess_chat_one(')
print(json.dumps(dict(source=s[i:i+6000])))
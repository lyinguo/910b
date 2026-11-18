import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json

base = Path(importlib.util.find_spec('vllm').origin).parent
out = {'UTC': datetime.now(timezone.utc).isoformat(), 'source_only_no_model_initialization': True, 'files': {}}
paths = [base / n for n in ('entrypoints/llm.py', 'renderers/registry.py', 'renderers/hf.py', 'renderers/base.py', 'sampling_params.py', 'multimodal/registry.py', 'multimodal/processing.py', 'model_executor/models/phi3v.py')]
for path in paths:
    if not path.is_file():
        continue
    data = path.read_bytes()
    text = data.decode('utf-8')
    selected = text
    if path.name in ('llm.py', 'sampling_params.py', 'registry.py', 'processing.py'):
        tree = ast.parse(text)
        names = {'chat', '__init__', 'create_renderer', 'renderer_from_config', 'render_chat', 'process', 'update_from_generation_config', '_init_renderer', 'create_processor'}
        segments = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                segments.append(ast.get_source_segment(text, node))
        selected = text[:5000] + '\n\n' + '\n\n'.join(segments)
    out['files'][str(path)] = {'SHA256': hashlib.sha256(data).hexdigest(), 'source': selected}
print(json.dumps(out, ensure_ascii=False))

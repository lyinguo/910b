import sys
sys.dont_write_bytecode = True
from pathlib import Path
import ast
import hashlib
import importlib.util
import json

R = Path('/root/projects/comperror')
B = R / 'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B / 'Phi35_own_actual_native_renderer_stop_contract_no_NN_v1_20261118'
review = json.loads((H / 'review.json').read_bytes())
native = []
for row in review.pop('native_first3_renderer_records'):
    native.append({k: v for k, v in row.items() if k != 'native_prompt_token_ids'} | {'native_token_count': len(row['native_prompt_token_ids']), 'native_first_ids': row['native_prompt_token_ids'][:10]})
base = Path(importlib.util.find_spec('vllm').origin).parent
paths = [base / 'v1/engine/input_processor.py', base / 'config/model.py', base / 'engine/arg_utils.py']
sources = {}
for path in paths:
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    tree = ast.parse(text)
    names = {'_get_and_verify_max_len', 'get_diff_sampling_param', 'get_generation_config', 'get_eos_token_id', '_get_eos_token_id', '_get_tokenizer_eos_token_id', '_validate_sampling_params', '_validate_model_input', 'process_inputs'}
    selected = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name in names or (path.name == 'input_processor.py' and node.name == '__init__')):
            selected.append(ast.get_source_segment(text, node))
    sources[str(path)] = {'SHA256': hashlib.sha256(raw).hexdigest(), 'source': '\n\n'.join(selected)}
print(json.dumps(dict(review=review, native_first3=native, saved_own=json.loads((H / 'saved_own_deployment_files.json').read_bytes()), exact_sources=sources), ensure_ascii=False))

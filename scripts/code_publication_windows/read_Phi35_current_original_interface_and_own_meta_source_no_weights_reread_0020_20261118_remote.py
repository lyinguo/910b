import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import hashlib
import struct

R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
state = json.loads((D / 'state.json').read_bytes())['models']
mid = 'microsoft/Phi-3.5-vision-instruct'
W = Path(state[mid]['weight_paths'][0])
paths = [R / name for name in (
 'universal_eval/model/phi35_own_original_source_20261117/configuration_phi3_v.py',
 'universal_eval/model/phi35_own_original_source_20261117/modeling_phi3_v.py',
 'universal_eval/model/phi35_own_original_source_20261117/processing_phi3_v.py',
 'scripts/prepare_phi35_own_frozen_source_meta_interface_preflight_v2_exact_frontend_no_NN_1916_20261117.py',
 'scripts/prepare_phi35_own_original_BF16_4096_HD_frontend200_v3_original_dataset_prompt_no_NN_20261117.py',
 'universal_eval/model/adapters/osatlas4b_own_original_mcscore_bf16_v2_concrete_interface_20261116.py',
 'scripts/verify_osatlas4b_own_original_mcscore_bf16_v2_actual_rope_config_20261116.py',
 'universal_eval/model/adapters/vllm_adapter.py',
)]
spec = importlib.util.find_spec('vllm')
if spec and spec.origin:
    vllm = Path(spec.origin).parent
    paths += [vllm / name for name in ('model_executor/models/phi3v.py', 'model_executor/models/phi3.py', 'model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py')]
paths += [W / name for name in ('config.json', 'generation_config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'model.safetensors.index.json')]
out = {'UTC': datetime.now(timezone.utc).isoformat(), 'own_model': mid, 'weight_root': str(W), 'state': state[mid], 'sources': {}, 'safetensor_headers_only': {}, 'AGENTS': {}}
for path in paths:
    if path.is_file():
        raw = path.read_bytes()
        out['sources'][str(path)] = {'SHA256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode('utf-8')}
for path in [Path('/AGENTS.md'), Path('/root/AGENTS.md'), Path('/root/projects/AGENTS.md'), R / 'AGENTS.md', R / 'universal_eval/AGENTS.md', R / 'universal_eval/model/AGENTS.md', R / 'scripts/AGENTS.md']:
    if path.is_file():
        out['AGENTS'][str(path)] = path.read_text()
for path in W.glob('*.safetensors'):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    out['safetensor_headers_only'][str(path)] = {'bytes': path.stat().st_size, 'header': {k: v for k, v in header.items() if k != '__metadata__'}}
front = B / 'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117/own_original_chat_frontend200.jsonl'
out['own_frontend_first3'] = [json.loads(line) for line in front.read_text().splitlines()[:3]]
print(json.dumps(out, ensure_ascii=False))

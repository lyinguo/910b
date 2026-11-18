import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import struct
import hashlib

R = Path('/root/projects/comperror')
B = R / 'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B / 'Phi35_own_actual_native_frontend200_engine_stop_v4_explicit_reference_cache_bypass_no_NN_20261118'
M = B / 'Phi35_own_frozen_source_primary_migration_and_current_HF_meta_interface_preflight_no_NN_v2_exact_existing_frontend_1916_20261117/native_meta_state_shapes.json'
W = Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Phi3VForCausalLM/microsoft/Phi-3.5-vision-instruct')
shapes = json.loads(M.read_bytes())
physical = {}
for path in W.glob('*.safetensors'):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    physical.update({k: v for k, v in h.items() if k != '__metadata__'})
front = H / 'own_actual_native_frontend200.jsonl'
rows = [json.loads(s) for s in front.read_text().splitlines()]
print(json.dumps(dict(no_NEW_NPU=True, own_native_frontend_completed_rows=len(rows), full_indices=[x['index'] for x in rows] == list(range(200)), native_front_SHA256=hashlib.sha256(front.read_bytes()).hexdigest(), actual_stop_sets=sorted({tuple(x['actual_engine_request_sampling']['all_stop_token_ids']) for x in rows}), native_pixel_dtypes=sorted({x['actual_native_processor_pixel_tensor']['dtype'] for x in rows}), prior_positive_ID_differences_examples=[x['prior_own_HF_positive_input_differences'] for x in rows[:3]], old_meta_extra={k: shapes[k] for k in set(shapes) - set(physical)}, old_meta_missing={k: physical[k] for k in set(physical) - set(shapes)}, old_meta_state_keys=len(shapes), physical_keys=len(physical), original_migration_meta_file=str(M), current_v4_failure=json.loads((H / 'actual_failure.json').read_bytes())), ensure_ascii=False))

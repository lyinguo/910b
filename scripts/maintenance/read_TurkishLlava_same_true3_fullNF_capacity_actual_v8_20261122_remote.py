"""Complete actual serial state, closed reports and current capacity without replay or mutation."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess
R = Path('/root/projects/comperror')
B = R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rec(p):
    b = p.read_bytes()
    try:
        v = json.loads(b)
    except ValueError:
        v = b.decode('utf8', errors='replace')
    return dict(SHA256=hashlib.sha256(b).hexdigest(), bytes=len(b), complete_value=v)

r = subprocess.run(['python3', '-B', str(R/'scripts/maintenance/read_TurkishLlava_serial_v2_after_Base_failure_and_visual_CPU_diagnostic_actual_v7_20261122_remote.py')],
                   cwd=R, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
assert r.returncode == 0, r.stderr
actual = json.loads(r.stdout)
new_records = actual['Turkish_serial_v2_and_Base_CPU_diagnostic_complete_records']
for stem in ['TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122',
             'TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122']:
    for name in ['own_passed3_complete_all200_shape_capacity.json', 'own_complete200_shape_capacity_review.json',
                 'approval.json', 'formal200_measurement_approval.json']:
        p = B/stem/name
        if p.is_file():
            new_records[str(p)] = rec(p)
    p = B/stem/'completion.json'
    if p.is_file():
        for a in json.loads(p.read_bytes()).get('accepted', []):
            z = Path(a['proof'])
            assert z.is_relative_to(R)
            new_records[str(z)] = rec(z)
captures = actual['all_original_and_current_actual']['Turkish_current_actual_native_capture_progress']
for c in captures:
    home = Path(c['home'])
    p = home/'generation_progress.jsonl'
    if p.is_file():
        lines = p.read_bytes().splitlines()
        c['all_committed_forward_rows'] = [json.loads(z) for z in lines]
runtime = {}
for n in ['configs/LlavaLlamaForCausalLM/ytu-ce-cosmos--Turkish-LLaVA-v0.1-own-original-Transformers-FP16-2048-native-v1-own-original-20261122.yaml',
          'configs/LlavaLlamaForCausalLM/ytu-ce-cosmos--Turkish-LLaVA-v0.1-own-original-Transformers-FP16-2048-native-v1-own-original-20261122-smoke.yaml',
          'configs/protocols/turkish_llava_own_original_Transformers_FP16_2048_native_v1_own_original_20261122.json']:
    p = R/n
    if p.is_file():
        runtime[n] = dict(SHA256=sha(p), bytes=p.stat().st_size)
actual['current_runtime_config_SHA256'] = runtime
actual['UTC'] = datetime.now(timezone.utc).isoformat()
print(json.dumps(actual, ensure_ascii=False))

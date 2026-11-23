"""Read the failed MedraN stage and exact native source, without loading tensors.

This audit cannot establish original A100 behavior or qualify any model. It
preserves the old failures and identifies missing source evidence before a new
private adaptation is considered.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, os, re

R = Path('/root/projects/comperror')
B = R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B/'MedraN_completed_FP16_failure_native_source_contract_readonly_no_model_20261123'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rec(p):
    b = p.read_bytes()
    try:
        v = json.loads(b)
    except ValueError:
        v = b.decode('utf8', errors='replace')
    return dict(path=str(p), bytes=len(b), SHA256=hashlib.sha256(b).hexdigest(), complete_value=v)

def main():
    assert not H.exists()
    H.mkdir()
    canonical = [R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx',
                 R/'workspace/diagnostics/batch_910b/state.json',
                 R/'workspace/diagnostics/batch_910b/A100_ALIGNMENT_LOCAL_WORKBOOK.json']
    before = {str(p): sha(p) for p in canonical}
    old = {}
    for stem, names in [
        ('MedraNE4B_own_original_FP16_true3_v1_real_FLOCK_20261116', ['completion.json']),
        ('MedraNE4B_own_FP16_actual_vision_CPU_same_operands_diagnostic_20261116', ['review.json', 'actual_completion.json']),
        ('MedraNE4B_original_FP16_weight_free_vision_graph_binding_20261116', ['review.json']),
    ]:
        for name in names:
            p = B/stem/name
            if p.is_file():
                old[str(p)] = rec(p)
    original = R/'configs/Gemma3nForConditionalGeneration/nicoboss--MedraN-E4B-Uncensored-EP7.yaml'
    files = [original,
        R/'scripts/prepare_medran_e4b_original_FP16_vision_graph_binding_20261116.py',
        R/'scripts/diagnostic_medran_own_FP16_same_operands_vision_CPU_graph_20261116.py',
        R/'universal_eval/model/adapters/medran_e4b_own_original_FP16_native_v1_20261116.py',
        R/'universal_eval/model/medran_own_FP16_20261116/vision.py']
    spec = importlib.util.find_spec('transformers')
    assert spec and spec.submodule_search_locations
    package = Path(list(spec.submodule_search_locations)[0])
    files += [package/'models/gemma3n/modeling_gemma3n.py',
              package/'models/timm_wrapper/modeling_timm_wrapper.py']
    timm = importlib.util.find_spec('timm')
    timm_locations = list(timm.submodule_search_locations or []) if timm else []
    for loc in timm_locations:
        files += [Path(loc)/'layers/fast_norm.py', Path(loc)/'models/mobilenetv5.py']
    # These are exact original source paths named by the archived operator graph,
    # not a directory search or an alternative checkpoint location.
    exact_original = Path('/mnt/afs/miniconda3/envs/lvzhiqiang/lib/python3.12/site-packages/timm')
    files += [exact_original/'layers/fast_norm.py', exact_original/'models/mobilenetv5.py']
    source_records = {}
    pattern = re.compile(r'keep_in_fp32|finfo|rms_norm|layer_norm|def fast_rms|class Gemma3n|vision_tower|is_timm_available|requires_backends|dtype|eps')
    for i, p in enumerate(dict.fromkeys(files)):
        if not p.is_file():
            source_records[str(p)] = dict(exists=False)
            continue
        b = p.read_bytes()
        dest = H/'exact_source_snapshots'/f'{i:02d}_{p.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(b)
        lines = b.decode('utf8').splitlines()
        selected = set()
        for j, line in enumerate(lines):
            if pattern.search(line):
                selected.update(range(max(0, j-3), min(len(lines), j+5)))
        source_records[str(p)] = dict(exists=True, SHA256=hashlib.sha256(b).hexdigest(), bytes=len(b),
            snapshot=str(dest), selected_lines=[dict(line=j+1, text=lines[j]) for j in sorted(selected)])
    after = {str(p): sha(p) for p in canonical}
    assert before == after
    q = dict(passed=True, UTC=datetime.now(timezone.utc).isoformat(), diagnostic_only=True,
             learned_CPU_forwards=0, model_NPU_forwards=0, no_weight_arrays_loaded=True,
             original_failures=old, exact_source_records=source_records,
             installed_timm_available=timm is not None, installed_timm_locations=timm_locations,
             original_CPU_diagnostic_replays_parameterized_graph=True,
             original_HF_model_runtime_equivalence_not_proven_by_this_source_audit=True,
             no_original_A100_failure_claim=True, no_acceptance_or_qualification=True,
             no_model_dtype_math_budget_threshold_or_extraction_change=True,
             no_deletion=True, canonical_SHA256=after, main_source_SHA256=sha(Path(__file__)))
    (H/'review.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (H/'actual_completion.json').write_text(json.dumps(dict(UTC=q['UTC'],actual_returncode=0,
        source_audit_only=True,learned_CPU_forwards=0,model_NPU_forwards=0))+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,UTC=q['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),
        installed_timm_available=q['installed_timm_available'],source_records=source_records,
        original_failure_summary={n:{'SHA256':v['SHA256'],'bytes':v['bytes']} for n,v in old.items()},
        no_model_or_learned_forward=True,no_deletion=True),ensure_ascii=False))

if __name__ == '__main__':
    main()

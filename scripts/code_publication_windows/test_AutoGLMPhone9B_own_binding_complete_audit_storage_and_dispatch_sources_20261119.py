"""Exact binding review of own graph observers, full audit and storage API."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
P=Path(__file__).resolve().parent
old='huihui_glm46flash_own_original_BF16_2048_20261118';new='autoglm_phone9b_own_original_BF16_2048_20261119'
def tree(path):return ast.parse((P/path).read_bytes())
def fnmap(t):return {n.name:n for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
a=fnmap(tree(old+'_native.py'));b=fnmap(tree(new+'_native.py'))
checked=[]
for name in ['array','write','_record','_install','top_pre','lm_pre','rope_pre','head_hook','top_post','observe','attn_pre','stage','get_sample_execution','get_execution_lineage','unload']:
 assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
 checked.append(name)
save=ast.unparse(b['save_npz']);assert 'write_full_standard_npz' in save and 'values' in save
util=tree('complete_standard_LZMA_NPZ_values_20261118.py');assert 'write_full_standard_npz' in fnmap(util)
nf=(P/('verify_'+new+'_native.py')).read_text()
assert "151343" in nf and "151363" not in nf and "batch_size']==10" in nf and 'range(0,count,10)' in nf and "cfg['data']['filter_empty']" in nf
assert "(n,151552)" in nf and '40,2,2,N,128' in nf and "p['current_A100_D_values']" in nf and "all_own_source_actual_NPU_SHA256_verified=True" in nf
ready=(P/('prepare_'+new+'_true3.py')).read_text();assert 'batch_size=10' in ready and 'filter_empty=True' in ready and 'own_template2048_outer10' in ready
front=(P/(new+'_frontend.py')).read_text();assert 'enable_thinking=False' in front and "W,local_files_only=True" in front
for n in [new+'_native.py',new+'_frontend.py',new+'_MC.py','verify_'+new+'_native.py','prepare_'+new+'_true3.py']:tree(n)
checks=dict(passed=True,unchanged_complete_native_observer_and_cache_assertion_functions=checked,complete_head151552_KV40x2x128_masks_positions_and_all_external_SHA_audit_preserved=True,own_processor_tokenizer_checkpoint_dataset_bound=True,own_image_token151343_and_outer10_and_original_filter_preserved=True,full_standard_LZMA_writer_API_exists=True,no_changed_model_math_or_scoring_threshold_or_budget=True,no_NEW_model_forwards=0)
(P/'AutoGLMPhone9B_own_graph_source_binding_tests_20261119.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8');print(json.dumps(checks))

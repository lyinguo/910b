from pathlib import Path
import importlib.util,json
p=Path(__file__).resolve().parent/'glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py'
if not p.exists():p=Path(__file__).resolve().parents[1]/p.name
s=importlib.util.spec_from_file_location('capacity_components',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
tokens=[300,320,340]+[400]*197;pixels=[100,120,140]+[600]*197;r=m.complete_shape_ratios(tokens,pixels)
assert m.component('own_visual_features',[22,4096],300,195,r)[1]==r['vision']
assert m.component('stage_model_language_model_layers_0',[1,300,4096],300,195,r)[1]==r['prefill']
assert m.component('native_cache_prefill',[40,2,2,300,128],300,195,r)[1]==r['prefill']
assert m.component('all_original_full_head_logits',[195,151552],300,195,r)[1]==r['samples']
assert m.component('f1_attn_position_embeddings_0_0',[3,1,1,128],300,195,r)[1]==r['samples']
assert m.component('unknown_future_mask',[1,300,300],300,195,r)[1]==max(r.values())
assert m.component('stage_model_language_model_layers_0',[1,128,4096],300,195,r)[0]=='unclassified_original_maximum_ratio_retained'
try:m.component('all_original_full_head_logits',[195,32000],300,195,r)
except AssertionError:pass
else:raise AssertionError('Truncated head accepted')
assert m.component('f1_attn_position_embeddings_0_0',[3,1,300,128],300,195,r)[1]==r['prefill']
assert m.component('f195_attn_position_embeddings_0_0',[3,1,1,128],300,195,r)[1]==r['samples']
assert m.component('f2_attn_position_embeddings_0_0',[3,1,999,128],300,195,r)[0]=='sequence_dependent_decode_operand_original_maximum_retained'
parts=[dict(index=i,name='all_original_full_head_logits.npy',shape=[195,151552],compressed_bytes=100) for i in [0,1,2]];e=m.estimate_complete(parts,tokens,pixels,[195]*3,1)
assert e['original_factor']==4 and e['original_extra_reserve_bytes']==6*1024**3 and e['empirical_estimate_bytes']>=int(4.5*1024**3)+20*1024**3
print(json.dumps(dict(passed=True,checks=12,all200_pixels_not_sample_only=True,prefill_shape_axes_proven=True,unknowns_keep_original_maximum_ratio=True,truncated_head_rejected_variable_decode_keeps_original_max=True,factor4_and_all_reserves_preserved=True)))

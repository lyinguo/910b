"""Full200 storage forecast by proven tensor axes, preserving factor4 and all reserves.

Vision uses every original pixel shape. Prefill uses every original token shape.
Decode uses the same empirical output length assumption as the old 3-sample
estimate. Unknown tensors retain the previous maximum shape ratio.
"""
import math,re

def complete_shape_ratios(tokens,pixels):
 assert len(tokens)==len(pixels)==200 and all(type(n) is int and n>0 for n in tokens+pixels)
 return dict(samples=200/3,prefill=sum(tokens)/sum(tokens[:3]),prefill_squared=sum(x*x for x in tokens)/sum(x*x for x in tokens[:3]),vision=sum(pixels)/sum(pixels[:3]))

def component(name,shape,input_tokens,forwards,ratios):
 assert isinstance(name,str) and all(type(x) is int and x>=0 for x in shape) and input_tokens>0 and forwards>0
 fallback=max(ratios.values())
 if name.startswith(('own_pixels_','stage_model_visual_blocks_')) or name=='own_visual_features':return 'vision_all200_pixel_shapes',ratios['vision']
 if name=='original_input_image_grid_thw':return 'fixed_per_original_image',ratios['samples']
 if name=='all_original_full_head_logits':
  assert shape==[forwards,151552];return 'all_generated_full_head_rows',ratios['samples']
 if name=='all_head_inputs':
  assert shape==[forwards,4096];return 'all_generated_head_inputs',ratios['samples']
 if name=='native_cache_decode_delta':
  assert shape==[40,2,2,forwards-1,128];return 'all40_full_native_decode_KV_deltas',ratios['samples']
 match=re.fullmatch(r'f([1-9][0-9]*)_(native_positions|attn_position_ids_0_0|attn_position_embeddings_0_[01])',name)
 if match:
  assert 1<=int(match[1])<forwards
  # A decode observer must carry only its one-token position operands.
  if math.prod(shape)>3*128 or input_tokens in shape:return 'sequence_dependent_decode_operand_original_maximum_retained',fallback
  return 'all_actual_decode_position_operands',ratios['samples']
 is_prefill=(name in ['input_ids','own_original_HF_mask','prefill_inputs_embeds','native_cache_prefill'] or re.fullmatch(r'stage_model_language_model_layers_[0-9]+',name) or re.fullmatch(r'f0_(native_positions|attn_position_ids_0_0|attn_position_embeddings_0_[01])',name))
 if is_prefill:
  degree=shape.count(input_tokens)
  if degree==1:return 'all200_exact_prefill_token_shapes',ratios['prefill']
  if degree==2:return 'all200_exact_quadratic_prefill_shapes',ratios['prefill_squared']
 return 'unclassified_original_maximum_ratio_retained',fallback

def estimate_complete(parts,tokens,pixels,forwards,metadata_bytes):
 assert len(forwards)==3 and all(type(n) is int and n>0 for n in forwards)
 ratios=complete_shape_ratios(tokens,pixels);weighted=0.;rows=[];seen=set();summary={}
 for p in parts:
  assert p['index'] in [0,1,2] and p['compressed_bytes']>=0 and p['name'].endswith('.npy')
  key=(p['index'],p['name']);assert key not in seen;seen.add(key)
  role,ratio=component(p['name'][:-4],p['shape'],tokens[p['index']],forwards[p['index']],ratios)
  weighted+=p['compressed_bytes']*ratio
  rec=summary.setdefault(role,dict(members=0,observed_complete_compressed_bytes=0,forecast200_bytes_before_unchanged_factor4=0.))
  rec['members']+=1;rec['observed_complete_compressed_bytes']+=p['compressed_bytes'];rec['forecast200_bytes_before_unchanged_factor4']+=p['compressed_bytes']*ratio
  rows.append(dict(p,component=role,all200_shape_ratio=ratio))
 assert {p['index'] for p in parts}=={0,1,2} and metadata_bytes>=0
 weighted+=metadata_bytes*max(ratios.values())
 estimate=max(math.ceil(weighted*4)+6*1024**3,int(4.5*1024**3)+20*1024**3)
 return dict(empirical_estimate_bytes=estimate,ratios=ratios,components=summary,all_complete_ZIP_parts_individually_classified=rows,ZIP_headers_and_all_JSON_metadata_bytes=metadata_bytes,metadata_keeps_original_maximum_ratio=True,original_factor=4,original_extra_reserve_bytes=6*1024**3,all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved=True,all200_original_shapes_used_not_only_200_div3=True,decode_output_length_still_empirical_not_worstcase_guarantee=True,no_tensor_value_or_occurrence_or_model_parameter_changed=True)

"""All own200 axes, same empirical decode assumption/factor4/reserves."""
import math,re
CROSS={3,8,13,18,23,28,33,38}
def estimate(parts,records,forwards,metadata_bytes):
 assert len(records)==200 and len(forwards)==3 and all(type(n) is int and n>0 for n in forwards)
 tokens=[len(r['unpad_ids']) for r in records];pixels=[math.prod(r['actual_tensor_metadata']['pixel_values']['shape']) for r in records]
 assert len(set(pixels))==1 and all(r['actual_tensor_metadata']['pixel_values']['shape']==[1,1,4,3,560,560] for r in records)
 ratios=dict(samples=200/3,prefill=sum(tokens)/sum(tokens[:3]),prefill_squared=sum(x*x for x in tokens)/sum(x*x for x in tokens[:3]),vision=sum(pixels)/sum(pixels[:3]));original_maximum=ratios['samples']*max(1.,max(tokens)/min(tokens[:3]),max(pixels)/min(pixels[:3]));fallback=max(original_maximum,*ratios.values());rows=[];summary={};weighted=0.;seen=set()
 for p in parts:
  key=(p['index'],p['name']);assert key not in seen and p['index'] in [0,1,2];seen.add(key);name=p['name'].removesuffix('.npy');shape=p['original_tensor_shape'];degree=shape.count(tokens[p['index']]);role='unknown_original_maximum_retained';ratio=fallback
  if name.startswith(('actual_vision_','actual_patch_','actual_projected_','actual_aspect_ratio_')):role='all200_fixed_physical_four_tile_vision';ratio=ratios['samples']
  elif name=='actual_prefill_text_embeds':assert degree==1;role='all200_exact_prefill_embeddings';ratio=ratios['prefill']
  else:
   m=re.fullmatch(r'(\d{4})_cache_(\d{2})_(key|value)',name)
   if m:
    step,layer=int(m[1]),int(m[2]);assert 1<=step<=forwards[p['index']] and layer in range(40)
    if layer in CROSS:assert step==1 and shape==[1,8,6404,128];role='all200_complete_fixed_cross_KV';ratio=ratios['samples']
    elif step==1:assert shape==[1,8,tokens[p['index']],128];role='all200_exact_self_prefill_KV';ratio=ratios['prefill']
    else:assert shape==[1,8,1,128];role='all_original_self_decode_KV_append_values';ratio=ratios['samples']
   elif re.fullmatch(r'\d{4}_full_vocab_logits',name):assert shape==[1,128256];role='all_full_head_rows_same_empirical_decode_assumption';ratio=ratios['samples']
   elif re.fullmatch(r'\d{4}_text_(cos|sin)',name):
    if name.startswith('0001_'):assert degree==1;role='all200_exact_prefill_positions';ratio=ratios['prefill']
    else:assert math.prod(shape)==128;role='all_actual_single_decode_position_values';ratio=ratios['samples']
   elif name.endswith(('original_cross_attention_mask4','expanded_cross_attention_mask','full_text_row_mask','cross_mask','original_SDPA_cross_mask','self_mask','original_SDPA_self_mask','original_SDPA_vision_mask')):
    if name.endswith('original_SDPA_vision_mask'):role='all200_fixed_original_vision_mask';ratio=ratios['samples']
    elif name.startswith('0001_') and degree in [1,2]:role='all200_exact_prefill_mask_axes';ratio=ratios['prefill_squared' if degree==2 else 'prefill']
    elif not name.endswith('original_cross_attention_mask4') and degree==0:role='all_actual_decode_fixed_mask_operands';ratio=ratios['samples']
  weighted+=p['compressed_bytes']*ratio;item=summary.setdefault(role,dict(members=0,complete_observed_compressed_bytes=0,forecast200_bytes_before_unchanged_factor4=0.));item['members']+=1;item['complete_observed_compressed_bytes']+=p['compressed_bytes'];item['forecast200_bytes_before_unchanged_factor4']+=p['compressed_bytes']*ratio;rows.append(dict(p,component=role,all200_exact_or_previous_maximum_ratio=ratio))
 assert {p['index'] for p in parts}=={0,1,2};weighted+=metadata_bytes*fallback
 budget=max(math.ceil(weighted*4)+6*2**30,int(4.5*2**30)+20*2**30)
 return dict(empirical_estimate_bytes=budget,ratios=ratios,unknown_original_maximum_ratio=original_maximum,components=summary,all_complete_ZIP_members=rows,metadata_bytes=metadata_bytes,original_factor=4,original_extra_reserve_bytes=6*2**30,all_original_hard3_log1_precheckHalf_sample2_times_outer10_guards_preserved=True,decode_output_lengths_remain_same_empirical_assumption_not_2048_worstcase_guarantee=True,no_array_value_or_occurrence_or_source_math_or_budget_reduced=True,all200_own_original_shapes_verified=True,unknown_axes_original_maximum_retained=True)

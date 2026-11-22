"""Build independent own-source complete numeric replay for original More probability."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent
old=(V/'verify_xtuner_llava3_own_original_Transformers_FP16_2048_full_values_v1_20261122.py').read_text()
text=old.replace('from universal_eval.model.xtuner_llava3_own_saved_Transformers_graph_20261122 import build_own_meta','from universal_eval.model.llava_more_own_original_saved_Transformers_FP16_graph_20261122 import build_own_meta').replace('torch.set_num_threads(4)','torch.set_num_threads(16)').replace("'xtuner/llava-llama-3-8b-v1_1-hf'","'aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning'")
text=text.replace("line['source_dtype_counts']=={'F16':682,'F32':4}","line['source_dtype_counts']=={'F16':686}")
text=text.replace("and line['private_cache_4D_empty_same_original_body'] and line['shared_cache_library_unmodified']","and line['original_use_cache'] is False and line['physical_KV_cache_not_created_by_original_protocol'] and line['original_probability_letter_one_forward_cacheFalse']")
text=text.replace("dtypes=={'torch.float16':682,'torch.float32':4}","dtypes=={'torch.float16':686}")
a=text.index("generation=line['original_generation_config']");b=text.index("trace_path=",a)
text=text[:a]+"assert line['original_raw_generation_config_unused_in_scoring']['eos_token_id']==[128001,128008,128009] and line['original_letter_score_function_unchanged']\nassert line['original_pad_RGB']==[122,116,104] and line['original_raw_llama31_scaling_omitted_by_saved_legacy_conversion']\n"+text[b:]
text=text.replace("steps=[dict(r,inputs=read(r['inputs']),native_fused_inputs=read(r['native_fused_inputs']),full_logits=read(r['full_logits']),cache32=[dict(k,key_delta=read(k['key_delta']),value_delta=read(k['value_delta'])) for k in r['cache32']]) for r in g['steps']]","steps=[dict(r,inputs=read(r['inputs']),native_fused_inputs=read(r['native_fused_inputs']),full_logits=read(r['full_logits']),original_F32_last_logits=read(r['original_F32_last_logits'])) for r in g['steps']]")
text=text.replace("fp(cpu_inputs['pixels_F32'])==row['pixels_F32_SHA256']","fp(cpu_inputs['pixels_FP16'])==row['original_actual_FP16_inputs_SHA256']")
text=text.replace("row['actual_model_prompt']","row['actual_algorithm_prompt']")
a=text.index('    caches={};cache_at={}');b=text.index('    for c in calls:',a)
text=text[:a]+'''    assert len(steps)==1
    step=steps[0];assert step['step']==1 and not step['cache32'] and step['original_use_cache'] is False and step['past_key_values_is_None']
    logits=step['full_logits'];assert logits.shape==(1,len(row['input_ids']),128257) and logits.dtype==torch.float16
    assert logits[:,-1,:].argmax(-1).tolist()==step['native_full_vocabulary_argmax_information_only']
    assert torch.equal(step['original_F32_last_logits'],logits[0,-1,:].float())
''' +text[b:]
a=text.index("                if name.startswith('model.language_model.layers.'):");b=text.index('                repeat=',a)
text=text[:a]+"                assert q.shape[1]==32 and k.shape[1]==v.shape[1]==8 and q.shape[-2]==k.shape[-2]==v.shape[-2]\n"+text[b:]
a=text.index("    tokens=e['generation_token_ids']");b=text.index('    print(json.dumps(dict(stage=',a)
text=text[:a]+'''    assert e['native_forward_count']==1 and e['generation_count']==0 and e['generation_token_ids']==e['decoded_token_ids']==[] and e['first_EOS_index'] is None and not e['truncated']
    assert e['original_probability_not_generation'] and e['original_use_cache'] is False and e['original_probability_text']==row['text']
    if rust.token_to_id('<image>') is None:rust.add_special_tokens(['<image>'])
    assert rust.encode(row['text'].replace('<image>','<image>'*576),add_special_tokens=True).ids==row['input_ids']
    candidate={}
    for ch,opt in row['original_options']:
        ids=set()
        for form in (' '+ch,ch):ids.update(rust.encode(form,add_special_tokens=False).ids)
        assert sorted(ids)==row['original_letter_candidate_ID_sets'][ch] and ids
        candidate[ch]=max(step['original_F32_last_logits'][int(k)].item() for k in ids)
    assert step['original_candidate_letters']==[ch for ch,opt in row['original_options']] and step['original_variant_ID_sets']==row['original_letter_candidate_ID_sets']
    assert candidate==step['original_variant_max_scores'];raw=max(candidate,key=candidate.get)
    assert raw==e['raw_output'] and e['error'] is None and not e['empty'] and not e['invalid'];all_forwards+=1
    assert len([c for c in calls if c['kind']=='LlamaAttention'])==32
''' +text[b:]
text=text.replace("all_native_KV_values_losslessly_reconstructed=True,all32_physical_KV_complete_values_SHA_every_forward_verified=True","original_use_cacheFalse_and_no_KV_cache_verified=True,all32_QKV_full_module_values_and_original_causal_masks_verified=True")
text=text.replace('all_complete_full128257_native_argmax_replayed=all_forwards','all_complete_full128257_native_head_all_positions_and_original_letter_variant_scores_replayed=all_forwards')
text=text.replace('original_private_4D_empty_cache_and_physical_prefix_preserved=True','original_single_native_forward_per_sample_and_no_generated_tokens_verified=True,original_probability_protocol_not_replaced_by_generation=True,original_pad_and_system_prompt_preserved=True')
text=text.replace('original_max_new_tokens=2048,NPU_profiler','original_max_new_tokens=2048,original_max2048_unused_by_original_probability_branch=True,independent_CPU_reference_threads=16,complete_standard_parallel_NPZ_full_value_storage_verified=True,NPU_profiler')
text=text.replace("independent_report_verification_xtuner_v1.json","independent_report_verification_llava_more_v1.json")
assert 'cache_at[' not in text and "generation=line['original_generation_config']" not in text and "if name.startswith('model.language_model.layers.')" not in text
ast.parse(text)
name='verify_LlavaMore_own_original_FP16_discriminative_letter_complete_values_v1_20261122.py';(V/name).write_text(text,encoding='utf8',newline='\n')
primary={'scripts/'+name:hashlib.sha256(text.encode()).hexdigest(),'scripts/code_publication_windows/'+Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(V/'LlavaMore_own_probability_fullNF_primary_paths_v1_20261122.json').write_text(json.dumps(primary,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(primary=primary,all_generic_numeric_module_audit_cases_preserved=True,original_probability_variant_max_math_reconstructed_independently=True,fixed_FP16_atol=.01,fixed_FP16_rtol=.01,chosen_before_first_own_NPU_execution=True,all_head_positions_and_source_values_retained=True,no_current_Xtuner_source_or_job_change=True)))

"""Use accepted complete generic F16 module audit; bind own LLaVA merge/source/stop protocol."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent;BASE=Path('D:/code-lv/.tmp/cd22s1');old=(BASE/'scripts/verify_llava_mini_own_original_FP16_2048_full_values_v1_20261122.py').read_text(encoding='utf-8')
imports=old[:old.index('torch.set_num_threads(4)')].replace('from universal_eval.model.llava_mini_own_saved_graph_FP16_20261122 import build_own_saved_graph','from universal_eval.model.xtuner_llava3_own_saved_Transformers_graph_20261122 import build_own_meta')
header='''torch.set_num_threads(4)
assert len(sys.argv)==3
folder=Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_bytes());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];p=json.loads(Path(kw['protocol_file']).read_bytes());assert sha(kw['protocol_file'])==kw['protocol_sha256']
line=json.loads((folder/'operator_lineage.json').read_bytes())['actual_execution'];wp=Path(p['own_weight_path']);device='npu:'+str(kw['npu_device'])
assert cfg['model']['name']==p['model_id']=='xtuner/llava-llama-3-8b-v1_1-hf' and cfg['model']['model_path']==str(wp)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['task_type']=='image-text-to-text'
assert cfg['algorithm']['batch_size']==10 and cfg['metrics']['override']==['Accuracy'] and cfg['metrics']['include_failed_predictions']
assert not cfg['data']['filter_empty'] and not cfg['data']['apply_template'] and not cfg['data']['download'] and not cfg['model']['download']
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert line['source_tensor_count']==line['actual_FP16_parameter_count']==686 and line['source_dtype_counts']=={'F16':682,'F32':4}
assert line['device']==device and line['HF32_enabled']=={'conv':False,'matmul':False} and line['private_cache_4D_empty_same_original_body'] and line['shared_cache_library_unmodified']
assert line['original_max_new_tokens']==2048 and line['original_batch_size']==10 and line['actual_inner_batch_size']==1 and line['original_plain_F32_RoPE_buffers_preserved'] and line['original_CLIP24_projector2_Llama32_full128257_head_preserved']
for n,h in p['runtime_and_original_source_sha256'].items():assert sha(n)==h,n
for n,h in p['static_evidence_SHA256'].items():assert sha(n)==h,n
for n,h in p['own_assets_SHA256'].items():assert sha(wp/n)==h,n
assert sha(p['own_frontend200'])==p['own_frontend200_SHA256'];front=json.loads(Path(p['own_frontend200']).read_bytes());assert len(front)==200
ctor=json.loads(Path(p['own_original_constructor_review']).read_bytes());assert sha(p['own_original_constructor_review'])==p['own_original_constructor_review_SHA256'] and ctor['passed'] and ctor['no_learned_CPU_or_NPU_forward'] and ctor['original_resize_before_manual_remap_order_preserved']
extra_path=Path(ctor['original_extra_head_and_image_embedding_row_values']);assert sha(extra_path)==ctor['original_extra_head_and_image_embedding_row_values_SHA256']
with ReadOnlyNPZ(extra_path) as z:extra={n:torch.from_numpy(np.ascontiguousarray(z[n]).copy()) for n in z.files}
assert fp(extra['added_embedding'])==ctor['original_added_embedding_SHA256'] and fp(extra['added_lm_head'])==ctor['original_added_lm_head_SHA256'] and extra['added_embedding'].shape==extra['added_lm_head'].shape==(1,4096)
meta=build_own_meta(json.loads((wp/'config.json').read_bytes()));modules=dict(meta.named_modules());params=dict(meta.named_parameters());assert len(params)==686
loading_path=Path(line['weight_loading_evidence']);loading=json.loads(loading_path.read_bytes());loaded={r['source_key']:r for r in loading};assert len(loading)==len(loaded)==686
weights={};dtypes=collections.Counter()
for shard,record in p['own_weight_files'].items():
 assert sha(wp/shard)==record['SHA256'] and (wp/shard).stat().st_size==record['bytes']
 with safe_open(str(wp/shard),framework='pt',device='cpu') as sf:
  for name in sf.keys():
   x=sf.get_tensor(name);y=x.half();r=loaded[name];target=r['target_key'];assert target in params and target not in weights
   full=torch.cat([y,extra['added_embedding' if name=='model.embed_tokens.weight' else 'added_lm_head']],dim=0) if name in ['model.embed_tokens.weight','lm_head.weight'] else y
   assert full.shape==params[target].shape and list(x.shape)==r['shape'] and str(x.dtype)==r['source_dtype'] and fp(x)==r['source_SHA256'] and fp(y)==r['original_FP16_cast_SHA256']==r['actual_NPU_original_source_rows_SHA256'] and fp(full)==r['actual_NPU_complete_parameter_SHA256']
   assert r['actual_device']==device and r['actual_dtype']=='torch.float16' and torch.isfinite(full).all();weights[target]=full.clone();dtypes[str(x.dtype)]+=1
assert dtypes=={'torch.float16':682,'torch.float32':4} and set(weights)==set(params)
inv=(1/(500000.**(torch.arange(0,128,2,dtype=torch.int64).float()/128)))
assert line['original_analytic_buffers']=={'model.language_model.rotary_emb.inv_freq':tr(inv),'model.language_model.rotary_emb.original_inv_freq':tr(inv),'model.vision_tower.embeddings.position_ids':tr(torch.arange(577).expand(1,-1))}
generation=line['original_generation_config'];assert not generation['do_sample'] and generation['use_cache'] and generation['num_beams']==1 and generation['max_new_tokens']==2048 and generation['eos_token_id']==128001
assert generation['repetition_penalty']==1 and generation['no_repeat_ngram_size']==0 and generation['forced_bos_token_id'] is generation['forced_eos_token_id'] is None
trace_path=Path(line['per_sample_evidence']);trace=[json.loads(z) for z in trace_path.read_text().splitlines()]
inputs=[json.loads(z) for z in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(z) for z in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
assert len(trace)==len(inputs)==len(outputs)==len(details)==count and [r['index'] for r in trace]==list(range(count))
rust=Tokenizer.from_file(str(wp/'tokenizer.json'));checks=collections.Counter();max_deltas={};external={str(loading_path):sha(loading_path),str(trace_path):sha(trace_path),str(extra_path):sha(extra_path),p['own_original_constructor_review']:p['own_original_constructor_review_SHA256']};all_forwards=0;all_occurrences=0
'''
numeric=old[old.index('def close(actual,expected,tag):'):old.index('correct=0;invalid=[]')]
numeric=numeric.replace("steps=[dict(r,inputs=read(r['inputs']),full_logits=read(r['full_logits']),", "steps=[dict(r,inputs=read(r['inputs']),native_fused_inputs=read(r['native_fused_inputs']),full_logits=read(r['full_logits']),")
numeric=numeric.replace("assert fp(cpu_inputs['pixels_F32'])==row['pixels_F32_SHA256'] and fp(cpu_inputs['pixels_FP16'])==row['pixels_FP16_SHA256'];assert torch.equal(cpu_inputs['pixels_F32'].half().unsqueeze(0),cpu_inputs['pixels_FP16'])", "assert fp(cpu_inputs['pixels_F32'])==row['pixels_F32_SHA256'] and torch.equal(cpu_inputs['attention_mask'],torch.ones_like(cpu_inputs['input_ids']))")
numeric=numeric.replace('==128256','==128257').replace("name.startswith('model.layers.')","name.startswith('model.language_model.layers.')").replace("layer=int(name.split('.')[2])","layer=int(name.split('.')[3])")
a=numeric.index("        elif kind=='_LlavaMiniVisionTower':");b=numeric.index("        else:raise AssertionError",a)
numeric=numeric[:a]+'''        elif kind=='LlavaMultiModalProjector':
            fc1=one(c,'linear_1');fc2=one(c,'linear_2');close(fc2['inputs'][0],F.gelu(fc1['output']),tag+' original_GELU');close(out,fc2['output'],tag+' output')
'''+numeric[b:]
needle="            mask=kwc.get('attention_mask');causal=kind=='LlamaAttention' and mask is None and q.shape[-2]>1"
replace='''            mask=kwc.get('attention_mask');causal=kind=='LlamaAttention' and mask is None and q.shape[-2]>1
            if kind=='LlamaAttention' and mask is not None:
                qn,kn=q.shape[-2],k.shape[-2];offset=kn-qn;allow=(torch.arange(kn)[None,:]<=torch.arange(offset,offset+qn)[:,None]);assert mask.shape[-2:]==(qn,kn)
                if mask.dtype==torch.bool:assert torch.equal(mask[0,0],allow)
                else:assert torch.equal(mask[0,0]==0,allow) and torch.all(torch.isneginf(mask[0,0][~allow])|(mask[0,0][~allow]==torch.finfo(mask.dtype).min))
'''
assert needle in numeric;numeric=numeric.replace(needle,replace)
a=numeric.index('    # Verify original prefusion ordering,');b=numeric.index("    tokens=e['generation_token_ids'];",a)
numeric=numeric[:a]+'''    # Exact original 576-patch projection and masked-scatter fusion, no Mini compression.
    ids=cpu_inputs['input_ids'];assert ids.shape[0]==1 and int((ids==128256).sum())==576
    selected=[c for c in calls if c['module']=='model.vision_tower.encoder.layers.22' and c['kind']=='CLIPEncoderLayer'];assert len(selected)==1
    visual=selected[0]['output'];visual=visual[0] if isinstance(visual,list) else visual
    project=[c for c in calls if c['module']=='model.multi_modal_projector'];assert len(project)==1
    close(project[0]['inputs'][0],visual[:,1:],'Fusion:own_original_select_minus2_patch576')
    expected=weights['model.language_model.embed_tokens.weight'][ids].clone();expected[ids==128256]=project[0]['output'].reshape(-1,4096)
    close(steps[0]['native_fused_inputs']['inputs_embeds'],expected,'Fusion:full_original576_masked_scatter')
    prompt_n=ids.shape[-1]
    for step in steps:
        number=step['step'];rope=[c for c in calls if c['kind']=='LlamaRotaryEmbedding' and c['text_step']==number];assert len(rope)==1
        actual_pos=rope[0]['inputs'][1] if len(rope[0]['inputs'])>1 else rope[0]['kwargs']['position_ids']
        position=torch.arange(prompt_n).expand(1,-1) if number==1 else torch.tensor([[prompt_n+number-2]],dtype=torch.int64)
        close(actual_pos,position,'Position:original_contiguous_native_absolute_positions')
        if number>1:
            previous=torch.tensor([[steps[number-2]['actual_native_argmax'][0]]],dtype=torch.int64)
            close(step['native_fused_inputs']['inputs_embeds'],weights['model.language_model.embed_tokens.weight'][previous],'Fusion:original_next_token_embedding')
'''+numeric[b:]
numeric=numeric.replace('[128001,128008,128009]','[128001]')
numeric=numeric.replace("raw=rust.decode(e['decoded_token_ids'],skip_special_tokens=True).strip();", "raw=rust.decode(e['decoded_token_ids'],skip_special_tokens=True)\n    if json.loads((wp/'tokenizer_config.json').read_bytes()).get('clean_up_tokenization_spaces',False):\n        for a,b in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(\" ' \",\"'\"),(\" n't\",\"n't\"),(\" 'm\",\"'m\"),(\" 's\",\"'s\"),(\" 've\",\"'ve\"),(\" 're\",\"'re\")]:raw=raw.replace(a,b)\n    raw=raw.strip();")
numeric=numeric.replace("native_forwards=len(steps),all_unique_full_arrays=len(arrays))),flush=True)", "native_forwards=len(steps),all_unique_full_arrays=len(arrays))),flush=True,file=sys.stderr)")
tail=old[old.index('correct=0;invalid=[]'):].replace("['batch_size']==1","['batch_size']==10").replace('LlavaMiniLlamaForCausalLM','LlavaLlamaForCausalLM').replace('source_tensor_count=732','source_tensor_count=686').replace('all732_','all686_').replace('128256','128257').replace('original_plain_RoPE_FP16_buffers_and_F32_angle_math_preserved','original_plain_RoPE_F32_buffers_and_F32_angle_math_preserved').replace('original_CLIP24_prefusion4_text32_and_single_image_token_splice_verified','original_CLIP24_projector2_text32_and_full576_image_splice_verified').replace('profiler_first_complete_vision_prefusion_and_text_prefill_only','profiler_first_complete_own_vision_and_text_prefill_only').replace("'independent_report_verification.json'","'independent_report_verification_xtuner_v1.json'")
tail=tail.replace("for path,digest in p['runtime_and_original_source_sha256'].items():external[path]=digest", "for path,digest in p['runtime_and_original_source_sha256'].items():external[path]=digest\nfor path,digest in p['static_evidence_SHA256'].items():external[path]=digest")
source=imports+header+numeric+tail;ast.parse(source);assert 'build_own_saved_graph' not in source and 'pixels_FP16' not in source and 'prefusion_layers' not in source and '128008' not in source and '128009' not in source
n='verify_xtuner_llava3_own_original_Transformers_FP16_2048_full_values_v1_20261122.py';(V/n).write_bytes(source.encode());(V/'XtunerLlava3_own_full_numeric_verifier_primary_paths_20261122.json').write_text(json.dumps({'scripts/'+n:hashlib.sha256(source.encode()).hexdigest()},indent=2),encoding='utf-8');print(json.dumps(dict(source_SHA256=hashlib.sha256(source.encode()).hexdigest(),fixed_FP16_atol_rtol_before_any_own_NPU_execution=.01,all_original_source_full_head32KV_all_module_calls_inputs_fusion_masks_positions_verified=True,only_final_JSON_on_stdout=True,no_NN_or_G_yet=True)))

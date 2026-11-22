"""Complete own-source/CPU numerical replay of every saved native module call."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import collections,hashlib,json,math,re
import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from egm8b_original_independent_cpu_910b import sha,fp,tr
from strict_mc_independent_reference_910b import independent_choice
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from universal_eval.model.llava_more_own_original_saved_Transformers_FP16_graph_20261122 import build_own_meta

torch.set_num_threads(16)
assert len(sys.argv)==3
folder=Path(sys.argv[1]).resolve();count=int(sys.argv[2]);assert count in [3,200]
assert {q.name for q in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
report=json.loads((folder/'eval_report.json').read_bytes());cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];p=json.loads(Path(kw['protocol_file']).read_bytes());assert sha(kw['protocol_file'])==kw['protocol_sha256']
line=json.loads((folder/'operator_lineage.json').read_bytes())['actual_execution'];wp=Path(p['own_weight_path']);device='npu:'+str(kw['npu_device'])
assert cfg['model']['name']==p['model_id']=='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning' and cfg['model']['model_path']==str(wp)
assert cfg['sample_count']==count and cfg['sample_range']=='0:'+str(count-1) and cfg['task_type']=='image-text-to-text'
assert cfg['algorithm']['batch_size']==10 and cfg['metrics']['override']==['Accuracy'] and cfg['metrics']['include_failed_predictions']
assert not cfg['data']['filter_empty'] and not cfg['data']['apply_template'] and not cfg['data']['download'] and not cfg['model']['download']
assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
assert line['source_tensor_count']==line['actual_FP16_parameter_count']==686 and line['source_dtype_counts']=={'F16':686}
assert line['device']==device and line['HF32_enabled']=={'conv':False,'matmul':False} and line['original_use_cache'] is False and line['physical_KV_cache_not_created_by_original_protocol'] and line['original_probability_letter_one_forward_cacheFalse']
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
assert dtypes=={'torch.float16':686} and set(weights)==set(params)
inv=(1/(500000.**(torch.arange(0,128,2,dtype=torch.int64).float()/128)))
assert line['original_analytic_buffers']=={'model.language_model.rotary_emb.inv_freq':tr(inv),'model.language_model.rotary_emb.original_inv_freq':tr(inv),'model.vision_tower.embeddings.position_ids':tr(torch.arange(577).expand(1,-1))}
assert line['original_raw_generation_config_unused_in_scoring']['eos_token_id']==[128001,128008,128009] and line['original_letter_score_function_unchanged']
assert line['original_pad_RGB']==[122,116,104] and line['original_raw_llama31_scaling_omitted_by_saved_legacy_conversion']
trace_path=Path(line['per_sample_evidence']);trace=[json.loads(z) for z in trace_path.read_text().splitlines()]
inputs=[json.loads(z) for z in (folder/'sample_inputs.jsonl').read_text().splitlines()];outputs=[json.loads(z) for z in (folder/'sample_outputs.jsonl').read_text().splitlines()];details=report['detailed_results']
assert len(trace)==len(inputs)==len(outputs)==len(details)==count and [r['index'] for r in trace]==list(range(count))
rust=Tokenizer.from_file(str(wp/'tokenizer.json'));checks=collections.Counter();max_deltas={};external={str(loading_path):sha(loading_path),str(trace_path):sha(trace_path),str(extra_path):sha(extra_path),p['own_original_constructor_review']:p['own_original_constructor_review_SHA256']};all_forwards=0;all_occurrences=0
def close(actual,expected,tag):
    assert isinstance(actual,torch.Tensor) and isinstance(expected,torch.Tensor) and actual.shape==expected.shape,(tag,getattr(actual,'shape',None),getattr(expected,'shape',None))
    assert actual.dtype==expected.dtype and torch.isfinite(actual).all() and torch.isfinite(expected).all(),tag
    if actual.is_floating_point():
        # Fixed before any genuine model execution; never widened on a failure.
        torch.testing.assert_close(actual,expected,rtol=.01,atol=.01,msg=lambda msg:tag+' '+msg)
        delta=float((actual.float()-expected.float()).abs().max()) if actual.numel() else 0.
        max_deltas[tag.split(':',1)[0]]=max(max_deltas.get(tag.split(':',1)[0],0.),delta)
    else:assert torch.equal(actual,expected),tag
    checks[tag.split(':',1)[0]]+=1

for i,e in enumerate(trace):
    group_path=Path(e['group_execution']);assert sha(group_path)==e['group_execution_SHA256'];g=json.loads(group_path.read_bytes());assert g['error'] is None and g['index']==i
    archive=Path(g['archive']);assert sha(archive)==g['archive_SHA256'];external[str(group_path)]=sha(group_path);external[str(archive)]=sha(archive)
    with ReadOnlyNPZ(archive) as z:arrays={n:torch.from_numpy(np.ascontiguousarray(z[n]).copy()) for n in z.files}
    used=collections.Counter()
    def read(value):
        if isinstance(value,dict) and 'array' in value:
            name=value['array'];used[name]+=1;v=arrays[name];assert list(v.shape)==value['shape'] and str(v.dtype)==value['dtype'] and fp(v)==value['raw_SHA256'];return v
        if isinstance(value,dict):return {n:read(v) for n,v in value.items()}
        if isinstance(value,list):return [read(v) for v in value]
        return value
    cpu_inputs=read(g['inputs']);fused=read(g['fused']);calls=[dict(r,inputs=read(r['inputs']),kwargs=read(r['kwargs']),output=read(r['output'])) for r in g['calls']]
    steps=[dict(r,inputs=read(r['inputs']),native_fused_inputs=read(r['native_fused_inputs']),full_logits=read(r['full_logits']),original_F32_last_logits=read(r['original_F32_last_logits'])) for r in g['steps']]
    assert set(used)==set(arrays) and sum(used.values())==g['complete_copied_numeric_occurrences'] and len(arrays)==g['unique_full_NPY_values']
    all_occurrences+=sum(used.values());row=front[i];assert cpu_inputs['input_ids'][0].tolist()==row['input_ids'] and sha(e['media_path'])==row['media_file_SHA256']
    assert fp(cpu_inputs['pixels_FP16'])==row['original_actual_FP16_inputs_SHA256'] and torch.equal(cpu_inputs['attention_mask'],torch.ones_like(cpu_inputs['input_ids']))
    assert e['actual_model_prompt']==row['actual_algorithm_prompt'] and e['prompt']==row['prompt'] and e['actual_device']==device
    def descendants(c,name):
        return [x for x in calls[c['sequence']+1:c['end_sequence']+1] if x['module']==name]
    def one(c,name):
        result=descendants(c,c['module']+'.'+name);assert len(result)==1,(c['module'],name,len(result));return result[0]
    assert len(steps)==1
    step=steps[0];assert step['step']==1 and not step['cache32'] and step['original_use_cache'] is False and step['past_key_values_is_None']
    logits=step['full_logits'];assert logits.shape==(1,len(row['input_ids']),128257) and logits.dtype==torch.float16
    assert logits[:,-1,:].argmax(-1).tolist()==step['native_full_vocabulary_argmax_information_only']
    assert torch.equal(step['original_F32_last_logits'],logits[0,-1,:].float())
    for c in calls:
        name=c['module'];kind=c['kind'];args=c['inputs'];kwc=c['kwargs'];out=c['output'];module=modules[name]
        x=args[0] if args else kwc.get('hidden_states');tag=kind+':'+name
        weight=weights.get(name+'.weight');bias=weights.get(name+'.bias')
        if kind=='Linear':close(out,F.linear(x,weight,bias),tag)
        elif kind=='Conv2d':close(out,F.conv2d(x,weight,bias,module.stride,module.padding,module.dilation,module.groups),tag)
        elif kind=='Embedding':close(out,F.embedding(x,weight,module.padding_idx,module.max_norm,module.norm_type,module.scale_grad_by_freq,module.sparse),tag)
        elif kind=='LayerNorm':close(out,F.layer_norm(x,module.normalized_shape,weight,bias,module.eps),tag)
        elif kind=='LlamaRMSNorm':
            y=x.float();ref=weight*(y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+module.variance_epsilon)).to(x.dtype);close(out,ref,tag)
        elif kind=='GELU':close(out,F.gelu(x,approximate=module.approximate),tag)
        elif kind=='SiLU':close(out,F.silu(x),tag)
        elif kind=='QuickGELUActivation':close(out,x*torch.sigmoid(1.702*x),tag)
        elif kind=='LlamaRotaryEmbedding':
            pos=args[1] if len(args)>1 else kwc['position_ids'];angles=(inv.float()[None,:,None]@pos[:,None,:].float()).transpose(1,2);angles=torch.cat([angles,angles],dim=-1)
            close(out[0],angles.cos().to(x.dtype),tag+'cos');close(out[1],angles.sin().to(x.dtype),tag+'sin')
        elif kind in ['LlamaAttention','CLIPAttention']:
            q=one(c,'q_proj')['output'];k=one(c,'k_proj')['output'];v=one(c,'v_proj')['output'];dim=128 if kind=='LlamaAttention' else 64
            q=q.reshape(*q.shape[:-1],-1,dim).transpose(1,2);k=k.reshape(*k.shape[:-1],-1,dim).transpose(1,2);v=v.reshape(*v.shape[:-1],-1,dim).transpose(1,2)
            if kind=='LlamaAttention':
                cosine,sine=kwc['position_embeddings'];cosine=cosine.unsqueeze(1);sine=sine.unsqueeze(1)
                rotate=lambda t:torch.cat([-t[...,dim//2:],t[...,:dim//2]],dim=-1)
                q=q*cosine+rotate(q)*sine;k=k*cosine+rotate(k)*sine
                assert q.shape[1]==32 and k.shape[1]==v.shape[1]==8 and q.shape[-2]==k.shape[-2]==v.shape[-2]
                repeat=q.shape[1]//k.shape[1];k=k.repeat_interleave(repeat,dim=1);v=v.repeat_interleave(repeat,dim=1)
            mask=kwc.get('attention_mask');causal=kind=='LlamaAttention' and mask is None and q.shape[-2]>1
            if kind=='LlamaAttention' and mask is not None:
                qn,kn=q.shape[-2],k.shape[-2];offset=kn-qn;allow=(torch.arange(kn)[None,:]<=torch.arange(offset,offset+qn)[:,None]);assert mask.shape[-2:]==(qn,kn)
                if mask.dtype==torch.bool:assert torch.equal(mask[0,0],allow)
                else:assert torch.equal(mask[0,0]==0,allow) and torch.all(torch.isneginf(mask[0,0][~allow])|(mask[0,0][~allow]==torch.finfo(mask.dtype).min))

            ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=causal,scale=dim**-.5).transpose(1,2).reshape(x.shape)
            projection=one(c,'o_proj' if kind=='LlamaAttention' else 'out_proj');close(projection['inputs'][0],ref,tag+' pre_out_projection')
        elif kind=='MultiheadAttention':
            q,k,v=args[:3];q=q.transpose(0,1);k=k.transpose(0,1);v=v.transpose(0,1)
            result=F.multi_head_attention_forward(q,k,v,module.embed_dim,module.num_heads,weights[name+'.in_proj_weight'],weights[name+'.in_proj_bias'],None,None,False,0.,weights[name+'.out_proj.weight'],weights[name+'.out_proj.bias'],training=False,key_padding_mask=kwc.get('key_padding_mask'),need_weights=True,attn_mask=kwc.get('attn_mask'),average_attn_weights=True)
            close(out[0],result[0].transpose(0,1),tag+' output');close(out[1],result[1],tag+' attention_weights')
        elif kind=='LlamaMLP':
            gate=one(c,'gate_proj')['output'];up=one(c,'up_proj')['output'];down=one(c,'down_proj');close(down['inputs'][0],F.silu(gate)*up,tag+' gate_product');close(out,down['output'],tag+' output')
        elif kind=='CLIPMLP':
            fc1=one(c,'fc1')['output'];fc2=one(c,'fc2');close(fc2['inputs'][0],fc1*torch.sigmoid(1.702*fc1),tag+' quick_gelu');close(out,fc2['output'],tag+' output')
        elif kind in ['LlamaDecoderLayer','CLIPEncoderLayer']:
            attention=one(c,'self_attn');attn=attention['output'][0];residual=x+attn
            norm=one(c,'post_attention_layernorm' if kind=='LlamaDecoderLayer' else 'layer_norm2');close(norm['inputs'][0],residual,tag+' first_residual')
            mlp=one(c,'mlp');close(out[0] if isinstance(out,list) else out,residual+mlp['output'],tag+' second_residual')
        elif kind=='CLIPVisionEmbeddings':
            patch=one(c,'patch_embedding')['output'].flatten(2).transpose(1,2);cls=weights[name+'.class_embedding'].expand(x.shape[0],1,-1);ref=torch.cat([cls,patch],dim=1)+weights[name+'.position_embedding.weight'][None];close(out,ref,tag)
        elif kind=='LlavaMultiModalProjector':
            fc1=one(c,'linear_1');fc2=one(c,'linear_2');close(fc2['inputs'][0],F.gelu(fc1['output']),tag+' original_GELU');close(out,fc2['output'],tag+' output')
        else:raise AssertionError(('Unverified complete numeric module kind',kind,name))
    # Exact original 576-patch projection and masked-scatter fusion, no Mini compression.
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
    assert e['native_forward_count']==1 and e['generation_count']==0 and e['generation_token_ids']==e['decoded_token_ids']==[] and e['first_EOS_index'] is None and not e['truncated']
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
    print(json.dumps(dict(stage='complete_own_numerical_group_verified',sample=i,all_module_calls=len(calls),native_forwards=len(steps),all_unique_full_arrays=len(arrays))),flush=True,file=sys.stderr)

correct=0;invalid=[];empty=[];truncated=[];per=[]
for i,(a,b,d,e) in enumerate(zip(inputs,outputs,details,trace)):
    raw=e['raw_output'];choice=independent_choice(raw);prediction=choice or ''
    assert a['index']==b['index']==d['index']==i and a['prompt']==d['prompt']==e['prompt'] and a['reference']==b['reference']==d['reference']==front[i]['reference']
    assert b['raw_output']==d.get('raw_output','')==raw and b['prediction']==d['prediction']==prediction and b['error'] is None and d['extra']['execution']==e
    hit=int(choice is not None and choice==str(d['reference']).strip().upper());correct+=hit;per.append(hit)
    if choice is None:invalid.append(i)
    if not raw:empty.append(i)
    if e['truncated']:truncated.append(i)
score=correct/count;assert report['metrics']=={'Accuracy':score} and report['summary']['current_score']==score and report['summary']['primary_metric']=='Accuracy'
assert report['statistics']['total_samples']==count and report['statistics']['batch_size']==10 and report['statistics']['error_count']==0
prof_path=Path(line['profiler']);prof=json.loads(prof_path.read_bytes());events=prof if isinstance(prof,list) else prof['traceEvents'];hw={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
kernels=[e for e in events if e.get('pid') in hw and e.get('ph')=='X' and (str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') or str(e.get('args',{}).get('Task Type','')) in ['AI_CORE','AI_CPU','AICPU','AIV','AIC'])];assert len(kernels)>100
external[str(prof_path)]=sha(prof_path)
for path,digest in line['all_external_loaded_runtime_files_sha256'].items():assert sha(path)==digest;external[path]=digest
for path,digest in p['runtime_and_original_source_sha256'].items():external[path]=digest
for path,digest in p['static_evidence_SHA256'].items():external[path]=digest
for name,r in p['own_weight_files'].items():external[str(wp/name)]=r['SHA256']
for name,digest in p['own_assets_SHA256'].items():external[str(wp/name)]=digest
external[p['own_frontend200']]=p['own_frontend200_SHA256'];external[kw['protocol_file']]=kw['protocol_sha256']
relative={n:str(abs(Decimal(str(score))-Decimal(str(v)))/abs(Decimal(str(v)))) for n,v in p['current_A100_D_values'].items()};within=all(Decimal(v)<=Decimal('.05') for v in relative.values())
source_hashes=dict(p['implementation_source_SHA256']);actual_yaml=Path('/root/projects/comperror/configs/LlavaLlamaForCausalLM')/(cfg['name']+'.yaml');assert actual_yaml.is_file();source_hashes[str(actual_yaml.relative_to('/root/projects/comperror'))]=sha(actual_yaml);source_hashes[str(Path(__file__).resolve().relative_to('/root/projects/comperror'))]=sha(__file__)
proof=dict(passed=True,report_dir=str(folder),sample_count=count,Accuracy=score,metrics={'Accuracy':score},correct=correct,sample_Accuracy=per,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_NPU_inference=True,historical_record_reuse=False,source_tensor_count=686,all686_own_source_cast_actual_NPU_SHA256_verified=True,all_own_source_actual_NPU_SHA256_verified=True,original_use_cacheFalse_and_no_KV_cache_verified=True,all32_QKV_full_module_values_and_original_causal_masks_verified=True,all_original_active_and_unused_source_bytes_verified=True,actual_forward_count=all_forwards,complete_numeric_module_check_counts=dict(checks),fixed_original_FP16_numeric_tolerance=dict(atol=.01,rtol=.01,chosen_before_first_NEW_NPU_model_execution=True),numeric_max_absolute_differences=max_deltas,all_actual_numeric_occurrences_replayed=all_occurrences,all_complete_full128257_native_head_all_positions_and_original_letter_variant_scores_replayed=all_forwards,original_plain_RoPE_F32_buffers_and_F32_angle_math_preserved=True,original_CLIP24_projector2_text32_and_full576_image_splice_verified=True,original_single_native_forward_per_sample_and_no_generated_tokens_verified=True,original_probability_protocol_not_replaced_by_generation=True,original_pad_and_system_prompt_preserved=True,original_max_new_tokens=2048,original_max2048_unused_by_original_probability_branch=True,independent_CPU_reference_threads=16,complete_standard_parallel_NPZ_full_value_storage_verified=True,NPU_profiler_kernel_events=len(kernels),profiler_first_complete_own_vision_and_text_prefill_only=True,all_full_inputs_tokens_raw_references_errors_preserved=True,source_sha256=source_hashes,external_evidence_sha256=external,report_sha256={q.name:sha(q) for q in folder.iterdir()},relative_error_to_A100_table_D=relative['server'],relative_error_to_current_A100_D=relative,formal_alignment_within_5pct=within,original_protocol_final_qualification_complete=(count==200 and within),small3_not_full200_qualification_or_extra_5pct_gate=True,historical_A100_engine_or_tensor_capture_claimed=False)
q=loading_path.parent/'independent_report_verification_llava_more_v1.json';assert not q.exists();q.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(proof,ensure_ascii=False,indent=2))

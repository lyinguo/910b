"""Independent own-checkpoint/full-head/cache/mask/standard-report audit; no CPU NN."""
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,sys,csv,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from universal_eval.algorithm.inference import GreedyInference
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import (sha,check,js,jl,bs,f32,meta,independent_choice,EXTERNAL,FILES,R,D)
MID='unsloth/gemma-3n-E4B-it'
def verify(folder,count):
    assert count in [3,10,200] and re.fullmatch(r'\d{8}_\d{6}',folder.name) and {p.name for p in folder.iterdir()}==FILES
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10
    assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
    assert cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False and cfg['algorithm']['kwargs']['extract_answer'] is True
    kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256'],R/'configs'));assert p['model_id']==MID and cfg['task_type']==p['report_task']
    assert folder.parent==R/'workspace/reports/unsloth_gemma-3n-E4B-it'/p['report_task']
    assert p['workbook_D_reference_authorized'] and p['saved_original_configuration_used']
    for path,h in p['implementation_source_SHA256'].items():assert not Path(path).is_absolute() and '..' not in Path(path).parts;check(R/path,h,R)
    for path,h in p['static_evidence_SHA256'].items():check(path,h)
    check(p['saved_original_YAML'],p['saved_original_YAML_SHA256'],R/'configs')
    check(p['reference_workbook'],p['reference_workbook_SHA256'],Path('/root/projects'))
    snap=js(check(p['preserved_original_snapshot'],p['preserved_original_snapshot_SHA256']));assert snap['models'][MID]['A100_D']==p['current_A100_D_values']['preserved_original']
    W=Path(p['weight_path'])
    for n,h in p['own_runtime_assets_SHA256'].items():check(W/n,h,W)
    check(p['dataset_path'],p['dataset_SHA256'],R/'workspace/datasets')
    own=js(W/'config.json')
    for name,rec in p['own_weight_files'].items():
        assert (W/name).stat().st_size==rec['bytes'];check(W/name,rec['SHA256'],W)
    front=jl(check(p['own_frontend200']));assert len(front)==200
    line=js(folder/'operator_lineage.json')['actual_execution']
    assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B'
    assert line['runtime_dtype']=='BF16' and line['max_new_tokens']==2048 and line['actual_inner_batch_size']==1
    assert line['configured_outer_batch_size']==10 and line['actual_batch_manifest']==line['final_trace_manifest']
    assert line['head_size']==262400 and line['per_layer_input_vocab']==262144 and line['cache_implementation']=='hybrid'
    assert line['EOS']==[1,106] and line['LM_layers']==35 and line['no_A100_tensor_capture_claimed']
    for path,h in line['runtime_library_source_SHA256'].items():check(path,h,Path('/usr/local/python3.12.16/lib/python3.12/site-packages'))
    final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'] and final['caption_audio_tower_forward_calls']==0
    groups=[list(range(i,min(i+10,count))) for i in range(0,count,10)]
    assert final['actual_outer_batch_sizes']==[len(group) for group in groups] and final['actual_outer_batch_indices']==groups
    assert final['actual_single_model_forward_batch_size']==1 and all(rec['status']=='completed' for rec in final['outer_batch_dispatches'])
    assert [rec['requested_indices'] for rec in final['outer_batch_dispatches']]==groups
    weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']))
    assert weights['model_id']==MID and len(weights['active_weights'])==1631 and len(weights['original_legacy_unused_shared_KV_source'])==45
    assert set(weights['actual_unused_source_keys'])==set(weights['original_legacy_unused_shared_KV_source'])
    head=weights['own_output_head_alias'];assert head['source_name']=='model.language_model.embed_tokens.weight' and head['target_name']=='lm_head.weight' and head['tied_parameter_identity']
    assert head['actual_NPU_SHA256']==weights['active_weights'][head['source_name']]['source_SHA256']
    assert any(set(group)=={head['source_name'],head['target_name']} for group in weights['actual_native_aliases'])
    source_records={**weights['active_weights'],**weights['original_legacy_unused_shared_KV_source']}
    index=js(W/'model.safetensors.index.json')['weight_map'];assert set(index)==set(source_records)
    from safetensors import safe_open
    for shard in sorted(set(index.values())):
        with safe_open(str(W/shard),framework='pt',device='cpu') as sf:
            assert set(sf.keys())=={n for n,s in index.items() if s==shard}
            for name in sf.keys():
                tensor=sf.get_tensor(name).contiguous();rec=source_records[name]
                assert list(tensor.shape)==rec['shape'] and str(tensor.dtype)==rec['source_dtype']=='torch.bfloat16'
                source_SHA=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest();assert source_SHA==rec['source_SHA256']
                if name in weights['active_weights']:assert rec['actual_NPU_device']=='npu:4' and rec['actual_NPU_SHA256']==source_SHA
    buffers=weights['native_config_derived_nonpersistent_buffers'];assert set(buffers)==set(weights['CPU_initializer_buffers'])
    for name,rec in buffers.items():assert rec['actual_NPU_SHA256']==rec['SHA256']==weights['CPU_initializer_buffers'][name]['SHA256']
    traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
    for rows in [traces,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
    from transformers import AutoProcessor
    tokenizer=AutoProcessor.from_pretrained(str(W),local_files_only=True,trust_remote_code=True).tokenizer
    scores=[];empty=[];invalid=[];truncated=[];numeric=[];steps=0;cfg_aa=own;types=cfg_aa['text_config']['layer_types'];window=cfg_aa['text_config']['sliding_window']
    for i,compact in enumerate(traces):
        e=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=front[i];tokens=e['generated_tokens'];N=e['prompt_length'];n=e['actual_native_forwards'];frames=e['frames'];assert n==len(tokens)-N==len(frames)==compact['actual_native_forwards'] and 0<n<=2048 and e['error'] is None and compact['error'] is None
        assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and e['actual_model_prompt']==compact['actual_model_prompt']==ref['original_model_prompt']==GreedyInference._add_answer_instruction(ref['prompt']) and e['actual_model_text']==ref['formatted_text']
        assert e['reference']==outs[i]['reference']==ins[i]['reference']==details[i]['reference']==ref['reference']
        check(e['media_path'],ref['image_SHA256'],R/'workspace/datasets');raw=tokenizer.decode(tokens[N:],skip_special_tokens=True).strip();assert raw==e['raw_output']==compact['raw_output']==outs[i]['raw_output']==details[i].get('raw_output',details[i].get('extra',{}).get('raw_output'));choice=independent_choice(raw);assert choice==outs[i]['prediction']==details[i]['prediction']
        assert tokens[:N]==e['input_ids'] and not set(tokens[N:-1]).intersection({1,106}) and (n==2048 or tokens[-1] in [1,106])
        archive=check(e['archive'],e['archive_SHA256']);assert archive.stat().st_size==e['archive_bytes'];visited=set();cache={}
        with ReadOnlyNPZ(archive) as a:
            for key,rec in ref['inputs'].items():meta(a['input_'+key],rec);visited.add('input_'+key)
            assert a['generated_tokens'].tolist()==tokens and a['input_input_ids'][0].tolist()==tokens[:N];visited.add('generated_tokens')
            logits=a['all_original_full_head_logits'];assert logits.shape==(n,262400) and logits.dtype==np.uint16;visited.add('all_original_full_head_logits')
            if i<3:
                for key,rec in e['activation_prefill'].items():
                    name=key.replace('.','_');meta(a[name],rec);visited.add(name)
                ids=a['input_input_ids'][0];image=ids==cfg_aa['image_token_id'];assert int(image.sum())==256
                injection=a['injection_inputs_embeds'][0];text=a['original_text_embeds_before_multimodal_scatter'][0];projected=a['vision_projected'][0]
                # Original Gemma3n replaces the entire own vision vocabulary range,
                # including start/end markers, before scattering actual soft tokens.
                vision_mask=(ids>=cfg_aa['vision_config']['vocab_offset'])&(ids<cfg_aa['audio_config']['vocab_offset']);audio_mask=ids>=cfg_aa['audio_config']['vocab_offset'];assert not audio_mask.any()
                expected_injection=text.copy();vision_words=a['stage_model_embed_vision'][0];assert vision_words.shape==text.shape
                expected_injection[vision_mask]=vision_words[vision_mask];expected_injection[image]=projected
                assert np.array_equal(injection,expected_injection),'Original own text/vision-marker/soft-image insertion differs'
            def consume(value):
                if value is None:return []
                if isinstance(value,list):return [x for v in value for x in consume(v)]
                name=value['array'];meta(a[name],value);visited.add(name);return [a[name]]
            for k,frame in enumerate(frames):
                score=f32(logits[k]).copy();assert np.isfinite(score).all() and int(score.argmax())==frame['head_argmax']
                prior=tokens[:N+k]
                assert int(score.argmax())==tokens[N+k],(i,k,'Own actual full head original HighTen unmodified greedy mismatch')
                root_ids=consume(frame.get('input_ids'));assert len(root_ids)==1 and root_ids[0].tolist()==[prior if k==0 else prior[-1:]]
                consume(frame.get('attention_mask'));consume(frame.get('position_ids'));consume(frame.get('cache_position'));consume(frame.get('token_type_ids'))
                for obs in frame['cache']:
                    g=str(obs['group']);kind=obs['kind']
                    if kind in ['initial','complete_native_shape_change']:
                        value=a[obs['array']];assert bs(value)==obs['SHA256'];visited.add(obs['array']);cache[g]=value.copy()
                    elif kind in ['native_length_growth_one','native_exact_sliding_roll_one']:
                        value=a[obs['array']];assert bs(value)==obs['SHA256'] and value.shape[3]==1;visited.add(obs['array']);cache[g]=np.concatenate([cache[g] if kind=='native_length_growth_one' else cache[g][:,:,:,1:],value],axis=3)
                    else:
                        assert kind=='exact_changed_slots'
                        if obs['changed_count']:
                            ix=a[obs['indices_array']];v=a[obs['values_array']];assert bs(ix)==obs['indices_SHA256'] and bs(v)==obs['values_SHA256'] and len(ix)==len(v)==obs['changed_count'] and np.all(ix[1:]>ix[:-1]) and ix[0]>=0 and ix[-1]<cache[g].size;cache[g].reshape(-1)[ix]=v;visited.update([obs['indices_array'],obs['values_array']])
                    assert list(cache[g].shape)==obs['shape'] and cache[g].dtype==np.uint16
                    if k:assert bs(cache[g])==obs['reconstructed_complete_SHA256']
                assert len(frame['layers'])==35
                for j,obs in enumerate(frame['layers']):
                    assert obs['name']=='model.language_model.layers.'+str(j)
                    assert obs['shared_kv_from']==(18 if types[j]=='sliding_attention' else 19) if j>=20 else obs['shared_kv_from'] is None
                    position=consume(obs.get('position_ids'));assert len(position)==1;pos=position[0];expected_pos=np.arange(N) if k==0 else np.array([N+k-1]);assert pos.tolist()==[expected_pos.tolist()]
                    consume(obs.get('position_embeddings'));consume(obs.get('per_layer_input'));masks=consume(obs.get('attention_mask'))
                    if masks:
                        assert len(masks)==1;m=f32(masks[0]);assert m.ndim==4 and m.shape[:2]==(1,1) and m.shape[-2]==len(expected_pos);L=m.shape[-1];offset=max(0,int(expected_pos[-1])+1-L) if k and types[j]=='sliding_attention' else 0;key_pos=np.arange(L)+offset;expected=(expected_pos[:,None]>=key_pos[None,:])
                        if types[j]=='sliding_attention':expected&=(expected_pos[:,None]-key_pos[None,:])<window
                        assert np.array_equal(masks[0][0,0] if masks[0].dtype==np.bool_ else m[0,0]==0,expected),(i,k,j,'Original causal/sliding geometry differs')
                    else:assert k>0 or N<=window or types[j]=='full_attention','Implicit SDPA mask requires original causal/sliding source and valid context'
            for g,rec in e['final_native_cache_groups_SHA256'].items():assert bs(cache[g])==rec['actual_NPU_SHA256']==rec['reconstructed_complete_SHA256']
            assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
            diagnostic=dict(index=i,full_262400_head_rows=n,all_40_physical_and_4_original_shared_KV_buffers_losslessly_reconstructed_every_step=True,actual_masks_positions_original_causal_sliding_geometry_verified=True)
            diagnostic['A100_value_comparison_available']=False
            numeric.append(diagnostic)
        scores.append(float(choice==str(ref['reference']).strip()));steps+=n
        if not raw:empty.append(i)
        if not choice:invalid.append(i)
        if e['original_budget_hit']:truncated.append(i)
    profile=final['first_real_vision_prefill_all_profiler_files_SHA256'];assert profile
    for path,h in profile.items():check(path,h)
    tasks=[Path(x) for x in profile if re.search(r'/task_time_[^/]+\.csv$',x)];kernel_count=0
    for t in tasks:
        with t.open(newline='') as f:rows=list(csv.DictReader(f))
        assert all(x['Device_id']=='4' for x in rows);kernel_count+=sum(x['kernel_type'].startswith('KERNEL') and float(x['task_time(us)'])>0 for x in rows)
    if not tasks:
        trace=[Path(x) for x in profile if x.endswith('/trace_view.json')];assert trace
        for t in trace:
            z=js(t);events=z if isinstance(z,list) else z['traceEvents'];hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernel_count+=sum(x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL') for x in events)
    assert kernel_count>100,'Real original vision and prefill NPU kernels are required'
    score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['batch_size']==10
    errors={k:abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v)) for k,v in p['current_A100_D_values'].items()};assert all(Decimal(v)!=0 for v in p['current_A100_D_values'].values())
    final_pass=count==200 and all(x<=Decimal('.05') for x in errors.values())
    return dict(passed=True, report_dir=str(folder), sample_count=count, metrics={'Strict_MC_Accuracy':score},
        sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,
        NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,
        all1631_own_source_actual_NPU_SHA256_verified=True,all45_unused_shared_KV_source_preserved=True,
        all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,
        numeric_sample_checks=numeric,NPU_profiler_kernel_events=kernel_count,
        relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D={k:str(v) for k,v in errors.items()},
        formal_alignment_within_5pct=final_pass,original_protocol_final_qualification_complete=final_pass,
        qualification_basis='User-confirmed workbook D; saved own task/configuration and genuine independent NPU200 audit',
        A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],
        independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)

if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root'])
    target=home/'independent_report_verification.json'
    assert not target.exists()
    target.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

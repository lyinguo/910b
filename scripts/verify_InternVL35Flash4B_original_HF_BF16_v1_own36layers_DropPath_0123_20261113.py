"""CPU-only independent own Flash4B report/source/input/head/KV/mask/gating audit. V3 reads the standard nested detailed error and preserves invalid raw fallback; no scoring or model changes."""
from decimal import Decimal
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
from strict_mc_independent_reference_910b import independent_choice
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics';MID='OpenGVLab/InternVL3_5-4B-Flash'
FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
EXTERNAL={}
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(path,digest=None,root=D):
    path=Path(path).resolve();assert path.is_relative_to(root.resolve()),path
    actual=EXTERNAL.get(str(path)) or sha(path);assert digest is None or actual==digest,(str(path),digest,actual)
    EXTERNAL[str(path)]=actual;return path
def js(path):return json.loads(Path(path).read_bytes())
def jl(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def bs(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def f32(value):return np.ascontiguousarray(value.astype(np.uint32)<<16).view(np.float32) if value.dtype==np.uint16 else value.astype(np.float32)
def finite(value):return np.isfinite(f32(value)).all() if value.dtype==np.uint16 else np.isfinite(value).all()
def meta_check(arr,meta):
    assert list(arr.shape)==meta['shape'] and bs(arr)==meta['SHA256']
    dtype={'torch.bfloat16':np.dtype('uint16'),'torch.float32':np.dtype('float32'),'torch.int64':np.dtype('int64'),'torch.bool':np.dtype('bool'),'torch.int32':np.dtype('int32')}[meta['dtype']]
    assert arr.dtype==dtype
def delta_stats(a,b):
    a=f32(a).astype(np.float64);b=f32(b).astype(np.float64);d=a-b
    return dict(max_abs=float(np.max(np.abs(d))),mean_abs=float(np.mean(np.abs(d))),RMSE=float(np.sqrt(np.mean(d*d))),nonzero=int(np.count_nonzero(d)),count=int(d.size),reference_RMS=float(np.sqrt(np.mean(b*b))))
def verify(folder,count):
    assert count in (3,10,200) and folder.parent==R/'workspace/reports'/MID.replace('/','_')/'image-text-to-text'
    assert re.fullmatch(r'\d{8}_\d{6}',folder.name) and {p.name for p in folder.iterdir()}==FILES
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['task_type']=='image-text-to-text'
    assert cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==1
    assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
    assert cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
    kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID
    for path,digest in p['static_evidence_SHA256'].items():check(path,digest)
    controls=js(check(p['independent_AA_controls_review']));assert controls['passed'] and controls['all3_original_NPZ_and_array_value_SHA_verified']
    for path,digest in controls['external_evidence_sha256'].items():check(path,digest)
    for name,digest in p['implementation_source_SHA256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,digest,root=R)
    W=Path(p['weight_path'])
    for name,digest in p['own_assets_SHA256'].items():check(W/name,digest,root=W)
    for name,rec in p['own_weight_files'].items():check(W/name,rec['SHA256'],root=W)
    check(p['dataset'],p['dataset_SHA256'],root=R/'workspace/datasets')
    front=js(check(p['actual_AA_frontend200']))['samples'];cpu=js(check(p['independent_R_CPU200']))['records'];aa_meta=js(check(p['actual_AA_metadata']));refs={r['name']:r for r in js(check(p['actual_AA_source_manifest']))}
    assert len(front)==len(cpu)==200 and len(refs)==806 and [r['index'] for r in front]==list(range(200))
    line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution']
    assert line['hardware']=='Ascend910B' and line['runtime_dtype']=='BF16' and line['max_new_tokens']==2048 and line['use_cache'] is True and line['head_size']==151936 and line['tokenizer_size']==151678 and line['actual_outer_batch_size']==line['actual_inner_batch_size']==1
    assert line['historical_source_boundary']==p['historical_source_boundary']
    assert line['own_original_DropPath_full_semantics'] and line['own_DropPath_instances']==46
    allowed_lib_roots=[Path('/usr/local/python3.12.16/lib/python3.12/site-packages/torch'),Path('/usr/local/python3.12.16/lib/python3.12/site-packages/torch_npu'),Path('/usr/local/python3.12.16/lib/python3.12/site-packages/transformers')]
    for path,digest in line['runtime_library_source_SHA256'].items():
        roots=[root for root in allowed_lib_roots if Path(path).is_relative_to(root)];assert len(roots)==1;check(path,digest,root=roots[0])
    home=Path(line['evidence_root']);assert home.parent==D/'model_runs'
    final=js(check(line['final_trace_manifest']));assert final['samples']==count and final['failed'] is False
    weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weights['weights'])==set(refs)
    for name,ref in refs.items():assert weights['weights'][name]==dict(ref,actual_NPU_SHA256=ref['cast_SHA256'],actual_NPU_device='npu:'+str(kw['npu_devices'][0]))
    assert sorted(sorted(v) for v in weights['aliases'])==sorted(sorted(v) for v in aa_meta['actual_parameter_aliases'])
    trace=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
    for rows in [trace,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(W/'tokenizer.json'))
    import torch
    scores=[];empty=[];invalid=[];truncated=[];steps=0;numeric=[];raw_differences=[]
    for i,compact in enumerate(trace):
        e=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=front[i];prefix=ref['input_ids'][0];frames=e['frames'];gen=e['generated_tokens'];n=len(frames);N=e['compressed_length']
        assert compact['index']==e['index']==i and compact['error'] is None and e['error'] is None
        assert e['actual_native_forwards']==compact['actual_native_forwards']==len(gen)==n and 0<n<=2048
        assert e['actual_model_prompt']==compact['actual_model_prompt']==ref['actual_model_prompt'] and e['query']==ref['text'] and e['input_ids']==prefix
        assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and ins[i]['reference']==outs[i]['reference']==details[i]['reference']==e['reference']==ref['reference']
        check(e['media_path'],ref['media_file_SHA256'],root=R/'workspace/datasets')
        raw=tokenizer.decode(gen,skip_special_tokens=True).strip();assert raw==e['raw_output']==compact['raw_output']==outs[i]['raw_output']==details[i]['raw_output']
        assert details[i]['extra']['execution']==compact and outs[i]['error'] is None and details[i]['extra']['error'] is None
        assert e['stop_ids']==[151645] and 151645 not in gen[:-1] and (n==2048 or gen[-1]==151645) and e['original_budget_hit']==(n==2048 and gen[-1]!=151645)
        with np.load(check(e['archive'],e['archive_SHA256']),allow_pickle=False) as a:
            visited={'input_ids','pixels_BF16','native_cache_prefill','native_cache_decode_delta','all_original_full_head_logits'}
            assert a['input_ids'].tolist()==ref['input_ids'] and bs(a['input_ids'])==ref['input_ids_SHA256']==cpu[i]['input_ids_SHA256']
            assert a['pixels_BF16'].shape==tuple(ref['pixels_shape']) and a['pixels_BF16'].dtype==np.uint16 and bs(a['pixels_BF16'])==ref['pixels_BF16_SHA256']==cpu[i]['pixels_BF16_SHA256']
            logits=a['all_original_full_head_logits'];assert logits.shape==(n,151936) and logits.dtype==np.uint16 and finite(logits)
            assert np.argmax(f32(logits),axis=-1).tolist()==gen
            pre=a['native_cache_prefill'];delta=a['native_cache_decode_delta'];assert pre.shape==(36,2,8,N,128) and delta.shape==(36,2,8,n-1,128) and pre.dtype==delta.dtype==np.uint16 and finite(pre) and finite(delta)
            for j,pair in enumerate(e['final_cache_SHA256']):
                assert len(pair)==2
                for t,digest in enumerate(pair):assert bs(np.concatenate((pre[j,t],delta[j,t]),axis=1)[None])==digest
            expected_stages={*(f'stage_vision_model_encoder_layers_{j}' for j in range(24)),*(f'stage_language_model_model_layers_{j}' for j in range(36)),*(f'own_visual64_256_gate_{j}' for j in range(3)),*(f'own_actual_compressed_inputs_mask_keep_{j}' for j in range(4)),'prefill_inputs_embeds','actual_gate_decisions'}
            assert set(e['activation_prefill'])==expected_stages
            for name,meta in e['activation_prefill'].items():
                retained=not name.startswith('stage_') or i<3
                if retained:meta_check(a[name],meta);visited.add(name)
                else:assert name not in a.files and re.fullmatch('[a-f0-9]{64}',meta['SHA256']) and meta['dtype']=='torch.bfloat16'
            gates=f32(a['own_visual64_256_gate_2']);assert gates.shape==(ref['num_patches'],2) and np.isfinite(gates).all()
            v=torch.from_numpy(gates[:,0].copy());threshold=float(torch.quantile(v,.1));decisions=(gates[:,0]>threshold)&(gates[:,0]>=.5)
            assert decisions.tolist()==e['gate_decisions']==a['actual_gate_decisions'].tolist()
            keep=np.ones(len(prefix),dtype=bool);selected=np.flatnonzero(np.array(prefix)==151671);assert len(selected)==ref['num_patches']*256
            for j,decision in enumerate(decisions):
                if decision:keep[selected[j*256+64:(j+1)*256]]=False
            assert np.array_equal(keep,a['own_actual_compressed_inputs_mask_keep_3'])
            compressed=np.array(prefix)[keep];assert N==len(compressed) and a['own_actual_compressed_inputs_mask_keep_1'].tolist()==compressed.tolist()
            visual=np.concatenate([a['own_visual64_256_gate_0'][j] if decision else a['own_visual64_256_gate_1'][j] for j,decision in enumerate(decisions)])
            merged=a['prefill_inputs_embeds'];assert merged.shape==(1,N,2560) and np.array_equal(merged[0,compressed==151671],visual)
            assert np.array_equal(merged[0,compressed!=151671],a['own_actual_compressed_inputs_mask_keep_0'][compressed!=151671])
            for k,frame in enumerate(frames):
                end=N+k;assert frame['step']==k and frame['argmax']==gen[k] and frame['cache_seq_len']==end and frame['unchanged_prefix_exact_checked_layers']==(36 if k else 0) and frame['head_dtype']=='torch.bfloat16' and frame['head_shape']==[1,1,151936]
                positions=[list(range(N))] if k==0 else [[end-1]]
                assert frame['position_ids_values']==positions
                for field in ['input_ids','position_ids','cache_position','attention_mask']:
                    meta=frame.get(field)
                    if meta is not None:
                        name='f%d_%s'%(k+1,field);meta_check(a[name],meta);visited.add(name)
                        if field=='attention_mask':assert a[name].shape==(1,end) and np.all(a[name]==1)
                if k:assert frame['input_ids_values']==[[gen[k-1]]]
                assert [at['layer'] for at in frame['attention']]==list(range(36))
                for att in frame['attention']:
                    for key in ['attention_mask','position_ids','position_embeddings']:
                        entries=att.get(key)
                        if entries is None:
                            if key=='attention_mask':assert k>0 or N==1,'Eager multi-query prefill must receive real causal mask; one-token decode sees only written native KV'
                            continue
                        for meta in entries:
                            name=meta['array'];visited.add(name);meta_check(a[name],meta)
                            if key=='attention_mask':
                                mask=a[name];q=N if k==0 else 1;assert mask.shape==(1,1,q,end)
                                allowed=np.arange(end-q,end)[:,None]>=np.arange(end)[None,:]
                                assert np.array_equal(mask[0,0]==0,allowed)
            assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
            diagnostic={'index':i,'own_full_head_rows':n,'all36_KV_prefill_and_every_append_reconstructed':True,'original_gate_CPU_recomputed':True,'compressed_length':N}
            if i<3:
                aa_rec=js(check(p['actual_AA_samples'][i]['metadata'],p['actual_AA_samples'][i]['metadata_SHA256']));aa_path=check(p['actual_AA_samples'][i]['arrays'],p['actual_AA_samples'][i]['arrays_SHA256']);stats={}
                with np.load(aa_path,allow_pickle=False) as aa:
                    for meta in aa_rec['records']:meta_check(aa[meta['key']],meta)
                    for name in sorted(expected_stages-{'actual_gate_decisions'}):
                        if name not in aa.files:continue
                        if a[name].shape==aa[name].shape:stats[name]=delta_stats(a[name],aa[name])
                        else:stats[name]={'shape_mismatch':{'NPU':list(a[name].shape),'AA':list(aa[name].shape)},'native_gate_not_forced_to_AA':True}
                    aa_rows=np.stack([aa['full_head_row_'+str(j+1)] for j in range(aa_rec['actual_forwards'])]);aa_gen=np.argmax(f32(aa_rows),axis=-1).tolist();prefix_same=True;row_stats=[]
                    for k in range(min(n,len(aa_gen))):
                        if prefix_same and a['prefill_inputs_embeds'].shape==aa['prefill_inputs_embeds'].shape:
                            row_stats.append(dict(step=k,**delta_stats(logits[k],aa_rows[k])))
                        prefix_same=prefix_same and gen[k]==aa_gen[k]
                    diagnostic.update(AA_stage_differences=stats,AA_same_prefix_full_head_differences=row_stats,AA_raw_exact=raw==aa_rec['raw_output'],AA_original_control_not_NPU_qualification=True)
            numeric.append(diagnostic)
        counts=e['activation_counts'];assert len(counts)==60 and all(v==(n if key.startswith('language_model.') else 1) for key,v in counts.items())
        choice=independent_choice(raw);assert outs[i]['prediction']==details[i]['prediction']==(raw if choice is None else choice)
        scores.append(int(choice==str(ref['reference']).strip().upper()));steps+=n
        if not raw:empty.append(i)
        if choice is None:invalid.append(i)
        if e['original_budget_hit']:truncated.append(i)
    profile=js(check(line['first_real_vision_and_prefill_profiler'],final['first_prefill_profiler_SHA256']));events=profile if isinstance(profile,list) else profile['traceEvents'];hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'}
    kernels=sum(x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL') for x in events);assert kernels>100
    score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['batch_size']==1
    exact=Decimal(sum(scores))/Decimal(count);rel={k:abs(exact-Decimal(v))/abs(Decimal(v)) for k,v in p['A100_D_values'].items()};assert all(Decimal(v)!=0 for v in p['A100_D_values'].values())
    return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,full_151936_original_argmax_rows=steps,all806_own_source_cast_actual_A100_and_NPU_SHA256=True,all_native_KV_values_reconstructed_with_unchanged_prefix_checks=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=kernels,profile_scope='first original vision and first decoder prefill only',relative_error_to_A100_table_D=float(rel['server']),relative_error_exact_three_D={k:str(v) for k,v in rel.items()},relative_error_to_each_A100_D={k:str(v) for k,v in rel.items()},formal_alignment_within_5pct=count==200 and all(v<=Decimal('.05') for v in rel.values()),historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);path=home/'independent_report_verification.json'
    if path.exists():assert js(path)==proof
    else:path.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

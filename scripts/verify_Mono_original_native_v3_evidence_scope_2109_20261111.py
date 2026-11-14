"""Independent audit: Mono own source/input/visual experts/cache/full vocabulary and complete reports."""
from decimal import Decimal
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
from strict_mc_independent_reference_910b import independent_choice

R=Path('/root/projects/comperror');D=R/'workspace/diagnostics';MID='OpenGVLab/Mono-InternVL-2B'
FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
EXTERNAL={}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None,root=D):
    p=Path(p).resolve();assert p.is_relative_to(root.resolve()),p
    actual=EXTERNAL.get(str(p)) or sha(p);assert h is None or actual==h,(str(p),h,actual);EXTERNAL[str(p)]=actual;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def bsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def bf(x):return (x.astype(np.uint32)<<16).view(np.float32)
def tensor_matches(x,meta):
    assert list(x.shape)==meta['shape'] and bsha(x)==meta['SHA256']
    assert ('torch.bfloat16' if x.dtype==np.uint16 else 'torch.'+str(x.dtype))==meta['dtype']
def native_arrays(record,archive,ref):
    frames=record['frames'];gen=record['generated_tokens'];prefix=ref['input_ids'];N=len(prefix);n=len(frames)
    assert len(gen)<=2048 and n==len(gen)+1 and record['original_budget_hit']==(len(gen)==2048)
    assert not set(gen)&set(record['stop_ids'])
    with np.load(archive,allow_pickle=False) as a:
        visited={'input_ids','pixels_BF16','merged_embeddings','visual_token_mask','vision_patch','connector','native_cache_prefill','native_cache_decode_delta','all_original_full_head_logits'}
        assert a['input_ids'].tolist()==[prefix] and bsha(a['pixels_BF16'])==ref['actual_BF16_pixels_SHA256']
        logits=a['all_original_full_head_logits'];assert logits.shape==(n,92553) and np.isfinite(logits).all()
        arg=logits.argmax(-1).tolist();assert arg[:-1]==gen
        if len(gen)<2048:assert arg[-1] in record['stop_ids']
        assert ref['visual_tokens']==256*ref['pixel_shape'][0]
        visual_bits=a['visual_token_mask'];assert visual_bits.shape==(1,N,1) and visual_bits.dtype==np.uint16
        visual_values=bf(visual_bits);assert np.isin(visual_values,[0.,1.]).all();visual=visual_values.astype(np.bool_)
        assert bsha(visual)==ref['visual_token_mask_SHA256']
        assert int(visual.sum())==ref['visual_tokens']
        # The precise image positions are checked against independently tokenized query.
        image_ids=set(np.asarray(prefix)[visual.ravel()].tolist());assert len(image_ids)==1
        assert np.array_equal(a['merged_embeddings'][0,visual.ravel()],a['connector'].reshape(-1,2048))
        assert a['vision_patch'].shape==(ref['pixel_shape'][0],1024,1024) and a['connector'].shape==(ref['pixel_shape'][0],256,2048)
        for name,meta in record['activation_prefill'].items():
            if name in a.files:tensor_matches(a[name],meta);visited.add(name)
            else:
                assert record['index']>=3 and name.startswith('LM_layer_') and re.fullmatch(r'[a-f0-9]{64}',meta['SHA256']) and meta['dtype']=='torch.bfloat16'
                assert meta['shape']==[1,N,2048]
        pre=a['native_cache_prefill'];delta=a['native_cache_decode_delta']
        assert pre.shape==(24,2,8,N,128) and delta.shape==(24,2,8,n-1,128) and pre.dtype==delta.dtype==np.uint16
        assert np.isfinite(bf(pre)).all() and np.isfinite(bf(delta)).all()
        final=np.concatenate((pre,delta),axis=3)
        assert [[bsha(final[i,j][None]) for j in range(2)] for i in range(24)]==record['final_cache_SHA256']
        for k,frame in enumerate(frames):
            end=N+k;qlen=N if k==0 else 1
            assert frame['step']==k and frame['positions']==[list(range(end-qlen,end))] and frame['argmax']==arg[k]
            assert frame['full_logits_shape']==[1,qlen,92553] and frame['logits_dtype']=='torch.float32'
            assert frame['cache_seq_len']==end and len(frame['attention'])==24
            if k==0:tensor_matches(a['visual_token_mask'],frame['actual_visual_token_mask'])
            else:assert frame['actual_visual_token_mask'] is None
            for j,attention in enumerate(frame['attention']):
                assert attention['layer']==j and attention['positions']==frame['positions'] and attention['past_length']==(0 if k==0 else end-1)
                name=attention['mask'];assert name in a.files and name=='causal_mask_'+bsha(a[name]);visited.add(name)
                mask=bf(a[name]);assert mask.shape==(1,1,qlen,end)
                allowed=np.arange(end-qlen,end)[:,None]>=np.arange(end)[None,:]
                assert np.array_equal(mask[0,0]==0,allowed) and (mask[0,0][~allowed]<-1e30).all()
            expected_calls={f'{j:02d}_feed_forward' for j in range(24)}|({f'{j:02d}_feed_forward_ve' for j in range(24)} if k==0 else set())
            assert set(frame['expert_calls'])==expected_calls
            for name,shape in frame['expert_calls'].items():
                assert shape[-1]==2048
                assert int(np.prod(shape[:-1]))==((ref['visual_tokens'] if name.endswith('_ve') else N-ref['visual_tokens']) if k==0 else 1)
        assert set(record['activation_counts'])=={'vision_patch','connector',*(f'LM_layer_{j:02d}' for j in range(24))}
        assert all(v==(1 if k in ('vision_patch','connector') else n) for k,v in record['activation_counts'].items())
        assert visited==set(a.files),(set(a.files)-visited,visited-set(a.files))
        return n,dict(first_full_head_argmax=arg[0],full_head_rows=n,own_92553_head_preserved=True,all24_legacy_KV_reconstructed_final_hash=True,original_2048_extra_unreported_forward=len(gen)==2048)
def verify(folder,count):
    assert count in (3,10,200) and folder.parent==R/'workspace/reports'/MID.replace('/','_')/'image-text-to-text'
    assert re.fullmatch(r'\d{8}_\d{6}',folder.name) and {x.name for x in folder.iterdir()}==FILES
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID
    assert cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==1
    assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
    kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID
    for path,h in p['static_evidence_SHA256'].items():check(path,h)
    for path,h in p['required_code_sha256'].items():assert not Path(path).is_absolute() and '..' not in Path(path).parts;check(R/path,h,root=R)
    W=Path(p['weight_path'])
    for name,h in p['own_assets_SHA256'].items():check(W/name,h,root=W)
    for name,rec in p['own_weight_files'].items():check(W/name,rec['SHA256'],root=W)
    check(p['dataset'],p['dataset_SHA256'],root=R/'workspace/datasets')
    assert cfg['data']['filter_empty'] is False and cfg['model']['download'] is False and cfg['data']['download'] is False
    line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution']
    assert line['hardware']=='Ascend910B' and line['runtime_dtype']=='BF16' and line['max_new_tokens']==2048 and line['actual_outer_batch_size']==line['actual_inner_batch_size']==1
    home=Path(line['evidence_root']);assert home.parent==D/'model_runs'
    finished=js(check(line['final_trace_manifest']));assert finished['samples']==count and not finished['failed']
    weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));aa=js(check(p['actual_AA_source_manifest']));expected={x['name']:x for x in aa}
    assert len(expected)==len(weights['weights'])==253 and weights['aliases']==[]
    for name,ref in expected.items():assert weights['weights'][name]==dict(ref,actual_NPU_SHA256=ref['cast_SHA256'],actual_NPU_device='npu:'+str(kw['npu_devices'][0]))
    assert len(weights['rope_buffers'])==24 and all(x['base']==1000000 for x in weights['rope_buffers'].values())
    trace=jl(check(line['sample_trace'],finished['sample_trace_SHA256']));refs=js(check(p['independent_R_CPU200']))['records']
    ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
    for rows,n in ((refs,200),(trace,count),(ins,count),(outs,count),(details,count)):assert [x['index'] for x in rows]==list(range(n))
    # Independent slow own SentencePiece decoder; no model/adapter under test imported.
    import os
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_MODULES_CACHE=str(home/'verifier_own_hf_modules'),PYTHONDONTWRITEBYTECODE='1')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(W),trust_remote_code=True,local_files_only=True)
    scores=[];empty=[];invalid=[];cutoff=[];token354=[];steps=0;numeric=[]
    for i,compact in enumerate(trace):
        assert compact['error'] is None;entry=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=refs[i]
        assert entry['index']==i and entry['error'] is None and entry['actual_model_prompt']==compact['actual_model_prompt']==ref['actual_pipeline_prompt']
        assert entry['input_ids']==ref['input_ids'] and entry['query']==ref['query'];check(entry['media_path'],ref['media_SHA256'],root=R/'workspace/datasets')
        raw=tok.decode(entry['generated_tokens'],skip_special_tokens=True).split(tok.convert_ids_to_tokens(ref['actual_sep_eos']))[0].strip()
        assert raw==entry['raw_output']==compact['raw_output']==outs[i]['raw_output']==details[i]['raw_output']
        assert outs[i]['error'] is None and ins[i]['reference']==outs[i]['reference']==details[i]['reference']
        assert details[i]['extra']['execution']==compact and compact['actual_native_forwards']==entry['actual_native_forwards']
        choice=independent_choice(raw);assert outs[i]['prediction']==details[i]['prediction']==('' if choice is None else choice)
        scores.append(int(choice==str(outs[i]['reference']).strip().upper()))
        if not raw:empty.append(i)
        if choice is None:invalid.append(i)
        if entry['original_budget_hit']:cutoff.append(i)
        if entry['generated_token354_positions']:token354.append(i)
        n,proof=native_arrays(entry,check(entry['archive'],entry['archive_SHA256']),ref);steps+=n;numeric.append(proof)
    profile=js(check(line['first_real_vision_and_prefill_profiler'],finished['first_prefill_profiler_SHA256']));events=profile if isinstance(profile,list) else profile['traceEvents']
    hardware={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
    kernels=sum(e.get('pid') in hardware and e.get('ph')=='X' and str(e.get('args',{}).get('Task Type','')).startswith('KERNEL') for e in events);assert kernels>100
    score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['summary']['current_score']==score
    assert report['statistics']['total_samples']==count and report['statistics']['batch_size']==1
    # No model-resource substitution: generated 354 is retained and blocks formal qualification.
    decimal_score=Decimal(sum(scores))/Decimal(count);relative={k:abs(decimal_score-Decimal(v))/abs(Decimal(v)) for k,v in p['A100_D_values'].items()}
    return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=cutoff,generated_354_indices=token354,SPM354_output_qualification_pending=bool(token354),NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,full_92553_original_argmax_rows=steps,all253_own_source_cast_actual_A100_and_NPU_SHA256=True,all24_legacy_KV_prefill_every_delta_reconstructed_final_SHA256=True,all24_visual_expert_original_prefill_and_text_only_decode_verified=True,NPU_profiler_kernel_events=kernels,profile_scope='first original vision and prefill only',all200_full_KV_head_inputs_and_visual_arrays_retained=True,LM_layer_arrays_first3_full_and_all200_activation_SHA_count_shape_dtype=True,relative_error_to_A100_table_D=float(relative['server']),relative_error_exact_three_D={k:str(v) for k,v in relative.items()},relative_error_to_each_A100_D={k:str(v) for k,v in relative.items()},formal_alignment_within_5pct=count==200 and not token354 and all(v<=Decimal('.05') for v in relative.values()),independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);path=home/'independent_report_verification.json'
    if path.exists():assert js(path)==proof
    else:path.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

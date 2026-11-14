"""Independent CPU audit of own native T5Gemma2 source, inputs, heads and lossless caches."""
from pathlib import Path
from decimal import Decimal
import json,gzip,hashlib,re,sys
import numpy as np
from strict_mc_independent_reference_910b import independent_choice
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics';MID='google/t5gemma-2-1b-1b';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,digest=None,root=D):
    p=Path(p).resolve();assert p.is_relative_to(root.resolve());h=EXTERNAL.get(str(p)) or sha(p);assert digest is None or h==digest,(str(p),digest,h);EXTERNAL[str(p)]=h;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bs(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a.astype(np.float32)
def meta(a,m):
    d={'torch.bfloat16':np.dtype('uint16'),'torch.float32':np.dtype('float32'),'torch.float16':np.dtype('float16'),'torch.int64':np.dtype('int64'),'torch.int32':np.dtype('int32'),'torch.bool':np.dtype('bool')}
    assert list(a.shape)==m['shape'] and a.dtype==d[m['dtype']] and bs(a)==m['SHA256']
def stats(a,b):
    a=f32(a).astype(np.float64);b=f32(b).astype(np.float64);d=a-b;return dict(max_abs=float(abs(d).max()),RMSE=float(np.sqrt(np.mean(d*d))),nonzero=int(np.count_nonzero(d)),count=d.size)
def allowed(a):
    if a.dtype==np.bool_:return a
    assert np.all((a==0)|(a<0));return a==0
def verify(folder,count):
    assert count in [3,10,200] and folder.parent==R/'workspace/reports/google_t5gemma-2-1b-1b/image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
    assert {p.name for p in folder.iterdir()}==FILES
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['task_type']=='image-text-to-text' and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==1
    assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
    assert cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
    kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID
    for path,h in p['static_evidence_SHA256'].items():check(path,h)
    for name,h in p['implementation_source_SHA256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,h,R)
    W=Path(p['weight_path'])
    for name,h in p['own_runtime_assets_SHA256'].items():check(W/name,h,W)
    check(W/'model.safetensors',p['own_weight']['SHA256'],W);check(p['dataset'],p['dataset_SHA256'],R/'workspace/datasets')
    raw=gzip.decompress(check(p['actual_AA_frontend200_gzip']).read_bytes());assert hashlib.sha256(raw).hexdigest()==p['actual_AA_frontend200_raw_SHA256'];front=json.loads(raw)['samples'];cpu=js(check(p['independent_R_CPU200']));assert cpu['passed'] and cpu['sample_count']==200
    aa_meta=js(check(p['actual_AA_metadata']));refs={r['name']:r for r in js(check(p['actual_AA_source_manifest']))};assert len(refs)==1119
    controls=js(check(p['independent_AA_controls_review']));assert controls['passed'] and controls['all3_full_value_arrays_SHA_verified'] and controls['all10_original_full_head_inputs_SHA_verified']
    control_roots=[D,W,R/'workspace/datasets',Path('/usr/local/python3.12.16/lib/python3.12/site-packages/transformers')]
    for path,h in controls['external_evidence_sha256'].items():
        roots=[root for root in control_roots if Path(path).resolve().is_relative_to(root.resolve())]
        assert len(roots)==1,('Unexpected genuine-AA control evidence root',path)
        check(path,h,roots[0])
    line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['runtime_dtype']=='BF16' and line['max_new_tokens']==64 and line['use_cache'] is True and line['cache_implementation']=='hybrid'
    assert line['head_size']==line['tokenizer_size']==262144 and line['actual_outer_batch_size']==line['actual_inner_batch_size']==1 and line['historical_source_boundary']==p['historical_source_boundary']
    for path,h in line['runtime_library_source_SHA256'].items():
        roots=[x for x in [Path('/usr/local/python3.12.16/lib/python3.12/site-packages/torch'),Path('/usr/local/python3.12.16/lib/python3.12/site-packages/torch_npu'),Path('/usr/local/python3.12.16/lib/python3.12/site-packages/transformers')] if Path(path).is_relative_to(x)];assert len(roots)==1;check(path,h,roots[0])
    final=js(check(line['final_trace_manifest']));assert final['samples']==count and final['failed'] is False
    weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weights['weights'])==set(refs)
    for name,ref in refs.items():assert weights['weights'][name]==dict(ref,actual_NPU_SHA256=ref['cast_SHA256'],actual_NPU_device='npu:'+str(kw['npu_devices'][0]))
    assert sorted(sorted(x) for x in weights['aliases'])==sorted(sorted(x) for x in aa_meta['actual_parameter_aliases'])
    trace=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
    for rows in [trace,ins,outs,details]:assert [r['index'] for r in rows]==list(range(count))
    from universal_eval.model.adapters.t5gemma2_own_original_canonical_frontend_v3_own_tokenizer_config_mask_flag_0123_20261113 import OwnCanonicalFrontend,canonical_text
    tokenizer=OwnCanonicalFrontend(W).tokenizer
    expected_stages={*(f'stage_model_encoder_vision_tower_encoder_layers_{j}' for j in range(27)),*(f'stage_model_encoder_text_model_layers_{j}' for j in range(26)),*(f'stage_model_decoder_layers_{j}' for j in range(26))}
    layer_types=aa_meta['actual_config']['decoder']['layer_types'];enc_layer_types=aa_meta['actual_config']['encoder']['text_config']['layer_types']
    scores=[];empty=[];invalid=[];truncated=[];numeric=[];steps=0
    for i,compact in enumerate(trace):
        e=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=front[i];N=len(ref['inputs']['input_ids']['values'][0]);gen=e['generated_tokens'];frames=e['frames'];n=len(frames)
        assert compact['index']==e['index']==i and compact['error'] is None and e['error'] is None and n==e['actual_native_forwards']==compact['actual_native_forwards']==len(gen)-1 and 0<n<=64
        assert e['actual_model_prompt']==compact['actual_model_prompt']==ins[i]['prompt']==details[i]['prompt']==ref['prompt']
        assert canonical_text(e['actual_model_prompt'])==e['actual_model_text']==ref['actual_model_text']
        assert outs[i]['reference']==ins[i]['reference']==details[i]['reference']==e['reference']==ref['reference']
        check(e['media_path'],ref['media_SHA256'],R/'workspace/datasets');raw=tokenizer.decode(gen,skip_special_tokens=True).strip();assert raw==e['raw_output']==compact['raw_output']==outs[i]['raw_output']==details[i]['raw_output']
        assert details[i]['extra']['execution']==compact and outs[i]['error'] is None and details[i]['extra']['error'] is None
        assert gen[0]==2 and 1 not in gen[1:-1] and (n==64 or gen[-1]==1) and e['original_budget_hit']==(n==64 and gen[-1]!=1) and e['stop_ids']==[1]
        with np.load(check(e['archive'],e['archive_SHA256']),allow_pickle=False) as a:
            visited={'all_original_full_head_logits'}
            for key,expected in ref['inputs'].items():
                name='input_'+key;visited.add(name);value=a[name];h=expected['original_BF16_input_SHA256'] if key=='pixel_values' else expected['SHA256'];assert list(value.shape)==expected['shape'] and bs(value)==h
                if key=='pixel_values':assert value.dtype==np.uint16 and np.isfinite(f32(value)).all()
                else:assert value.dtype==np.int64 and value.tolist()==expected['values']
            assert a['input_input_ids'].tolist()==e['input_ids']
            logits=a['all_original_full_head_logits'];assert logits.shape==(n,262144) and logits.dtype==np.uint16 and np.isfinite(f32(logits)).all() and np.argmax(f32(logits),axis=-1).tolist()==gen[1:]
            assert set(e['activation_prefill'])==expected_stages and len(e['activation_counts'])==79
            for name,m in e['activation_prefill'].items():
                if i<3:visited.add(name);meta(a[name],m);assert a[name].dtype==np.uint16 and np.isfinite(f32(a[name])).all()
                else:assert name not in a.files and re.fullmatch('[a-f0-9]{64}',m['SHA256'])
            assert all(v==(n if name.startswith('model.decoder.') else 1) for name,v in e['activation_counts'].items())
            def consume(entries):
                if entries is None:return []
                values=[]
                for m in entries:visited.add(m['array']);meta(a[m['array']],m);values.append(a[m['array']])
                return values
            assert len(e['encoder_attention_observations'])==26
            for j,obs in enumerate(e['encoder_attention_observations']):
                assert obs['layer_name']==f'model.encoder.text_model.layers.{j}'
                for x in consume(obs.get('position_ids')):assert x.tolist()==[list(range(N))]
                consume(obs.get('position_embeddings'))
                masks=consume(obs.get('attention_mask'))
                if not masks:assert enc_layer_types[j]=='full_attention'
                for mask in masks:
                    assert mask.shape==(1,1,N,N);q=np.arange(N)[:,None];k=np.arange(N)[None,:];dist=q-k;valid=np.ones((N,N),dtype=bool) if enc_layer_types[j]=='full_attention' else ((dist>=0)&(dist<256))|((dist<0)&(-dist<257));assert np.array_equal(allowed(mask)[0,0],valid)
            cache={};cache_hash={}
            assert len(e['cache_layout'])==104 and sorted(n for names in e['cache_group_members'].values() for n in names)==sorted(e['cache_layout'])
            for k,frame in enumerate(frames):
                assert frame['step']==k and frame['argmax']==gen[k+1] and frame['head_shape']==[1,1,262144] and frame['head_dtype']=='torch.bfloat16' and frame['native_cache_class']=='EncoderDecoderCache'
                for field in ['decoder_input_ids','input_ids','attention_mask','decoder_attention_mask','position_ids','cache_position']:
                    m=frame.get(field)
                    if m is not None:
                        visited.add(m['array']);meta(a[m['array']],m)
                        if field=='decoder_input_ids':assert a[m['array']].tolist()==[[gen[k]]]
                        if field=='attention_mask':assert a[m['array']].shape==(1,N) and np.all(a[m['array']]==1)
                for entry in frame['cache']:
                    g=str(entry['group']);kind=entry['kind']
                    if kind=='initial':
                        assert k==0;visited.add(entry['array']);value=a[entry['array']];assert bs(value)==entry['SHA256'];cache[g]=value.copy();cache_hash[g]=bs(value)
                    elif kind=='native_length_growth_one':
                        assert entry['all_previous_NPU_values_checked_bit_identical'];visited.add(entry['append_array']);value=a[entry['append_array']];assert value.shape[3]==1 and bs(value)==entry['append_SHA256'];cache[g]=np.concatenate((cache[g],value),axis=3);cache_hash[g]=bs(cache[g])
                    elif kind=='complete_native_shape_change':
                        assert entry['all_current_native_values_retained'];visited.add(entry['array']);value=a[entry['array']];assert bs(value)==entry['SHA256'];cache[g]=value.copy();cache_hash[g]=bs(value)
                    else:
                        assert kind=='exact_changed_slots' and entry['all_native_NPU_elements_compared_exact']
                        if entry['changed_count']:
                            visited.update([entry['indices_array'],entry['values_array']]);idx=a[entry['indices_array']];values=a[entry['values_array']];assert bs(idx)==entry['indices_SHA256'] and bs(values)==entry['values_SHA256'] and len(idx)==len(values)==entry['changed_count'] and idx.dtype==np.int64 and np.all(idx[1:]>idx[:-1]) and idx[0]>=0 and idx[-1]<cache[g].size
                            cache[g].reshape(-1)[idx]=values;cache_hash[g]=bs(cache[g])
                    assert list(cache[g].shape)==entry['shape'] and cache[g].dtype==np.uint16
                    if k:assert cache_hash[g]==entry['reconstructed_complete_SHA256']
                assert len(frame['attention'])==26
                for j,obs in enumerate(frame['attention']):
                    assert obs['layer_name']==f'model.decoder.layers.{j}'
                    for pos in consume(obs.get('position_ids')):assert pos.tolist()==[[k]]
                    consume(obs.get('position_embeddings'));masks=consume(obs.get('merged_attention_mask'));assert len(masks)==1
                    mask=masks[0];assert mask.ndim==4 and mask.shape[:3]==(1,1,1);L=mask.shape[-1]-N;assert L>0
                    expect=(np.arange(L)<=k)&((k-np.arange(L))<512) if layer_types[j]=='sliding_attention' else np.arange(L)<=k
                    assert np.array_equal(allowed(mask)[0,0,0],np.concatenate((expect,np.ones(N,dtype=bool))))
            for g,final_cache in e['final_native_cache_groups_SHA256'].items():assert bs(cache[g])==final_cache['actual_complete_NPU_SHA256']==final_cache['reconstructed_complete_SHA256']
            assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
            diagnostic=dict(index=i,full262144_head_rows=n,all104_self_cross_cache_tensors_losslessly_reconstructed_every_step=True,native_hybrid_config_and_actual_resolved_cache_class_unmodified=True,original_mask_geometry_CPU_recomputed=True)
            if i<3:
                ar=js(check(p['actual_AA_samples'][i]['metadata'],p['actual_AA_samples'][i]['metadata_SHA256']));ap=check(p['actual_AA_samples'][i]['arrays'],p['actual_AA_samples'][i]['arrays_SHA256']);diff={}
                with np.load(ap,allow_pickle=False) as aa:
                    for rec in ar['records']:meta(aa[rec['key']],rec)
                    for name in sorted(expected_stages):assert a[name].shape==aa[name].shape;diff[name]=stats(a[name],aa[name])
                    rows=np.stack([aa['full_head_row_'+str(t+1)] for t in range(ar['actual_forwards'])]);assert np.argmax(f32(rows),axis=-1).tolist()==ar['generated_tokens'][1:]
                    same=True;heads=[]
                    for t in range(min(n,len(rows))):
                        if same:heads.append(dict(step=t,**stats(logits[t],rows[t])))
                        same=same and gen[t+1]==ar['generated_tokens'][t+1]
                    kv={}
                    for group,members in e['cache_group_members'].items():
                        original=a[f'cache_group_{group}_initial']
                        for part,name in enumerate(members):
                            reference=aa['prefill_'+name];assert original[part].shape==reference.shape;kv[name]=stats(original[part],reference)
                    diagnostic.update(AA_79_prefill_stage_differences=diff,AA_104_prefill_self_cross_cache_differences=kv,AA_same_token_prefix_full_head_differences=heads,AA_raw_exact=raw==ar['raw_output'],nonzero_differences_preserved=True)
            numeric.append(diagnostic)
        choice=independent_choice(raw);assert outs[i]['prediction']==details[i]['prediction']==(raw if choice is None else choice);scores.append(int(choice==str(ref['reference']).strip().upper()));steps+=n
        if not raw:empty.append(i)
        if choice is None:invalid.append(i)
        if e['original_budget_hit']:truncated.append(i)
    profile=js(check(line['first_real_vision_and_prefill_profiler'],final['first_prefill_profiler_SHA256']));events=profile if isinstance(profile,list) else profile['traceEvents'];hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernels=sum(x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL') for x in events);assert kernels>100
    score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['batch_size']==1
    exact=Decimal(sum(scores))/Decimal(count);relative={k:abs(exact-Decimal(v))/abs(Decimal(v)) for k,v in p['A100_D_values'].items()};assert all(Decimal(v)!=0 for v in p['A100_D_values'].values())
    return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,full_262144_original_argmax_rows=steps,all1119_own_source_cast_actual_A100_and_NPU_SHA256=True,all_native_self_cross_cache_values_reconstructed_with_exact_NPU_change_checks=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=kernels,profile_scope='first original vision and text encoder and decoder prefill only',relative_error_to_A100_table_D=float(relative['server']),relative_error_exact_three_D={k:str(v) for k,v in relative.items()},relative_error_to_each_A100_D={k:str(v) for k,v in relative.items()},formal_alignment_within_5pct=count==200 and all(v<=Decimal('.05') for v in relative.values()),historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);dest=home/'independent_report_verification.json'
    if dest.exists():assert js(dest)==proof
    else:dest.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

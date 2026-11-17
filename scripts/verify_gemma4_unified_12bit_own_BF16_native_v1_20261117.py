"""Independent full-value audit of own genuine Unified12B NPU execution.

CPU vision/head/RoPE computations are diagnostics using this checkpoint only.
No CPU model generation or CPU score is substituted for the actual NPU run.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv,contextlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
from universal_eval.algorithm.inference import GreedyInference

R=Path('/root/projects/comperror');MID='google/gemma-4-12B-it'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4UnifiedForConditionalGeneration/google/gemma-4-12B-it')
FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
EXTERNAL={}
TYPES={'torch.bfloat16':np.dtype('uint16'),'torch.float32':np.dtype('uint32'),'torch.int64':np.dtype('int64'),'torch.int32':np.dtype('int32'),'torch.bool':np.dtype('bool')}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def bs(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def ts(t):return bs(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
def f32(a):
    if a.dtype==np.uint16:return (a.astype(np.uint32)<<16).view(np.float32)
    if a.dtype==np.uint32:return a.view(np.float32)
    return a.astype(np.float32)
def check(p,h=None,root=None):
    p=Path(p).resolve();allowed=[R,Path('/root/projects'),W,Path('/usr/local/python3.12.16/lib/python3.12/site-packages')]
    assert p.is_file() and (p.is_relative_to(Path(root).resolve()) if root else any(p.is_relative_to(x) for x in allowed)),str(p)
    value=EXTERNAL.get(str(p)) or sha(p);assert h is None or value==h,(str(p),h,value);EXTERNAL[str(p)]=value;return p
def close(actual,expected,name):
    actual=f32(actual);expected=expected.detach().float().numpy();assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    rmse=float(np.sqrt(np.mean((actual-expected)**2)));scale=float(np.sqrt(np.mean(expected**2)));normalized=rmse/max(scale,1e-6)
    assert normalized<.02,(name,normalized,rmse)
    return dict(RMSE=rmse,normalized_RMSE=normalized,max_absolute=float(np.max(np.abs(actual-expected))))
def original_response(processor,text):
    text=text.strip()
    try:
        if hasattr(processor,'parse_response'):
            value=processor.parse_response(text)
            if isinstance(value,dict):text=value.get('content',text)
            elif isinstance(value,str):text=value
    except Exception:pass
    return text.strip()

def verify(folder,count):
    assert count in [3,200] and re.fullmatch(r'\d{8}_\d{6}',folder.name) and {p.name for p in folder.iterdir()}==FILES
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256'],R/'configs'))
    assert p['model_id']==cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}'
    assert folder.parent==R/'workspace/reports'/MID.replace('/','_')/'image-text-to-text' and cfg['task_type']=='image-text-to-text'
    assert cfg['algorithm']['batch_size']==1 and cfg['algorithm']['kwargs']=={'extract_answer':True} and cfg['metrics']=={'override':['Accuracy']}
    assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
    assert cfg['data']['filter_empty'] is False and cfg['data']['download'] is False and cfg['model']['download'] is False
    assert p['workbook_D_reference_authorized'] and p['saved_original_configuration_used'] and p['own_original_root']==str(W)
    for rel,h in p['implementation_source_SHA256'].items():assert not Path(rel).is_absolute() and '..' not in Path(rel).parts;check(R/rel,h,R)
    for name,h in p['static_evidence_SHA256'].items():check(name,h)
    check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['preserved_original_snapshot'],p['preserved_original_snapshot_SHA256'])
    assert Decimal(str(js(p['preserved_original_snapshot'])['models'][MID]['A100_D']))==Decimal(p['current_A100_D_values']['preserved_original'])
    front=jl(check(p['frontend200']));assert len(front)==200;check(W/'model.safetensors','5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d',W)
    own=js(check(p['own_headers_manifest']));headers=own['headers'];meta=js(check(p['own_META_contract']))['meta']
    from universal_eval.model.reference_sources.gemma4_unified_owned_checkpoint_layout import owned_mapping
    mapped,alias=owned_mapping(headers,{k:v['shape'] for k,v in meta.items()});assert len(mapped)==677 and set(x['dtype'] for x in headers.values())=={'BF16'}
    line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend 910B'
    assert line['actual_head_size']==262144 and line['actual_language_layers']==48 and line['actual_GQA_sliding_heads']==8 and line['actual_global_heads']==1 and line['actual_K_equals_V_from_own_config'] and line['actual_shared_KV_layers']==0
    assert line['actual_execution_dtype']=='BF16' and line['actual_input_floating_dtype']=='float32' and line['generation_kwargs']==dict(max_new_tokens=2048,do_sample=False,temperature=0.0)
    assert line['own_saved_HF_serial1_2048_enable_thinking_False_restored'] and line['no_new_disable_thinking'] and line['original_legacy_adapter_initializer_not_executed'] and line['CPU_learned_model_forward_count']==0
    gen=line['original_generation_config'];assert gen['eos_token_id']==[1,106,50] and gen['suppress_tokens']==[258883,258882] and not gen.get('forced_bos_token_id') and not gen.get('forced_eos_token_id')
    loading=js(check(line['own_source_loading'],line['own_source_loading_SHA256']));assert loading['source_count']==677 and len(loading['records'])==677
    records={x['target_name']:x for x in loading['records']};assert set(records)==set(mapped) and loading['own_head_alias']['actual_tied_identity']
    from safetensors import safe_open
    with safe_open(W/'model.safetensors',framework='pt',device='cpu') as sf:
        assert set(sf.keys())==set(headers)
        for target,source in mapped.items():
            rec=records[target];value=sf.get_tensor(source)
            assert rec['source_name']==source and rec['shape']==list(value.shape)==headers[source]['shape'] and value.dtype==torch.bfloat16
            assert ts(value)==rec['source_SHA256']==rec['actual_NPU_SHA256'] and rec['actual_device']=='npu:4' and rec['actual_dtype']=='torch.bfloat16'
        head=sf.get_tensor('model.language_model.embed_tokens.weight');assert tuple(head.shape)==(262144,3840)
        assert ts(head)==loading['own_head_alias']['actual_NPU_SHA256']
        vision={k:sf.get_tensor(v) for k,v in mapped.items() if k.startswith('model.embed_vision.')}
    from transformers import AutoConfig,AutoProcessor
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextRotaryEmbedding
    config=AutoConfig.from_pretrained(str(W),local_files_only=True);tc=config.text_config;vc=config.vision_config;processor=AutoProcessor.from_pretrained(str(W),local_files_only=True)
    assert tc.num_hidden_layers==48 and tc.hidden_size==3840 and tc.sliding_window==1024 and tc.num_kv_shared_layers==0 and tc.attention_k_eq_v
    assert tc.final_logit_softcapping==30.0 and vc.model_patch_size==48 and tc.use_bidirectional_attention in ['vision',None]
    torch.set_num_threads(4);rope=Gemma4UnifiedTextRotaryEmbedding(tc,device='cpu');linebuffers=line['actual_analytic_buffers']
    for name,value in rope.named_buffers():assert linebuffers['model.language_model.rotary_emb.'+name]['SHA256']==ts(value)
    scale_rec=linebuffers['model.language_model.embed_tokens.embed_scale'];assert scale_rec['dtype'] in ['torch.float32','torch.bfloat16']
    assert scale_rec['SHA256']==ts(torch.tensor(3840**.5,dtype=torch.float32 if scale_rec['dtype']=='torch.float32' else torch.bfloat16))
    for item in linebuffers.values():assert item['device']=='npu:4'
    final=js(check(line['final_manifest']));assert final['samples']==count and final['failed'] is False and final['actual_single_model_forward_batch_size']==1
    expected_groups=[[i] for i in range(count)];assert final['outer_batch_dispatches']==[dict(requested_indices=g,status='completed') for g in expected_groups]
    runtime=js(check(final['final_runtime_SHA256_file'],final['final_runtime_SHA256_file_SHA256']))
    for name,h in runtime.items():check(name,h)
    for name,h in js(check(line['runtime_SHA256_file'],line['runtime_SHA256_file_SHA256'])).items():check(name,h)
    traces=jl(check(line['per_sample_evidence'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
    for rows in [traces,ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
    assert explicit_choice('A small object is visible.') is None and explicit_choice('This is a sample.') is None
    scores=[];empty=[];invalid=[];truncated=[];numeric=[];steps=0
    for i,row in enumerate(traces):
        e=js(check(row['actual_execution_file'],row['actual_execution_SHA256']));ref=front[i];tokens=e['generation_token_ids'];n=len(tokens);assert 0<n<=2048 and n==len(e['frames'])==row['actual_native_forwards']==e['actual_native_forwards'] and row['error'] is None
        assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and row['prompt']==ref['original_model_prompt']==GreedyInference._add_answer_instruction(ref['prompt'])
        assert row['reference']==outs[i]['reference']==ins[i]['reference']==details[i]['reference']==ref['reference'];check(row['media_path'],ref['source_image_SHA256'],R/'workspace/datasets')
        raw_decoded=processor.decode(tokens,skip_special_tokens=True).strip();raw=original_response(processor,raw_decoded);assert raw_decoded==e['raw_decoded'] and raw==e['raw_output']==row['raw_output']==outs[i]['raw_output']==details[i].get('raw_output',details[i].get('extra',{}).get('raw_output'))
        choice=explicit_choice(raw) or '';assert outs[i]['prediction']==details[i]['prediction'] and (explicit_choice(str(outs[i]['prediction'])) or '')==choice
        assert not set(tokens[:-1]).intersection({1,106,50}) and (n==2048 or tokens[-1] in [1,106,50])
        archive=check(e['archive'],e['archive_SHA256']);assert archive.stat().st_size==e['archive_bytes'];visited=set();cache={};head_errors=[]
        with ReadOnlyNPZ(archive) as arrays:
            def consume(rec):
                a=arrays[rec['key']];assert list(a.shape)==rec['shape'] and a.dtype==TYPES[rec['dtype']] and bs(a)==rec['SHA256'] and rec['device']=='npu:4';visited.add(rec['key']);return a
            inputs={name:consume(rec) for name,rec in e['input_values'].items()}
            assert set(inputs)==set(ref['inputs'])
            for name,a in inputs.items():assert list(a.shape)==ref['inputs'][name]['shape'] and e['input_values'][name]['dtype']==ref['inputs'][name]['dtype'] and bs(a)==ref['inputs'][name]['SHA256']
            prefix=inputs['input_ids'][0].tolist();N=len(prefix);assert prefix==row['actual_original_input_ids'][0]
            projected=consume(e['visual']);injected=consume(e['language_prefill']);assert projected.shape==(1,280,3840) and injected.shape==(1,N,3840)
            def v(name):return vision['model.embed_vision.'+name]
            with torch.inference_mode():
                x=torch.from_numpy(f32(inputs['pixel_values']).copy()).to(torch.bfloat16)
                x=torch.nn.functional.layer_norm(x,(6912,),v('patch_ln1.weight'),v('patch_ln1.bias'),1e-5)
                x=torch.nn.functional.linear(x,v('patch_dense.weight'),v('patch_dense.bias'))
                x=torch.nn.functional.layer_norm(x,(vc.mm_embed_dim,),v('patch_ln2.weight'),v('patch_ln2.bias'),1e-5)
                xy=torch.from_numpy(inputs['image_position_ids'].copy());valid=(xy!=-1).to(torch.bfloat16).unsqueeze(-1)
                pos=(v('pos_embedding')[xy.clamp(min=0).long(),torch.arange(2)]*valid).sum(-2);x=x+pos
                x=torch.nn.functional.layer_norm(x,(vc.mm_embed_dim,),v('pos_norm.weight'),v('pos_norm.bias'),1e-5)
                x=(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+vc.rms_norm_eps)).to(torch.bfloat16)
                expected_vision=torch.nn.functional.linear(x,v('multimodal_embedder.embedding_projection.weight'))
            vision_error=close(projected,expected_vision,(i,'Own exact vision pipeline'))
            isimage=inputs['input_ids']==config.image_token_id;validpatch=~np.all(inputs['image_position_ids']==-1,axis=-1);assert int(isimage.sum())==int(validpatch.sum())
            expected_embedding=(head[torch.from_numpy(inputs['input_ids'].copy())]*torch.tensor(3840**.5).to(torch.bfloat16)).view(torch.uint16).numpy()
            assert np.array_equal(injected[~isimage],expected_embedding[~isimage]) and np.array_equal(injected[isimage],projected[validpatch])
            if i<3:
                assert set(e['prefill_LM_layers'])=={str(j) for j in range(48)}
                for rec in e['prefill_LM_layers'].values():a=consume(rec);assert a.shape==(1,N,3840) and np.isfinite(f32(a)).all()
            for j,frame in enumerate(e['frames']):
                root={name:consume(rec) for name,rec in frame['root_inputs'].items()};assert root['input_ids'].tolist()==[prefix if j==0 else [tokens[j-1]]]
                hidden=consume(frame['head_hidden']);actual=consume(frame['head_logits']);assert hidden.shape==(3840,) and actual.shape==(262144,)
                with torch.inference_mode():z=torch.nn.functional.linear(torch.from_numpy(hidden.copy()).view(torch.bfloat16),head);expected_head=((z/30.0).tanh()*30.0)
                head_errors.append(close(actual,expected_head,(i,j,'Own independent all262144 BF16 head')))
                score=f32(actual).copy();assert np.isfinite(score).all()
                for item in e['logits_processors']:
                    assert item['name']=='SuppressTokensLogitsProcessor',item
                    attrs=item['attributes'];suppressed=attrs.get('suppress_tokens');suppressed=suppressed['values'] if isinstance(suppressed,dict) else suppressed
                    assert sorted(suppressed)==[258882,258883] and set(attrs)<={'suppress_tokens','device'};score[suppressed]=-np.inf
                assert int(score.argmax())==tokens[j],(i,j,'Actual original unmodified greedy token differs')
                qpos=np.arange(N) if j==0 else np.array([N+j-1]);assert set(frame['rope'])=={'sliding_attention','full_attention'}
                for kind,rec in frame['rope'].items():
                    positions=consume(rec['position_ids']);assert positions.tolist()==[qpos.tolist()]
                    with torch.inference_mode():cs=rope(torch.empty((1,len(qpos),3840),dtype=torch.bfloat16),torch.tensor([qpos.tolist()]),kind)
                    for field,expected_cs in zip(['cos','sin'],cs):a=consume(rec[field]);delta=f32(a)-expected_cs.float().numpy();assert np.max(np.abs(delta))<=.02 and np.sqrt(np.mean(delta**2))<=.004
                assert len(frame['cache'])==48
                for layer,item in enumerate(frame['cache']):
                    kind=tc.layer_types[layer];K=consume(item['K']);V=consume(item['V']);assert K.shape==V.shape and item['layer']==layer and item['prefix_unchanged_on_NPU']
                    assert K.shape[:2]==(1,8 if kind=='sliding_attention' else 1) and K.shape[-1]==(256 if kind=='sliding_attention' else 512)
                    if j==0:assert item['mode']=='actual_prefill' and K.shape[-2]==N;fullK,fullV=K.copy(),V.copy()
                    else:
                        assert item['mode']=='actual_append_with_exact_eviction' and K.shape[-2]==1
                        oldK,oldV=cache[layer];fullK=np.concatenate([oldK,K],axis=-2);fullV=np.concatenate([oldV,V],axis=-2)
                    assert list(fullK.shape)==item['attention_shape'] and item['logical_seen']==N+j
                    retained=min(1023,N+j) if kind=='sliding_attention' else N+j
                    cache[layer]=(fullK[:,:,-retained:].copy(),fullV[:,:,-retained:].copy())
                    assert list(cache[layer][0].shape)==item['shape'] and item['retained_cache_length']==retained
                    if j:assert item['drop']==(1 if kind=='sliding_attention' and N+j>1023 else 0)
                for kind,rec in frame['masks'].items():
                    assert kind in ['full_attention','sliding_attention']
                    item=frame['cache'][tc.layer_types.index(kind)];L=item['attention_returned_length'];offset=N+j-L;kpos=np.arange(L)+offset
                    allowed=qpos[:,None]>=kpos[None,:]
                    if j==0 and tc.use_bidirectional_attention=='vision':allowed|=isimage[0,qpos,None]&isimage[0,kpos][None,:]
                    if kind=='sliding_attention':allowed&=(qpos[:,None]-kpos[None,:])<1024
                    if rec is None:assert len(qpos)==1 and allowed.all()
                    else:
                        m=consume(rec);assert m.shape==(1,1,len(qpos),L)
                        actual_allowed=m[0,0] if m.dtype==np.bool_ else f32(m)[0,0]==0
                        assert np.array_equal(actual_allowed,allowed),(i,j,kind,'Original mask geometry differs')
                        if m.dtype!=np.bool_:assert np.all((f32(m)==0)|(f32(m)<-1e20))
            for item in e['final_native_cache']:
                for kind,a in zip(['K','V'],cache[item['layer']]):rec=item[kind];assert bs(a)==rec['SHA256'] and list(a.shape)==rec['shape'] and rec['device']=='npu:4'
            assert set(arrays.files)==visited,(set(arrays.files)-visited,visited-set(arrays.files))
        scores.append(int(choice==str(ref['reference']).strip()));steps+=n
        if not raw:empty.append(i)
        if not choice:invalid.append(i)
        if n==2048 and tokens[-1] not in [1,106,50]:truncated.append(i)
        numeric.append(dict(index=i,actual_full_head_rows=n,own_vision_CPU_diagnostic=vision_error,max_head_normalized_RMSE=max(x['normalized_RMSE'] for x in head_errors),all48_native_cache_layers_losslessly_reconstructed=True,original_masks_positions_and_greedy_tokens_verified=True))
    profile=final['first_vision_prefill_all_profiler_files_SHA256'];assert profile;kernel_count=0
    for name,h in profile.items():check(name,h)
    tasks=[Path(x) for x in profile if re.search(r'/task_time_[^/]+\.csv$',x)]
    for path in tasks:
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
        assert all(x['Device_id']=='4' for x in rows);kernel_count+=sum(x['kernel_type'].startswith('KERNEL') and float(x['task_time(us)'])>0 for x in rows)
    if not tasks:
        for path in [x for x in profile if x.endswith('npu_trace.json') or x.endswith('trace_view.json')]:
            z=js(path);events=z if isinstance(z,list) else z['traceEvents'];hw={x['pid'] for x in events if x.get('name')=='process_name' and x.get('args',{}).get('name')=='Ascend Hardware'};kernel_count+=sum(x.get('pid') in hw and x.get('ph')=='X' and str(x.get('args',{}).get('Task Type','')).startswith('KERNEL') for x in events)
    assert kernel_count>100,'Genuine vision+prefill NPU kernels required'
    score=sum(scores)/count;assert report['metrics']=={'Accuracy':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
    errors={k:abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v)) for k,v in p['current_A100_D_values'].items()};passed200=count==200 and all(v<=Decimal('.05') for v in errors.values())
    return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Accuracy':score},sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,NEW_real_NPU_inference=True,actual_NPU_inference=True,historical_record_reuse=False,relative_error_to_A100_table_D=float(errors['server']),actual_forward_count=steps,all677_own_source_to_actual_NPU_SHA256_verified=True,all262144_head_rows_independent_CPU_diagnostic=True,all48_native_cache_values_losslessly_reconstructed=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=kernel_count,profiler_scope='First real vision+prefill only, not all forwards',relative_error_to_current_A100_D={k:str(v) for k,v in errors.items()},formal_alignment_within_5pct=passed200,original_protocol_final_qualification_complete=passed200,A100_historical_engine_or_tensor_capture_claimed=False,CPU_model_generation_or_score_substitution=False,independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)

if __name__=='__main__':
    with contextlib.redirect_stdout(sys.stderr):proof=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
    home=Path(js(Path(sys.argv[1])/'operator_lineage.json')['actual_execution']['evidence_root']);target=home/'independent_report_verification.json';assert not target.exists();target.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

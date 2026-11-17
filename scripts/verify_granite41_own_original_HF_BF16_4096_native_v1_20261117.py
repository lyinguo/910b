"""Independent own Granite4.1 source, native full-head/cache and report audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
from contextlib import ExitStack
import os,json,hashlib,re,csv,inspect
import numpy as np
import torch
from safetensors import safe_open
R=Path(__file__).resolve().parent.parent;D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from universal_eval.model.coder_gemma4_e2b_own_20261116.observer import sha,bs
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import independent_choice
from universal_eval.algorithm.inference import GreedyInference
from universal_eval.model.granite41_own_original_source_20261117.configuration import Granite4VisionConfig
from universal_eval.model.granite41_own_original_source_20261117.processing import Granite4VisionProcessor
from transformers.models.granite.modeling_granite import GraniteRotaryEmbedding
MID='ibm-granite/granite-vision-4.1-4b';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def check(p,h=None):
    p=Path(p).resolve();assert p.is_file();digest=EXTERNAL.get(str(p)) or sha(p);assert h is None or h==digest,(str(p),h,digest);EXTERNAL[str(p)]=digest;return p
def f32(a):return torch.from_numpy(a.copy()).view(torch.bfloat16).float() if a.dtype==np.uint16 else torch.from_numpy(a.copy()).float()
def verify(folder,count):
    torch.set_num_threads(4);assert count in [3,200] and re.fullmatch(r'\d{8}_\d{6}',folder.name) and {p.name for p in folder.iterdir()}==FILES
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}'
    assert cfg['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs=dict(extract_answer=True));assert cfg['data']['filter_empty'] is False and not cfg['data']['download'] and not cfg['model']['download']
    p=js(check(cfg['model']['kwargs']['protocol_file'],cfg['model']['kwargs']['protocol_sha256']));assert p['model_id']==MID and p['report_task']==cfg['task_type']
    for name,h in p['implementation_source_SHA256'].items():assert not Path(name).is_absolute() and '..' not in Path(name).parts;check(R/name,h)
    for name,h in p['static_evidence_SHA256'].items():check(name,h)
    W=Path(p['weight_path']);front=jl(check(p['own_frontend200']));headers=js(check(p['own_source_headers']));mapping=js(check(p['own_source_to_native_name_map']))['mapping'];assert len(headers)==len(mapping)==1083 and len(front)==200
    for n,h in p['own_runtime_assets_SHA256'].items():check(W/n,h)
    for n,x in p['own_weight_files'].items():assert check(W/n,x['SHA256']).stat().st_size==x['bytes']
    line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['runtime_dtype']=='BF16' and line['max_new_tokens']==4096 and line['configured_outer_batch_size']==10 and line['actual_inner_batch_size']==1 and line['EOS']==[100257] and line['head_size']==100352 and line['LM_layers']==40 and line['vision_layers']==27 and line['use_cache'] and not line['A100_historical_tensor_capture_claimed']
    for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
    final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'] and final['outer_dispatches']==[dict(indices=list(range(i,min(i+10,count))),status='completed') for i in range(0,count,10)]
    for name,h in final['all_actual_imported_source_and_mapped_library_SHA256'].items():check(name,h)
    weights=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert weights['model_id']==MID and len(weights['all_own_source_tensors'])==1083 and weights['no_foreign_or_random_weight']
    with ExitStack() as stack:
        readers={n:stack.enter_context(safe_open(str(W/n),framework='pt',device='cpu')) for n in p['own_weight_files']};own_head=None
        for name,rec in weights['all_own_source_tensors'].items():
            t=readers[rec['source_file']].get_tensor(name);assert t.dtype==torch.bfloat16 and list(t.shape)==rec['shape']==headers[name]['shape'] and rec['native_name']==mapping[name] and tsha(t)==rec['source_SHA256']==rec['actual_NPU_SHA256'] and rec['actual_NPU_device']=='npu:4'
            if name=='model.language_model.embed_tokens.weight':own_head=t
        assert own_head is not None and own_head.shape==(100352,2560) and tsha(own_head)==weights['own_tied_head_actual_NPU_SHA256']
        head_F32=own_head.float();config=Granite4VisionConfig.from_pretrained(W,local_files_only=True);rope=GraniteRotaryEmbedding(config=config.text_config,device=torch.device('cpu'));processor=Granite4VisionProcessor.from_pretrained(W,local_files_only=True)
        traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results'];scores=[];empty=[];invalid=[];truncated=[];numeric=[];steps=0
        for rows in [traces,ins,outs,details]:assert [x['index'] for x in rows]==list(range(count))
        for i,compact in enumerate(traces):
            e=js(check(compact['actual_execution_file'],compact['actual_execution_SHA256']));ref=front[i];tokens=e['generated_tokens'];prefix=e['input_ids'];N=len(prefix);n=e['actual_native_forwards'];assert n==len(tokens)-N==len(e['frames'])==compact['actual_native_forwards'] and 0<n<=4096 and e['error'] is None and compact['error'] is None
            assert prefix==ref['input_ids'][0] and tokens[:N]==prefix and 100257 not in tokens[N:-1] and (n==4096 or tokens[-1]==100257)
            assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and e['actual_model_prompt']==ref['original_model_prompt']==GreedyInference._add_answer_instruction(ref['prompt']) and e['formatted_text']==ref['formatted_text']
            assert e['reference']==ref['reference']==ins[i]['reference']==outs[i]['reference']==details[i]['reference'];check(e['media_path'],ref['image_SHA256'])
            raw=processor.tokenizer.decode(tokens[N:],skip_special_tokens=True).strip();assert raw==e['raw_output']==compact['raw_output']==outs[i].get('raw_output',outs[i].get('extra',{}).get('raw_output'))==details[i].get('raw_output',details[i].get('extra',{}).get('raw_output'))
            choice=independent_choice(raw);assert independent_choice(outs[i]['prediction'])==choice and outs[i]['prediction']==details[i]['prediction']
            visited=set();cache=None;max_head_relative=0
            archive=check(e['archive'],e['archive_SHA256']);assert archive.stat().st_size==e['archive_bytes']
            with ReadOnlyNPZ(archive) as a:
                def consume(rec):
                    if rec is None:return None
                    if isinstance(rec,list):return [consume(v) for v in rec]
                    name=rec['array'];v=a[name];assert list(v.shape)==rec['shape'] and bs(v)==rec['SHA256'];expected='torch.bfloat16' if v.dtype==np.uint16 else str(torch.from_numpy(v).dtype);assert rec['dtype']==expected;visited.add(name);return v
                for key,rec in e['inputs'].items():
                    v=consume(rec);assert rec['SHA256']==ref['inputs'][key]['SHA256'] and rec['shape']==ref['inputs'][key]['shape'] and rec['dtype']==ref['inputs'][key]['dtype']
                for k,frame in enumerate(e['frames']):
                    actual=f32(consume(frame['full_head']));x=f32(consume(frame['head_input']));assert actual.shape==(100352,) and x.shape==(2560,) and torch.isfinite(actual).all() and frame['head_argmax']==tokens[N+k]==int(actual.argmax())
                    with torch.inference_mode():expected=(torch.nn.functional.linear(x,head_F32).bfloat16()/10).bfloat16().float()
                    difference=actual-expected;rms=float(difference.square().mean().sqrt());relative=rms/max(float(expected.square().mean().sqrt()),1e-12);assert relative<=.02,(i,k,'Fixed independent full-head bound',relative);max_head_relative=max(max_head_relative,relative)
                    ids=consume(frame.get('input_ids'));assert ids is not None and ids.tolist()==[tokens[:N] if k==0 else tokens[N+k-1:N+k]]
                    mask=consume(frame.get('attention_mask'));assert mask is not None and np.all(mask==1) and mask.shape==(1,N+k)
                    consume(frame.get('position_ids'));consume(frame.get('cache_position'))
                    c=frame['cache'];v=a[c['array']];visited.add(c['array']);assert v.dtype==np.uint16 and bs(v)==c['SHA256'] and c['all_previous_values_bitwise_preserved']
                    if k==0:assert c['kind']=='initial' and v.shape==(80,1,8,N,64);cache=v.copy()
                    else:assert c['kind']=='append_one' and v.shape==(80,1,8,1,64);cache=np.concatenate([cache,v],axis=3)
                    assert list(cache.shape)==c['shape'] and bs(cache)==c['reconstructed_complete_SHA256']==c['actual_complete_NPU_SHA256']
                    assert len(frame['layers'])==40
                    qpos=np.arange(N) if k==0 else np.array([N+k-1]);K=N+k
                    for j,obs in enumerate(frame['layers']):
                        assert obs['name']=='model.language_model.layers.'+str(j);pos=consume(obs['position_ids']);assert pos.tolist()==[qpos.tolist()]
                        actual_cs=consume(obs['position_embeddings']);assert len(actual_cs)==2
                        with torch.inference_mode():expected_cs=rope(torch.empty((1,len(qpos),2560),dtype=torch.bfloat16),torch.tensor([qpos.tolist()]))
                        for actual_part,expected_part in zip(actual_cs,expected_cs):
                            delta=f32(actual_part)-expected_part.float();assert float(delta.abs().max())<=.02 and float(delta.square().mean().sqrt())<=.004,(i,k,j,'Own original analytic RoPE')
                        m=consume(obs['attention_mask'])
                        if m is not None:
                            value=f32(m);assert value.ndim==4 and value.shape[:2]==(1,1) and value.shape[-2]==len(qpos);allowed=m[0,0] if m.dtype==np.bool_ else value.numpy()[0,0]==0;expected_mask=qpos[:,None]>=np.arange(value.shape[-1])[None,:];assert np.array_equal(allowed,expected_mask),(i,k,j,'Original causal full mask')
                assert len(e['full_vision_injection_features'])==8
                for group in e['full_vision_injection_features']:
                    assert group['target_layer'] in [0,3,6,9]
                    for rec in group['features']:v=consume(rec);assert v.shape[-1]==2560 and np.isfinite(f32(v).numpy()).all()
                for name,rec in e['activation_prefill'].items():
                    if i<3:assert 'values' in rec;v=consume(rec['values']);assert list(v.shape)==rec['shape'] and bs(v)==rec['SHA256']
                    assert e['activation_counts'][name]==1
                assert all(e['activation_counts'].get('model.vision_tower.encoder.layers.'+str(j))==1 for j in range(27))
                assert set(a.files)==visited,(set(a.files)-visited,visited-set(a.files))
            score=float(choice==str(ref['reference']).strip());scores.append(score);steps+=n
            if not raw:empty.append(i)
            if not choice:invalid.append(i)
            if e['original_budget_hit']:truncated.append(i)
            numeric.append(dict(index=i,actual_native_forwards=n,all100352_head_rows=n,max_full_head_relative_RMSE=max_head_relative,all40_native_KV_values_losslessly_reconstructed=True,all_original_inputs_masks_RoPE_positions_and8_deepstack_spatial_features_preserved=True))
    profile=final['first_real_vision_prefill_all_profiler_files_SHA256'];assert profile;kernel_count=0
    for name,h in profile.items():check(name,h)
    tasks=[Path(n) for n in profile if re.search(r'/task_time(?:_[^/]+)?\.csv$',n)]
    for t in tasks:
        with t.open(newline='') as f:rows=list(csv.DictReader(f))
        assert all(x['Device_id']=='4' for x in rows);kernel_count+=sum(x['kernel_type'].startswith('KERNEL') and float(x['task_time(us)'])>0 for x in rows)
    assert kernel_count>100,'Actual own vision and prefill kernels required'
    score=sum(scores)/count;assert report['metrics']=={'Accuracy':score} and report['summary']['current_score']==score and report['statistics']['total_samples']==count and report['statistics']['batch_size']==10
    errors={k:abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v)) for k,v in p['current_A100_D_values'].items()};qualification=count==200 and all(v<=Decimal('.05') for v in errors.values())
    return dict(passed=True,model_id=MID,report_dir=str(folder),sample_count=count,metrics={'Accuracy':score},sample_Accuracy=scores,error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=steps,numeric_sample_checks=numeric,all1083_own_source_actual_NPU_SHA_verified=True,all100352_head_rows_independently_recomputed=True,all40_native_KV_values_losslessly_reconstructed=True,all_original_inputs_tokens_raw_references_errors_preserved=True,NPU_profiler_kernel_events=kernel_count,formal_alignment_within_5pct=qualification,original_protocol_final_qualification_complete=qualification,relative_error_to_current_A100_D={k:str(v) for k,v in errors.items()},A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
    import contextlib
    folder=Path(sys.argv[1]).resolve()
    with contextlib.redirect_stdout(sys.stderr):proof=verify(folder,int(sys.argv[2]))
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

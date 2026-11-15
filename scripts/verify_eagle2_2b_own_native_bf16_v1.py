"""Independent own Eagle2-2B complete five reports, source, full head, KV and masks."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,re,csv
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
from scripts.verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113 import independent_choice
MID='nvidia/Eagle2-2B';FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};EXTERNAL={}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(p,h=None):
    p=Path(p);v=sha(p);assert h is None or v==h,(str(p),'SHA');assert str(p) not in EXTERNAL or EXTERNAL[str(p)]==v;EXTERNAL[str(p)]=v;return p
def js(p):return json.loads(Path(p).read_bytes())
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def f32(x):return (x.astype(np.uint32)<<16).view(np.float32) if x.dtype==np.uint16 else x
def verify(folder,count):
    torch.set_num_threads(4);assert count in [3,200] and {p.name for p in folder.iterdir()}==FILES
    assert folder.parent==R/'workspace/reports/nvidia_Eagle2-2B/image-text-to-text' and re.fullmatch(r'\d{8}_\d{6}',folder.name)
    report=js(folder/'eval_report.json');cfg=report['pipeline_config'];assert cfg['model']['name']==MID and cfg['sample_count']==count and cfg['sample_range']==f'0:{count-1}' and cfg['algorithm']['batch_size']==10
    kw=cfg['model']['kwargs'];p=js(check(kw['protocol_file'],kw['protocol_sha256']));assert p['model_id']==MID and p['workbook_D_reference_authorized'] and not cfg['data']['filter_empty']
    assert cfg['output']==dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True)
    for n,h in p['implementation_source_SHA256'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts;check(R/n,h)
    for n,h in p['static_evidence_SHA256'].items():check(n,h)
    W=Path(p['weight_path'])
    for n,h in p['own_assets_SHA256'].items():check(W/n,h)
    for n,v in p['own_weight_files'].items():assert (W/n).stat().st_size==v['bytes'];check(W/n,v['SHA256'])
    check(p['reference_workbook'],p['reference_workbook_SHA256']);check(p['dataset_path'],p['dataset_SHA256']);front=js(check(p['own_frontend200'],p['own_frontend200_SHA256']));assert len(front)==200
    source=js(check(p['own_source_manifest'],p['own_source_manifest_SHA256']));assert len(source)==793 and len({v['runtime_name'] for v in source.values()})==793
    line=js(folder/'operator_lineage.json')['actual_execution'];assert line==report['operator_lineage']['actual_execution'] and line['hardware']=='Ascend910B' and line['devices']==[4] and line['max_new_tokens']==256 and line['head_size']==151674 and line['use_cache']
    for n,h in line['runtime_library_source_SHA256'].items():check(n,h)
    final=js(check(line['final_trace_manifest']));assert final['samples']==count and not final['failed'];assert [v['indices'] for v in final['actual_outer_batches']]==[list(range(i,min(i+10,count))) for i in range(0,count,10)] and all(v['status']=='completed' for v in final['actual_outer_batches'])
    weight=js(check(line['weight_evidence'],line['weight_evidence_SHA256']));assert set(weight['weights'])==set(source) and weight['own_tied_head']
    from safetensors import safe_open
    with safe_open(str(W/'model.safetensors'),framework='pt',device='cpu') as f:
        assert set(f.keys())==set(source)
        for n in f.keys():
            t=f.get_tensor(n);v=weight['weights'][n];assert v['runtime_name']==source[n]['runtime_name'] and str(t.dtype)==v['source_dtype']=='torch.bfloat16' and list(t.shape)==v['shape'] and v['actual_NPU_device']=='npu:4' and bsha(t.view(torch.uint8).numpy())==v['source_SHA256']==v['cast_SHA256']==v['actual_NPU_SHA256']
        embedding=f.get_tensor('language_model.model.embed_tokens.weight');head_weight=f.get_tensor('language_model.lm_head.weight');assert torch.equal(embedding,head_weight)
    inv=1/(1000000.0**(torch.arange(0,128,2).float()/128))
    for n,v in weight['buffers'].items():
        expected=torch.arange(1024).expand(1,-1) if n.endswith('position_ids') else inv
        assert list(expected.shape)==v['shape'] and bsha(expected.contiguous().view(torch.uint8).numpy())==v['CPU_analytic_SHA256']==v['actual_NPU_SHA256']
    from transformers import AutoTokenizer
    from universal_eval.algorithm.inference import GreedyInference
    tokenizer=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=False);assert len(tokenizer)==151674 and tokenizer.eos_token_id==151645
    traces=jl(check(line['sample_trace'],final['sample_trace_SHA256']));ins=jl(folder/'sample_inputs.jsonl');outs=jl(folder/'sample_outputs.jsonl');details=report['detailed_results']
    for rows in [traces,ins,outs,details]:assert [v['index'] for v in rows]==list(range(count))
    scores=[];empty=[];invalid=[];truncated=[];numeric=[];total=0
    for i,small in enumerate(traces):
        e=js(check(small['actual_execution_file'],small['actual_execution_SHA256']));ref=front[i];N=len(ref['input_ids']);tokens=e['generated_tokens'];n=e['actual_native_forwards'];total+=n
        assert len(tokens)==len(e['frames'])==n and 0<n<=256 and e['error'] is None and small['error'] is None
        assert ins[i]['prompt']==details[i]['prompt']==ref['prompt'] and GreedyInference._add_answer_instruction(ref['prompt'])==e['actual_model_prompt']==ref['actual_model_prompt']
        assert ins[i]['reference']==outs[i]['reference']==details[i]['reference']==ref['reference'];check(ref['media_path'],ref['media_file_SHA256'])
        assert n==256 or tokens[-1]==151645;assert 151645 not in tokens[:-1]
        raw=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip();choice=independent_choice(raw)
        assert raw==e['raw_output']==outs[i]['raw_output'] and (choice or '')==outs[i]['prediction']==details[i]['prediction']
        with ReadOnlyNPZ(check(e['archive'],e['archive_SHA256'])) as a:
            assert a['input_ids'].tolist()==[ref['input_ids']] and bsha(a['pixels_BF16'])==ref['pixels_BF16_SHA256']
            full=a['all_original_full_head_logits'];hidden=a['all_head_inputs'];assert full.shape==(n,151674) and full.dtype==np.uint16 and hidden.shape==(n,1536) and hidden.dtype==np.uint16 and np.isfinite(f32(full)).all() and f32(full).argmax(axis=1).tolist()==tokens
            merged=a['prefill_inputs_embeds'];visual=a['own_visual_features'];mask=np.asarray([ref['input_ids']])==151667;assert int(mask.sum())==ref['tiles']*256 and merged.shape==(1,N,1536) and visual.shape==(ref['tiles'],256,1536)
            assert np.array_equal(merged[mask],visual.reshape(-1,1536));text=embedding[torch.tensor(ref['input_ids'])].view(torch.uint16).numpy();assert np.array_equal(merged[~mask],text[~mask[0]])
            cache=a['native_cache_prefill'];delta=a['native_cache_decode_delta'];assert cache.shape==(28,2,2,N,128) and delta.shape==(28,2,2,n-1,128) and cache.dtype==delta.dtype==np.uint16
            for step,frame in enumerate(e['frames']):
                assert frame['step']==step and frame['cache_seq_len']==N+step and frame['unchanged_prefix_exact_checked_layers']==(28 if step else 0)
                assert frame['head_SHA256']==bsha(full[step]) and frame['argmax']==tokens[step] and frame['head_dtype']=='torch.bfloat16'
                pos=list(range(N)) if not step else [N+step-1]
                assert [v['layer'] for v in frame['attention']]==list(range(28))
                if frame.get('position_ids_values') is not None:assert frame['position_ids_values']==[pos]
                for layer in frame['attention']:
                    refs=layer.get('attention_mask')
                    if refs:
                        for rec in refs:
                            t=a[rec['array']];assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'];actual=f32(t);assert actual.shape==(1,1,len(pos),N+step);want=np.asarray(pos)[:,None]>=np.arange(N+step)[None,:];assert np.array_equal(actual[0,0]==0,want)
                    refs=layer.get('position_embeddings')
                    assert refs is not None and len(refs)==2
                    for j,rec in enumerate(refs):
                        t=a[rec['array']];assert list(t.shape)==rec['shape'] and bsha(t)==rec['SHA256'];phase=torch.tensor(pos).float()[:,None]*inv[None,:];phase=torch.cat((phase,phase),-1);want=(phase.cos() if j==0 else phase.sin()).numpy()[None];assert np.max(np.abs(f32(t)-want))<.02
            complete=np.concatenate([cache,delta],axis=3)
            for layer in range(28):
                for kind in range(2):assert bsha(complete[layer,kind][None])==e['final_cache_SHA256'][layer][kind]
            for name,rec in e['activation_prefill'].items():
                if name in a.files:assert list(a[name].shape)==rec['shape'] and bsha(a[name])==rec['SHA256']
            assert len([name for name in e['activation_prefill'] if name.startswith('stage_')])==55
            assert all(v==1 for k,v in e['activation_counts'].items() if k.startswith('vision_model.')) and all(v==n for k,v in e['activation_counts'].items() if k.startswith('language_model.'))
            head_reviews=[]
            for step in sorted({0,n-1}):
                h=torch.from_numpy(hidden[step].copy()).view(torch.bfloat16);expected=torch.nn.functional.linear(h,head_weight).float().numpy();actual=f32(full[step]);rmse=float(np.sqrt(np.mean((actual-expected)**2)));scale=float(np.sqrt(np.mean(expected**2)));assert rmse/max(scale,1e-6)<.02
                head_reviews.append(dict(step=step,CPU_BF16_reference_relative_RMSE=rmse/max(scale,1e-6),CPU_argmax=int(expected.argmax()),actual_NPU_argmax=tokens[step],CPU_diagnostic_not_a_grade=True))
        scores.append(choice==ref['reference']);empty.extend([i] if not raw else []);options=re.findall(r'(?m)^\s*([A-E])\s*[.)]\s+',ref['prompt']);invalid.extend([i] if choice not in options else []);truncated.extend([i] if e['original_budget_hit'] else []);numeric.append(dict(index=i,full_head_rows=n,all28_native_KV_reconstructed=True,all_masks_positions_checked=True,own_source_visual_injection_exact=True,head_reference_checks=head_reviews))
    events=0
    for path,h in final['first_real_vision_prefill_profiler_SHA256'].items():
        f=check(path,h)
        if f.name=='task_time.csv' or re.search(r'/task_time_[^/]+\.csv$',path):
            rows=list(csv.DictReader(f.open()));assert all(v['Device_id']=='4' for v in rows);events+=sum(v['kernel_type'].startswith('KERNEL') for v in rows)
    assert events>100
    score=sum(scores)/count;assert report['metrics']=={'Strict_MC_Accuracy':score} and report['statistics']['total_samples']==count and report['statistics']['error_count']==0
    errors={k:str(abs(Decimal(str(score))-Decimal(v))/abs(Decimal(v))) for k,v in p['current_A100_D_values'].items()};qualified=count==200 and all(Decimal(v)<=Decimal('.05') for v in errors.values())
    return dict(passed=True,report_dir=str(folder),sample_count=count,metrics={'Strict_MC_Accuracy':score},error_indices=[],empty_indices=empty,invalid_indices=invalid,truncated_indices=truncated,actual_NPU_inference=True,NEW_real_NPU_inference=True,historical_record_reuse=False,actual_forward_count=total,all_own_source_actual_NPU_SHA256_verified=True,source_tensor_count=793,all_native_KV_values_losslessly_reconstructed=True,all_full_inputs_tokens_raw_references_errors_preserved=True,numeric_sample_checks=numeric,NPU_profiler_kernel_events=events,relative_error_to_A100_table_D=float(errors['server']),relative_error_to_current_A100_D=errors,formal_alignment_within_5pct=qualified,original_protocol_final_qualification_complete=qualified,qualification_basis='User workbook D; saved own Eagle2 deployment and complete NPU200 audit',A100_historical_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],independent_verifier_SHA256=sha(__file__),report_sha256={n:sha(folder/n) for n in sorted(FILES)},external_evidence_sha256=EXTERNAL)
if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve();result=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['weight_evidence']).parent;out=home/'independent_report_verification.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

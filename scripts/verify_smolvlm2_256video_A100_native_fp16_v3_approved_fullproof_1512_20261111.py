"""Independent standalone report and native FP16 trace acceptance for own SmolVLM2."""
import csv,hashlib,json,re,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
from strict_mc_independent_reference_910b import independent_choice
MODEL='HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
FILES={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(l) for l in Path(path).read_bytes().splitlines() if l.strip()]
def rel(score,ref):
    ref=Decimal(str(ref));score=Decimal(str(score))
    return abs(score-ref)/abs(ref) if ref else Decimal(0) if not score else Decimal('Infinity')
def verify(folder,n):
    assert n in (3,10,200)
    assert folder.parent.parent.name==MODEL.replace('/','_') and re.fullmatch(r'\d{8}_\d{6}',folder.name)
    assert {p.name for p in folder.iterdir()}==FILES
    report=read(folder/'eval_report.json');cfg=report['pipeline_config'];args=cfg['model']['kwargs']
    assert cfg['model']['name']==MODEL and cfg['sample_range']==f'0:{n-1}' and cfg['algorithm']['batch_size']==10
    assert cfg['algorithm']['name']=='native_explicit_mc_v2' and cfg['algorithm']['kwargs']['extract_answer'] is True
    assert cfg['output']=={'reports_dir':'workspace/reports','layout':'model_task_timestamp',
        'evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}
    assert not cfg['data']['filter_empty'] and not cfg['data']['download'] and not cfg['model']['download']
    assert sha(args['protocol_file'])==args['protocol_sha256']
    protocol=read(args['protocol_file']);assert protocol['model_id']==MODEL
    assert read(protocol['full200_input_comparison_review'])['all_input_tensors_bitwise_equal'] is True
    approval=None
    if n==200:
        approval=read(protocol['planned_alignment_approval_review'])
        assert approval['approved'] and approval['model_id']==MODEL
        assert approval['actual_3_and_original10_within5pct'] and approval['complete_input200_bitwise_verified']

    external={args['protocol_file']:args['protocol_sha256']}
    if approval is not None:
        external[protocol['planned_alignment_approval_review']]=sha(protocol['planned_alignment_approval_review'])
        for p,h in approval['prerequisite_artifacts_sha256'].items():
            assert sha(p)==h
            external[p]=h
        numerical=read(approval['numerical_proof'])
        assert sha(approval['numerical_proof'])==approval['numerical_proof_sha256'] and numerical['passed']
        assert numerical['complete_input200_bitwise_verified'] and numerical['all3_tokens_and_raw_equal']
        for p,h in numerical['external_evidence_sha256'].items():
            assert sha(p)==h
            external[p]=h

    for p,h in protocol['static_evidence_sha256'].items():assert sha(p)==h;external[p]=h
    original=read(protocol['original_A100_deployment'])
    assert len(original['cohort'])==200 and original['original_metric']['Accuracy']==.335
    assert sum(independent_choice(x['raw_output'])==x['reference'] for x in original['cohort'])==67
    cpu=read(protocol['R_independent_CPU200'])['CPU200']
    aa=read(protocol['actual_A100_native_controls'])
    for i in range(3):
        assert all(aa['CPU200'][i][k]==cpu[i][k] for k in aa['CPU200'][i])
    line=read(folder/'operator_lineage.json')['actual_execution']
    assert line['device_type']=='npu' and line['dtype']=='torch.float16'
    assert line['original_A100_outer_batch']==10 and line['actual_model_inner_batch']==1
    assert line['source_parameter_count']==471 and line['quantization'] is None
    assert line['protocol_sha256']==args['protocol_sha256']
    evidence=Path(line['evidence_directory']);assert evidence.is_relative_to(Path.cwd()/'workspace/diagnostics/model_runs')
    assert sha(line['source_manifest'])==line['source_manifest_SHA256']
    manifest=read(line['source_manifest'])
    assert manifest['source_tensor_count']==471
    aa_params=aa['source_cast_actual_CUDA']['source_to_actual']
    for key,rec in manifest['source_to_actual'].items():
        assert rec['actual_device_type']=='npu' and rec['source']==aa_params[key]['source'] and rec['cast']==aa_params[key]['cast']
    trace=rows(line['request_trace']);ins=rows(folder/'sample_inputs.jsonl');outs=rows(folder/'sample_outputs.jsonl')
    details=report['detailed_results'];assert len(trace)==len(ins)==len(outs)==len(details)==n
    forwards=read(line['whole_forward_manifest']);stages=read(line['block_positions_masks_manifest'])
    assert len(forwards)==sum(r['actual_forward_count'] for r in trace)
    idx=read(line['array_index']);archive_cache={}
    def array(record):
        key=record['array_key'];path=idx['keys'][key]
        if path not in archive_cache:archive_cache[path]=np.load(path,allow_pickle=False)
        x=archive_cache[path][key]
        assert list(x.shape)==record['shape'] and hashlib.sha256(x.tobytes()).hexdigest()==record['SHA256']
        return x
    score=0;empty=[];invalid=[];truncated=[]
    for i,(a,b,c,t) in enumerate(zip(ins,outs,details,trace)):
        old=original['cohort'][i]
        assert a['index']==b['index']==c['index']==t['index']==i
        assert a['prompt']==old['prompt'] and a['reference']==b['reference']==c['reference']==old['reference']
        assert b['error'] is None and b['raw_output']==c['raw_output']==t['raw_output']
        chosen=independent_choice(t['raw_output'])
        assert b['prediction']==c['prediction']
        assert b['prediction']==chosen if chosen is not None else b['prediction'] in (None,t['raw_output'])
        assert Path(a['media_path']).name==old['media_basename'] and sha(a['media_path'])==old['media_SHA256']
        assert t['prompt_token_ids']==cpu[i]['input_ids']
        assert t['inner_model_batch']==1 and t['group_index']==i//10 and t['group_size']==min(10,n-10*(i//10))
        assert t['rendered_text']==cpu[i]['text'] and t['prompt'].endswith('in the format <answer>X</answer>.')
        tokens=t['generated_token_ids'];assert 0<len(tokens)<=2048 and all(0<=v<49280 for v in tokens)
        fs=[f for f in forwards if f['index']==i];assert len(fs)==len(tokens)
        assert [f['forward_index'] for f in fs]==list(range(len(tokens)))
        for j,f in enumerate(fs):
            assert f['arguments']['use_cache'] is True and f['whole_vocabulary_size']==49280
            logits=array(f['whole_logits']);assert logits.shape==(1,1,49280)
            assert int(logits[0,-1].astype(np.float64).argmax())==tokens[j]==f['last_position_argmax'][0]
            assert len(f['native_cache'])==30
            expect=len(t['prompt_token_ids'])+j
            assert all(k['K_full']['shape']==[1,3,expect,64] and k['V_full']['shape']==[1,3,expect,64] for k in f['native_cache'])
        if not t['raw_output'].strip():empty.append(i)
        if independent_choice(t['raw_output']) is None:invalid.append(i)
        if t['finish_reason']=='length':truncated.append(i);assert len(tokens)==2048
        else:assert t['finish_reason']=='stop' and tokens[-1]==49279
        score+=independent_choice(t['raw_output'])==old['reference']
    score/=n
    assert abs(report['metrics']['Accuracy']-score)<1e-12
    assert report['statistics']['total_samples']==n and report['statistics']['error_count']==0
    assert len({t['request_id'] for t in trace})==n
    kernels={}
    for f in evidence.rglob('*'):
        if not f.is_file():continue
        external[str(f)]=sha(f)
        if f.name in ('kernel_details.csv','op_summary.csv'):
            with f.open(encoding='utf-8-sig',newline='') as g:count=sum(1 for _ in csv.DictReader(g))
            if count:kernels[str(f)]=count
    assert sum(kernels.values())>0
    same=aa['actual_3_and_original10'][str(n)] if n in (3,10) else original['cohort']
    aa_score=sum(independent_choice(r['raw_output'])==r['reference'] for r in same)/n
    differences={k:rel(score,v) for k,v in protocol['A100_D'].items()}
    return {'passed':True,'report_dir':str(folder),'sample_count':n,'metrics':{'Accuracy':score},
        'actual_NPU_inference':True,'historical_record_reuse':False,
        'report_sha256':{name:sha(folder/name) for name in sorted(FILES)},'external_evidence_sha256':external,
        'source_cast_actual_NPU_tensor_count':471,'generated_token_count':sum(len(t['generated_token_ids']) for t in trace),
        'actual_forward_count':len(forwards),'whole_vocabulary_size':49280,'native_cache_layers':30,
        'NPU_profiler_kernel_events':sum(kernels.values()),'kernel_counts':kernels,
        'error_indices':[],'empty_indices':empty,'invalid_indices':invalid,'truncated_indices':truncated,
        'actual_same_A100_accuracy':aa_score,'actual_same_A100_relative_error':str(rel(score,aa_score)),
        'raw_output_matches_archived_A100':sum(t['raw_output']==original['cohort'][i]['raw_output'] for i,t in enumerate(trace)),
        'raw_output_matches_actual_A100':sum(t['raw_output']==same[i]['raw_output'] for i,t in enumerate(trace)),
        'relative_error_exact_three_D':{k:str(v) for k,v in differences.items()},
        'relative_error_to_A100_table_D':float(differences['current_remote']),'formal_alignment_within_5pct':n==200 and all(x<=Decimal('.05') for x in differences.values()),
        'numerical_alignment_approved':approval is not None,'requires_separate_same_A100_numerical_review':approval is None,
        'A100_reference_scope':'current_genuine_control' if n in (3,10) else 'selected_archived_actual200',
        'pending_full200_pixel_precast_indices':protocol['pending_full200_pixel_precast_indices']}
if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1]).resolve(),int(sys.argv[2])),ensure_ascii=False))


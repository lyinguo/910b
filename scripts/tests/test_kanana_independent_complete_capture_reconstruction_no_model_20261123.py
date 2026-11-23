"""Validate new independent capture reconstruction, without producer replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib,json,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
PARENT=B/'Kanana_complete_native_observer_synthetic_values_and_real_meta_registration_tests_v1_20261123'
H=B/'Kanana_independent_complete_capture_and_full32KV_reconstruction_synthetic_only_v1_20261123'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    H.mkdir(exist_ok=False);sys.path[:0]=[str(R),str(R/'scripts')]
    import torch,numpy as np
    m=importlib.import_module('universal_eval.model.kanana_own_original_graphtext_v2_20261123.independent_complete_capture_values_20261123')
    ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());canonical={p:sha(p) for p in ptr['canonical_SHA256']};assert canonical==ptr['canonical_SHA256']
    assert sha(PARENT/'review.json')=='3ea092c400e335679619c6d951f327a30d62a8bb4933a214b7129da76d76a181'
    review=json.loads((PARENT/'review.json').read_bytes());assert review['passed'] and review['check_count']==14
    reports=[];checks=[]
    records=[]
    for capture in review['complete_synthetic_captures']:
        path=Path(capture['path']);assert sha(path)==capture['SHA256']
        reader=m.IndependentCompleteCapture(path);events=reader.decoded_events();reports.append(reader.summary());records.append((reader,events))
    checks.append('Both complete standard archives and every unique value are independently read and SHA-verified')
    first,events=records[0];values={row['name']:row['value'] for row in events}
    expected=np.array([0,0x8000,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1,0x7f81],dtype=np.uint16)
    for name in ['all_BF16_bit_patterns','second_independent_occurrence']:
        actual=values[name];assert actual.dtype==torch.bfloat16 and actual.shape==(2,4)
        assert np.array_equal(actual.view(torch.uint16).reshape(-1).numpy(),expected)
    checks.append('All BF16 NaN payload, infinity, signed-zero and finite values reconstruct bit-for-bit')
    assert values['same_bits_other_shape'].shape==(1,8) and values['same_bits_FP16'].dtype==torch.float16
    checks.append('Shared raw strings retain distinct FP16/BF16 and shape semantics')
    values['all_BF16_bit_patterns'].zero_()
    assert np.array_equal(values['second_independent_occurrence'].view(torch.uint16).reshape(-1).numpy(),expected)
    checks.append('Independent reconstruction cannot mutate a different semantic occurrence')
    assert values['rank_zero'].shape==torch.Size([]) and int(values['rank_zero'])==-7 and values['zero_elements'].shape==(0,8)
    checks.append('Scalar rank and zero-element axes survive independent reconstruction')
    assert torch.equal(values['noncontiguous_source'],torch.arange(36,dtype=torch.float32).reshape(6,6)[:,::2])
    checks.append('All values from the original strided source reconstruct in exact logical order')
    head=values['complete_128259_head'];assert head.shape==(1,1,128259) and int(torch.count_nonzero(head))==1 and float(head[...,128258])==3
    checks.append('Every column of the original full 128259 head is present, including the last column')
    before=values['real_HF_full32KV_before_append']['attributes']['layers']
    after=values['real_HF_full32KV_after_append']['attributes']['layers']
    assert len(before)==len(after)==32
    for index,(a,b) in enumerate(zip(before,after)):
        for key in ['keys','values']:
            x=a['attributes'][key];y=b['attributes'][key]
            assert x.shape==(1,8,7,128) and y.shape==(1,8,8,128)
            assert torch.equal(x.view(torch.uint16),y[...,:7,:].contiguous().view(torch.uint16))
        assert torch.all(b['attributes']['keys'][...,-1,:]==torch.tensor(2+index/64,dtype=torch.bfloat16))
        assert torch.all(b['attributes']['values'][...,-1,:]==torch.tensor(-2-index/64,dtype=torch.bfloat16))
    checks.append('All 32 complete physical K/V prefixes and appended values independently reconstruct and agree exactly')
    assert torch.all(before[0]['attributes']['keys']==torch.tensor(.125,dtype=torch.bfloat16))
    checks.append('Later mutation of the live original cache has not altered the saved full prefix')
    event=copy.deepcopy(first.manifest['events'][0]['value']);event['dtype']='torch.float16'
    try:first.tensor(event)
    except ValueError:checks.append('A forged logical dtype occurrence is rejected')
    else:raise AssertionError('Forged dtype accepted')
    event=copy.deepcopy(first.manifest['events'][0]['value']);event['shape']=[1,8]
    try:first.tensor(event)
    except ValueError:checks.append('A forged shape occurrence is rejected')
    else:raise AssertionError('Forged shape accepted')
    second,events=records[1]
    assert [row['phase'] for row in events]==['forward_pre','forward_post'] and events[0]['call_id']==events[1]['call_id']
    assert events[0]['value']['args'][0].shape==(2,8) and events[1]['value'].shape==(2,4)
    checks.append('The complete native pre/post values and exact call pairing are independently recoverable')
    assert {p:sha(p) for p in canonical}==canonical
    result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),check_count=len(checks),checks=checks,complete_capture_summaries=reports,
        producer_14_checks_not_replayed=True,source_sha256={Path(m.__file__).relative_to(R).as_posix():sha(m.__file__),Path(__file__).relative_to(R).as_posix():sha(__file__)},
        no_checkpoint_tensor_loaded=True,learned_CPU_GPU_NPU_forwards=0,actual_NEW_model_forwards=0,formal_NEW_NPU3=False,
        formal_fullNF_or_200_or_qualification=False,model_arithmetic_reconstruction_and_NPU_loader_and_YAML_still_required=True,all_canonical_unchanged=True)
    put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,capture_format_reconstruction_only=True,no_model_NN_or_fullNF=True))
    print(json.dumps(dict(passed=True,UTC=result['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),check_count=len(checks),capture_summaries=reports,no_original_model_or_fullNF_or_score=True),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_NN_or_fullNF=True))
        raise

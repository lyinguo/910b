"""Bind the completed original input proof with relative source/external scopes.

Read-only acceptance of CPU inputs only. No pixel, token or model replay, no
new hardware score and no change to the original failure or success receipts.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import hashlib,json,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
PARENT=B/'Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123'
H=B/'Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v3_20261123'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,z):(H/n).write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    H.mkdir(exist_ok=False)
    ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');canonical={p:sha(p) for p in ptr['canonical_SHA256']};assert canonical==ptr['canonical_SHA256']
    assert sha(PARENT/'review.json')=='db892b6f4e4948825d55d5d75208454a7360e56639b7c0fdda3acc8b483d922f'
    old=js(PARENT/'review.json');done=js(PARENT/'actual_completion.json')
    assert done['actual_returncode']==0 and done['frontend_only'] and done['no_NN_or_fullNF']
    assert old['passed'] and old['samples']==200 and old['all200_original_F32_pixel_bits_and_tokens_independently_equal']
    assert old['learned_CPU_GPU_NPU_forwards']==0 and not old['formal_NEW_NPU3'] and not old['formal_fullNF_or_200_or_qualification']
    front=Path(old['frontend']);assert front==PARENT/'own_original_frontend200.jsonl' and sha(front)==old['frontend_SHA256']
    rows=[json.loads(s) for s in front.read_text().splitlines()]
    assert [r['index'] for r in rows]==list(range(200)) and all(r['all_original_pixel_bits_and_input_tokens_independently_equal'] and r['no_model_forward'] for r in rows)
    sources={};external={}
    for p,h in old['source_SHA256'].items():
        path=Path(p);assert sha(path)==h
        if path.is_relative_to(R) and path.relative_to(R).parts[0] in ['universal_eval','scripts','configs']:
            sources[path.relative_to(R).as_posix()]=h
        else:external[str(path)]=h
    for p,h in old['external_assets_SHA256'].items():assert sha(p)==h;external[p]=h
    for n in ['run.py','universal_eval/pipeline.py','configs/KananaVForConditionalGeneration/kakaocorp--kanana-1.5-v-3b-instruct.yaml',
              'universal_eval/model/adapters/high_nine_kanana_canonical.py','universal_eval/model/adapters/high_nine_kanana_reference.py',
              'scripts/prepare_kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123.py',
              'scripts/prepare_kanana_own_original_BF16_dedicated_frontend200_v2_original_whitespace_20261123.py',
              Path(__file__).relative_to(R).as_posix()]:sources[n]=sha(R/n)
    external[str(front)]=sha(front);external[str(PARENT/'review.json')]=sha(PARENT/'review.json')
    external[str(PARENT/'actual_completion.json')]=sha(PARENT/'actual_completion.json')
    for p in sorted(PARENT.glob('necessary_original_input_example_*.npz')):external[str(p)]=sha(p)
    assert len(list(PARENT.glob('necessary_original_input_example_*.npz')))==3
    gen=js(PARENT/'native_processor_and_original_generation_contract.json')
    assert gen['current_C_unique']['row']==379 and gen['current_C_unique']['cells']['C']=='kakaocorp/kanana-1.5-v-3b-instruct'
    assert gen['current_C_unique']['cells']['G']=='' and all(Decimal(x)==Decimal('.89') for x in gen['current_three_A100_D'].values())
    assert gen['original_generate']==dict(max_new_tokens=2048,do_sample=False,temperature=0.,eos_token_id=128009,pad_token_id=128001)
    assert gen['original_dedicated_input_max_length']==8192 and gen['original_batch_size']==1
    assert all(not n.startswith('/') and Path(n).parts[0] in ['universal_eval','scripts','configs','run.py'] for n in sources)
    assert {p:sha(p) for p in canonical}==canonical
    review=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),acceptance_scope='Completed independent original CPU input frontend only',
        model_id='kakaocorp/kanana-1.5-v-3b-instruct',samples=200,source_sha256=sources,external_evidence_sha256=external,
        original_input_proof=str(PARENT/'review.json'),original_input_proof_SHA256=sha(PARENT/'review.json'),
        frontend=str(front),frontend_SHA256=sha(front),current_C_unique_row=379,current_three_A100_D=gen['current_three_A100_D'],
        all200_complete_original_input_tokens_and_independent_pixel_SHA_preserved=True,all_three_complete_input_examples_preserved=True,
        actual_NEW_model_forwards=0,actual_NEW_frontend_computations=0,actual_NPU_inference=False,formal_fullNF_or_200_or_qualification=False,
        no_model_adapter_or_formal_dispatch_approved=True,no_G_or_canonical_write=True,no_original_success_failure_or_value_mutation=True)
    put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,frontend_acceptance_only=True,no_model_NN_or_fullNF=True))
    print(json.dumps(dict(passed=True,UTC=review['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),samples=200,frontend_SHA256=sha(front),
                         current_C_unique_row=379,current_three_A100_D=gen['current_three_A100_D'],relative_source_count=len(sources),external_SHA_count=len(external),no_replay_or_NPU_score=True),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_model_NN_or_fullNF=True))
        raise

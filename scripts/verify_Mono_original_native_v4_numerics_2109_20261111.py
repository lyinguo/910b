"""Independent final Mono verifier with original reference and disclosed numerical control binding."""
import hashlib,json,sys
from pathlib import Path
from verify_Mono_original_native_v3_evidence_scope_2109_20261111 import verify as parent_verify,js,check,sha,EXTERNAL,R,MID
from strict_mc_independent_reference_910b import independent_choice
def verify(folder,count):
    proof=parent_verify(folder,count);cfg=js(folder/'eval_report.json')['pipeline_config'];p=js(cfg['model']['kwargs']['protocol_file'])
    nr=js(check(p['numerical_review'],p['numerical_review_SHA256']));assert nr['passed'] and nr['full_head_rows']==57 and nr['same3_original_raw_exact'] and nr['relative_error_same3']=='0'
    assert nr['own253_source_cast_A100_NPU_checked_by_complete_independent_proof'] and nr['visual_token_mask_original_BF16_preserved']
    for path,h in nr['external_evidence_SHA256'].items():
        path=Path(path);root=Path(p['weight_path']) if path.is_relative_to(Path(p['weight_path'])) else R
        check(path,h,root=root)
    original=js(check(p['original_AA_fixed200_rows'],p['original_AA_fixed200_rows_SHA256']));rows=original['rows'];assert len(rows)==200
    assert sum(independent_choice(x['raw_output'])==str(x['reference']).strip().upper() for x in rows)==167
    r=js(folder/'eval_report.json');changed=[]
    for i,row in enumerate(r['detailed_results']):
        old=rows[i];assert old['index']==i and row['prompt']==old['original_prompt'] and row['reference']==old['reference']
        if old['raw_output']!=row['raw_output']:changed.append(i)
    proof.update(original_A100_stored_and_same_raw_strict_accuracy='0.835',original_A100_and_new_raw_different_indices=changed,own_same_count_A100_control_accuracy=sum(independent_choice(x['raw_output'])==str(x['reference']).strip().upper() for x in rows[:count])/count,numerical_disclosure=nr['nonzero_numerical_differences_disclosed'],numerical_control_scope='all57 full92553 head rows in genuine same3 control; no claim whole model CPU/FP32 or bitwise A100/NPU equality',source_own_nonLoRA_native_computation_AST_and_original_cfg_verified=True,independent_verifier_SHA256=sha(__file__),external_evidence_sha256=EXTERNAL)
    return proof
if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve();proof=verify(folder,int(sys.argv[2]));home=Path(js(folder/'operator_lineage.json')['actual_execution']['evidence_root']);path=home/'independent_report_verification.json'
    if path.exists():assert js(path)==proof
    else:path.write_text(json.dumps(proof,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))

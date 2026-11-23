"""Verify private source/interface without loading a checkpoint or model score."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,importlib,json,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Kanana_own_original_real_RegStage_cross_version_CPU_drift_diagnostic_and739_meta_only_v3_20261123'
sys.path[:0]=[str(R)]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    plan=Path(sys.argv[1])
    def load_overlay(p,h):
        assert sha(p)==h;z=json.loads(p.read_bytes())
        if 'parent_control_plan' in z:
            parent=load_overlay(Path(z['parent_control_plan']),z['parent_control_plan_SHA256']);parent.update(z);return parent
        return z
    q=load_overlay(plan,sys.argv[2])
    H.mkdir(exist_ok=False)
    ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
    for n,h in q['new_primary_source_SHA256'].items():assert sha(R/n)==h,n
    import torch
    torch.set_num_threads(4);torch._C._set_graph_executor_optimize(False)
    compat=importlib.import_module('universal_eval.model.kanana_own_original_graphtext_v2_20261123.original_parameterized_RegStage_compat_20261123')
    def tensor(encoded,shape):
        b=base64.b64decode(encoded,validate=True)
        return torch.frombuffer(bytearray(b),dtype=torch.bfloat16).reshape(shape)
    def bits(t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    checks=[]
    for stage in q['complete_AA_synthetic_graph_controls']['stages']:
        m=compat.RegStage(2,stage['input_channels'],1024,stride=1,dilation=1,act_layer=torch.nn.SiLU,norm_layer=compat.LayerNorm2d).eval()
        with torch.no_grad():
            for p in stage['parameter_inputs']:
                t=tensor(p['complete_synthetic_parameter_raw_base64'],p['shape'])
                assert hashlib.sha256(bits(t)).hexdigest()==p['synthetic_control_SHA256']
                m.get_parameter(p['relative_key']).copy_(t)
        for c in stage['CPU_synthetic_controls']:
            x=tensor(c['complete_synthetic_input_raw_base64'],c['input_shape'])
            y=tensor(c['complete_synthetic_output_raw_base64'],c['output_shape'])
            assert hashlib.sha256(bits(x)).hexdigest()==c['input_SHA256']
            assert hashlib.sha256(bits(y)).hexdigest()==c['native_SHA256']==c['graph_SHA256']
            with torch.inference_mode():out=m(x)
            equal=torch.equal(out,y);delta=out.double()-y.double();rmse=float(delta.square().mean().sqrt());denom=float(y.double().square().mean().sqrt())
            checks.append(dict(stage=stage['name'],input_shape=c['input_shape'],output_shape=list(out.shape),
                all_raw_bits_identical_to_original_AA_synthetic_control=equal,max_abs=float(delta.abs().max()),RMSE=rmse,relative_RMSE=rmse/denom if denom else None,SHA256=hashlib.sha256(bits(out)).hexdigest(),complete_actual_CPU_BF16_output_raw_base64=base64.b64encode(bits(out)).decode()))
    put('complete_synthetic_CPU_control_review.json',dict(diagnostic_only=True,passed_full_raw_bit_control=all(c['all_raw_bits_identical_to_original_AA_synthetic_control'] for c in checks),torch_version=torch.__version__,checks=checks,
        no_checkpoint_values_loaded=True,no_learned_CPU_GPU_NPU_forward=True))
    own=importlib.import_module('universal_eval.model.kanana_own_original_graphtext_v2_20261123.configuration')
    modeling=importlib.import_module('universal_eval.model.kanana_own_original_graphtext_v2_20261123.modeling')
    W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/KananaVForConditionalGeneration/kakaocorp/kanana-1.5-v-3b-instruct')
    assert sha(W/'config.json')=='500c82ccfd5f8368c7a4dbbf3af13139a379db853a21a85c42df42f127d81e52'
    cfg=own.KananaVConfig(**json.loads((W/'config.json').read_bytes()))
    cfg.vision_config._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa'
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('meta'):model=modeling.KananaVForConditionalGeneration(cfg).eval()
    finally:torch.set_default_dtype(previous_dtype)
    headers=q['own_header_metadata'];state=model.state_dict()
    assert set(state)==set(headers),(sorted(set(state)-set(headers)),sorted(set(headers)-set(state)))
    mapped={}
    for name,t in state.items():
        assert t.device.type=='meta' and t.dtype==torch.bfloat16 and list(t.shape)==headers[name]['shape'],(name,list(t.shape),str(t.dtype))
        mapped[name]=dict(shape=list(t.shape),dtype=str(t.dtype),device='meta')
    assert len(mapped)==739
    put('all739_own_header_to_actual_meta_parameter_map.json',mapped)
    result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),own_model_id='kakaocorp/kanana-1.5-v-3b-instruct',
        source_and_meta_interface_only=True,not_numeric_acceptance=True,passed_full_raw_bit_control=all(c['all_raw_bits_identical_to_original_AA_synthetic_control'] for c in checks),complete_original_RegStage_synthetic_controls=checks,
        physical_own_BF16_source_tensor_headers=739,all739_actual_meta_shapes_and_dtypes_match=True,
        no_checkpoint_values_loaded=True,learned_CPU_GPU_NPU_forwards=0,formal_NEW_NPU3=False,
        formal_fullNF_or_200_or_qualification=False,no_new_adapter_or_formal_dispatch_approved=True,
        own_original_model_math_functions_AST_unchanged=True,only_private_real_timm_imports_changed=True,
        source_SHA256=q['new_primary_source_SHA256'],canonical_SHA256=ptr['canonical_SHA256'])
    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
    put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,source_meta_only=True))
    print(json.dumps(dict(passed=True,UTC=result['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),
        synthetic_controls=len(checks),full_raw_bit_control_passed=result['passed_full_raw_bit_control'],control_summary=[{k:c[k] for k in ['stage','input_shape','all_raw_bits_identical_to_original_AA_synthetic_control','max_abs','RMSE','relative_RMSE']} for c in checks],own739_meta_matched=True,no_learned_forward_or_formal_qualification=True),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_formal_model_score=True))
        raise

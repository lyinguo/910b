"""Exercise real original RegStage graphs on NPU with untrained operands only.

This is an interface diagnostic. It never loads checkpoint values, evaluates
model samples or claims model numerical acceptance. Preserve all discrepancies.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,fcntl,hashlib,importlib,json,os,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Kanana_original_parameterized_RegStage_actual_NPU_synthetic_interface_diagnostic_v4_20261123'
PLAN=D/'heartbeat_handoff_plans/Kanana_private_original_RegStage_complete_synthetic_control_and739_meta_v2_original_decorators_plan_20261123.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    H.mkdir(exist_ok=False);os.chdir(R);sys.path[:0]=[str(R),str(R/'scripts')]
    from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
    assert not runner_lock_owners()
    assert not [p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
    ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
    before={p:sha(p) for p in ptr['canonical_SHA256']};assert before==ptr['canonical_SHA256']
    guard=(D/'runner.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert [p['PID'] for p in runner_lock_owners()]==[os.getpid()]
    check=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
    assert check.returncode==0 and 'No running processes found in NPU 4' in check.stdout
    put('actual_admission.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),actual_cwd=str(Path.cwd()),
        actual_argv=(Path('/proc')/str(os.getpid())/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
        original_kernel_FLOCK=runner_lock_owners(),card=4,npu_smi_before=check.stdout,checkpoint_tensors_loaded=0,no_model_forward=True))
    assert sha(PLAN)=='02d4ae1b5dae2dd07db8286ff7b3d53d8945b92a35d5952b2391ff759d62e26e'
    q=json.loads(PLAN.read_bytes());controls=q['complete_AA_synthetic_graph_controls']
    import torch,torch_npu,numpy as np
    torch.set_num_threads(4);torch._C._set_graph_executor_optimize(False);torch.npu.set_device('npu:4')
    mod=importlib.import_module('universal_eval.model.kanana_own_original_graphtext_v2_20261123.original_parameterized_RegStage_compat_20261123')
    manifest=R/'configs/protocols/kanana_own_original_RegStage_BF16_parameterized_graphs_v2_graphtext_20261123.json'
    source={str(manifest):sha(manifest),str(Path(mod.__file__)):sha(mod.__file__)}
    for stage in json.loads(manifest.read_bytes())['stages']:
        p=R/stage['graph_relative_path'];assert sha(p)==stage['graph_SHA256'];source[str(p)]=sha(p)
    def tensor(raw,shape):return torch.frombuffer(bytearray(base64.b64decode(raw,validate=True)),dtype=torch.bfloat16).reshape(shape)
    def bits(t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    def stats(actual,reference):
        delta=actual.double()-reference.double();rmse=float(delta.square().mean().sqrt());denom=float(reference.double().square().mean().sqrt())
        return dict(all_raw_bits_equal=bits(actual)==bits(reference),max_abs=float(delta.abs().max()),RMSE=rmse,relative_RMSE=rmse/denom if denom else None)
    reviews=[];source_checks=[]
    for stage in controls['stages']:
        cpu=mod.RegStage(2,stage['input_channels'],1024,stride=1,dilation=1,act_layer=torch.nn.SiLU,norm_layer=mod.LayerNorm2d).eval()
        with torch.no_grad():
            for p in stage['parameter_inputs']:
                t=tensor(p['complete_synthetic_parameter_raw_base64'],p['shape'])
                assert hashlib.sha256(bits(t)).hexdigest()==p['synthetic_control_SHA256']
                cpu.get_parameter(p['relative_key']).copy_(t)
        npu=mod.RegStage(2,stage['input_channels'],1024,stride=1,dilation=1,act_layer=torch.nn.SiLU,norm_layer=mod.LayerNorm2d).eval().to('npu:4')
        with torch.no_grad():
            for p in stage['parameter_inputs']:
                npu.get_parameter(p['relative_key']).copy_(cpu.get_parameter(p['relative_key']).to('npu:4'))
                t=npu.get_parameter(p['relative_key'])
                assert t.device==torch.device('npu:4') and t.dtype==torch.bfloat16
                actual_sha=hashlib.sha256(bits(t)).hexdigest();assert actual_sha==p['synthetic_control_SHA256']
                source_checks.append(dict(name=stage['name']+'.'+p['relative_key'],shape=list(t.shape),dtype=str(t.dtype),actual_NPU_SHA256=actual_sha,synthetic_only=True))
        for case,c in enumerate(stage['CPU_synthetic_controls']):
            x=tensor(c['complete_synthetic_input_raw_base64'],c['input_shape']);aa=tensor(c['complete_synthetic_output_raw_base64'],c['output_shape'])
            assert hashlib.sha256(bits(x)).hexdigest()==c['input_SHA256']
            assert hashlib.sha256(bits(aa)).hexdigest()==c['native_SHA256']==c['graph_SHA256']
            put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='NPU4_untrained_original_RegStage_operator_interface_only',stage=stage['name'],case=case,completed=len(reviews),no_model_NN_or_fullNF=True))
            with torch.inference_mode():
                ref=cpu(x);xn=x.to('npu:4');yn=npu(xn);torch.npu.synchronize();actual=yn.cpu()
            assert yn.dtype==torch.bfloat16 and yn.device==torch.device('npu:4') and list(yn.shape)==c['output_shape']
            assert torch.isfinite(actual).all() and torch.isfinite(ref).all() and torch.isfinite(aa).all()
            f=H/('all_untrained_values_stage'+str(stage['input_channels'])+'_case'+str(case)+'.npz')
            np.savez_compressed(f,full_synthetic_input_BF16_bits=x.view(torch.uint16).numpy(),full_original_AA_CPU_BF16_bits=aa.view(torch.uint16).numpy(),
                full_same_R_original_graph_CPU_BF16_bits=ref.view(torch.uint16).numpy(),full_actual_NPU_BF16_bits=actual.view(torch.uint16).numpy())
            reviews.append(dict(stage=stage['name'],input_shape=list(x.shape),output_shape=list(actual.shape),NPU_vs_original_AA_CPU=stats(actual,aa),
                NPU_vs_same_R_original_graph_CPU=stats(actual,ref),full_value_file=str(f),full_value_SHA256=sha(f),not_model_acceptance=True))
        del npu,cpu
        torch.npu.empty_cache()
    assert len(source_checks)==58 and len(reviews)==6
    assert before=={p:sha(p) for p in before}
    result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),diagnostic_interface_execution_only=True,
        all6_original_graph_NPU_executions_completed=True,all58_untrained_parameter_source_actual_NPU_SHA_verified=True,
        source_SHA256=source,synthetic_parameter_sources=source_checks,controls=reviews,
        original_full_AA_control_plan=str(PLAN),original_full_AA_control_plan_SHA256=sha(PLAN),
        checkpoint_values_loaded=0,learned_CPU_GPU_NPU_forwards=0,actual_NEW_model_forwards=0,
        formal_NEW_NPU3=False,formal_fullNF_or_200_or_qualification=False,no_model_adapter_or_dispatch_approved=True,
        original_cross_version_strict_raw_bit_control_not_reclassified=True,no_numeric_tolerance_lowered_or_redefined=True,
        all_discrepancies_and_full_untrained_values_preserved=True,all_canonical_unchanged=True)
    put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,diagnostic_only=True,no_model_NN_or_fullNF=True))
    print(json.dumps(dict(passed=True,UTC=result['UTC'],home=str(H),review_SHA256=sha(H/'review.json'),controls=reviews,
        all58_synthetic_sources_actual_NPU_SHA_verified=True,not_formal_model_acceptance=True),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,no_model_NN_or_fullNF=True))
        raise

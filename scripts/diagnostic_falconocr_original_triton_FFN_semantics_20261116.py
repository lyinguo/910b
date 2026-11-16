"""Check original own Triton BF16 gate semantics on real NPU, no model grade."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,traceback,inspect
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')];D=R/'workspace/diagnostics/batch_910b';H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/FalconOCR_original_own_Triton_FFN_intrinsic_precision_diagnostic_20261116'
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch,torch_npu,triton
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 from universal_eval.model.falconocr_own_original_20261116 import modeling_falcon_ocr as original
 from universal_eval.model.falconocr_own_npu_20261116.npu_attention_backend import falcon_squared_relu_gate_npu
 assert original._MLU_MODE is False and not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');torch.npu.set_device(4);gen=torch.Generator().manual_seed(220345);rows=[]
  for N in [1,3,17]:
   packed=torch.randn(N,4608,generator=gen).to(torch.bfloat16).to('npu:4')
   with torch.inference_mode():actual=original.squared_relu_gate(packed,2304);proposed=falcon_squared_relu_gate_npu(packed);bf16=(torch.relu(packed[...,0::2])**2)*packed[...,1::2]
   torch.npu.synchronize();a=actual.cpu();p=proposed.cpu();b=bf16.cpu();rows.append(dict(rows=N,source_kernel_actual_NPU=True,current_F32_internal_single_cast_equal=torch.equal(a,p),eager_BF16_intermediate_equal=torch.equal(a,b),different_values_current=int((a!=p).sum()),different_values_BF16=int((a!=b).sum()),current_max_abs=float((a.float()-p.float()).abs().max()),BF16_max_abs=float((a.float()-b.float()).abs().max())))
  src=Path(inspect.getfile(original));r=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,no_model_or_CPU_score=True,no_source_or_checkpoint_edit=True,own_source_path=str(src),own_source_SHA256=hashlib.sha256(src.read_bytes()).hexdigest(),Triton_version=triton.__version__,cases=rows,original_own_kernel_current_F32_port_equivalent=all(r['current_F32_internal_single_cast_equal'] for r in rows));put('review.json',r);put('actual_completion.json',dict(UTC=r['UTC'],actual_returncode=0));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc()));raise

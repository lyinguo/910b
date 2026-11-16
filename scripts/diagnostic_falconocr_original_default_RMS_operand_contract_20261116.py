"""Own completed operands: default RMS epsilon semantics, no new model score."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')];D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'FalconOCR_existing_own_operands_original_default_RMS_epsilon_diagnostic_20261116'
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch,torch_npu
 from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');torch.set_num_threads(4);torch.npu.set_device(4)
  archive=R/'workspace/diagnostics/model_runs/FalconOCR_own_BF16_native512_original_Triton_v2_20261116_125953_611785_1379184/sample_00000.npz'
  with ReadOnlyNPZ(archive) as a:
   x=torch.from_numpy(a['merged_embeddings'].copy()).view(torch.bfloat16);recorded=torch.from_numpy(a['QKV_input_0_step_0'].copy()).view(torch.bfloat16);qkv=torch.from_numpy(a['QKV_0_step_0'].copy()).view(torch.bfloat16)
  rows=[]
  for name,t in [('own_layer0_merged_embeddings',x),('own_layer0_q_before_head_RMS',qkv[...,:1024].reshape(1,-1,16,64))]:
   dim=t.shape[-1];v=t.to('npu:4');ep=torch.finfo(torch.bfloat16).eps;ef=torch.finfo(torch.float32).eps
   with torch.inference_mode():
    default=torch.nn.functional.rms_norm(v,(dim,)).cpu();bf=torch.nn.functional.rms_norm(v,(dim,),eps=ep).cpu();f=torch.nn.functional.rms_norm(v,(dim,),eps=ef).cpu();cpu=torch.nn.functional.rms_norm(t,(dim,));math=(t.float()*torch.rsqrt(t.float().square().mean(-1,keepdim=True)+ep)).to(t.dtype)
   row=dict(operand=name,shape=list(t.shape),documented_input_BF16_default_epsilon=ep,F32_epsilon=ef,NPU_default_equal_explicit_BF16_epsilon=torch.equal(default,bf),NPU_default_equal_explicit_F32_epsilon=torch.equal(default,f),CPU_default_equal_F32_math_with_BF16_epsilon=torch.equal(cpu,math),NPU_default_vs_CPU_max_abs=float((default.float()-cpu.float()).abs().max()),NPU_default_vs_CPU_unequal_values=int((default!=cpu).sum()),NPU_default_vs_explicit_BF16_max_abs=float((default.float()-bf.float()).abs().max()),NPU_explicit_BF16_vs_CPU_max_abs=float((bf.float()-cpu.float()).abs().max()))
   if name=='own_layer0_merged_embeddings':row['same_operand_default_NPU_equals_original_recorded_QKV_input']=torch.equal(default,recorded)
   rows.append(row)
  r=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,no_new_model_forward_or_grade=True,archive=str(archive),archive_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest(),checks=rows,default_epsilon_semantic_mismatch_proven=all(v['NPU_default_equal_explicit_F32_epsilon'] and not v['NPU_default_equal_explicit_BF16_epsilon'] for v in rows));put('review.json',r);put('actual_completion.json',dict(UTC=r['UTC'],actual_returncode=0));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc()));raise

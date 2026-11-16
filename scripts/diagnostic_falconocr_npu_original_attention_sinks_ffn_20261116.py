"""Real NPU primitive controls, not a model score or formal small-sample run."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,fcntl,os,math,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'FalconOCR_real_NPU_attention_predicate_LSE_sinks_FFN_primitive_controls_20261116'
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch,torch_npu
 from safetensors import safe_open
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
 from universal_eval.model.falconocr_own_npu_20261116.npu_attention_backend import CANNPredicateMask,falcon_attention_npu,falcon_squared_relu_gate_npu
 from universal_eval.model.falconocr_own_original_20261116.rope import apply_golden_freqs_cis_to_visual_pos,apply_3d_rotary_emb
 assert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 with (R/'workspace/diagnostics/batch_910b/runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),R/'workspace/diagnostics/batch_910b/runner.lock');core_unchanged()
  torch.set_num_threads(4);torch.npu.set_device(4);device=torch.device('npu:4')
  for n in ['matmul','conv']:getattr(torch.npu,n).allow_hf32=False
  gen=torch.Generator().manual_seed(124000);checks=[]
  for Q,K,offset in [(5,5,0),(1,17,16),(3,19,16)]:
   q=torch.randn(1,16,Q,64,generator=gen).to(torch.bfloat16);k=torch.randn(1,16,K,64,generator=gen).to(torch.bfloat16);v=torch.randn(1,16,K,64,generator=gen).to(torch.bfloat16)
   def predicate(b,h,qi,ki):return qi+offset>=ki
   mask=CANNPredicateMask(predicate,1,Q,K,device)
   with torch.inference_mode():out,lse=falcon_attention_npu(q.to(device),k.to(device),v.to(device),mask)
   allow=torch.arange(Q).view(-1,1)+offset>=torch.arange(K).view(1,-1)
   scores=(q.float()@k.float().transpose(-2,-1))/math.sqrt(64);scores=scores.masked_fill(~allow,float('-inf'));ref_lse=torch.logsumexp(scores,-1);ref=(scores.softmax(-1)@v.float()).to(torch.bfloat16)
   err=(out.cpu().float()-ref.float()).abs();le=(lse.cpu()-ref_lse).abs();assert float(err.max())<=.03125 and float(le.max())<=.001
   sinks=torch.randn(16,generator=gen);got=(out*torch.sigmoid(lse-sinks.to(device).view(1,-1,1)).unsqueeze(-1)).to(out.dtype).cpu();wanted=(ref*torch.sigmoid(ref_lse-sinks.view(1,-1,1)).unsqueeze(-1)).to(ref.dtype);se=(got.float()-wanted.float()).abs();assert float(se.max())<=.03125
   checks.append(dict(Q=Q,K=K,offset=offset,output_max_abs=float(err.max()),LSE_max_abs=float(le.max()),original_separate_sink_scale_max_abs=float(se.max()),actual_NPU=True))
  packed=torch.randn(3,2304*2,generator=gen).to(torch.bfloat16);ref=(torch.relu(packed[...,0::2].float()).square()*packed[...,1::2].float()).to(torch.bfloat16);actual=falcon_squared_relu_gate_npu(packed.to(device)).cpu();assert torch.equal(actual,ref)
  W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/FalconOCRForCausalLM/tiiuae/Falcon-OCR')
  with safe_open(str(W/'model.safetensors'),framework='pt',device='cpu') as f:freq=f.get_tensor('freqs_cis_golden')
  pos=torch.tensor([[[float('nan'),float('nan')],[0.,0.],[1.,2.],[2.,3.]]]);cpu=apply_golden_freqs_cis_to_visual_pos(freq,pos);npu=apply_golden_freqs_cis_to_visual_pos(freq.to(device),pos.to(device)).cpu();re=(cpu-npu).abs();assert float(re.max())<=.00001
  q=torch.randn(1,4,16,64,generator=gen).to(torch.bfloat16);k=q.clone();from universal_eval.model.falconocr_own_original_20261116.rope import precompute_freqs_cis
  f=precompute_freqs_cis(32,4).unsqueeze(0);ca=apply_3d_rotary_emb(q,k,f,cpu,pos);na=apply_3d_rotary_emb(q.to(device),k.to(device),f.to(device),npu.to(device),pos.to(device));roe=max(float((x.float()-y.cpu().float()).abs().max()) for x,y in zip(ca,na));assert roe<=.03125
  r=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),NPU_card=4,attention=checks,original_Triton_single_cast_FFN_exact=True,own_source_F32_golden_position_max_abs=float(re.max()),own_real_3D_RoPE_max_abs=roe,no_model_forward=True,no_formal_true3_or_score_or_G_claim=True,CPU_reference_only_diagnostic=True)
  put('review.json',r);put('actual_completion.json',dict(UTC=r['UTC'],actual_returncode=0));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc()));raise

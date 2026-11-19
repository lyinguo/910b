"""Tiny original decode attention interface on card4; no checkpoint or model forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,importlib,importlib.util,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Moondream3_own_original_flex_decode_operator_diagnostic_no_model_real_FLOCK_0610_20261119'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,v):
 p=H/name;p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 H.mkdir(exist_ok=False);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());before={p:sha(p) for p in ptr['canonical_SHA256']};assert before==ptr['canonical_SHA256']
 guard=(D/'runner.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert [v['PID'] for v in runner_lock_owners()]==[os.getpid()]
 pre=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert pre.returncode==0 and 'No running processes found in NPU 4' in pre.stdout
 put('actual_admission.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),argv=sys.argv,kernel_FLOCK=runner_lock_owners(),NPU=4,npu_smi_before=pre.stdout,perceptron_package_available=importlib.util.find_spec('perceptron') is not None,no_model_or_checkpoint_loaded=True))
 import torch,torch_npu,numpy as np
 from torch.nn.attention.flex_attention import create_block_mask,flex_attention
 module=importlib.import_module('universal_eval.model.moondream3_own_original_source_20261119.moondream')
 torch.npu.set_device('npu:4');torch.set_num_threads(4);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
 # Target-device placement is the only proposed port under test. Original
 # block dimensions, causal function, slicing, and absolute position stay exact.
 pos=torch.tensor([730],device='npu:4',dtype=torch.long)
 mask=create_block_mask(module.causal_mask,B=None,H=None,Q_LEN=4096,KV_LEN=4096,device='npu:4')
 block_index=pos//mask.BLOCK_SIZE[0];sliced=mask[:,:,block_index];sliced.seq_lengths=(1,sliced.seq_lengths[1]);sliced.mask_mod=module.get_mask_mod(mask.mask_mod,pos[0])
 q=(torch.arange(32*64,dtype=torch.float32).reshape(1,32,1,64)%31-15).div(16).to(torch.bfloat16).to('npu:4')
 k=(torch.arange(32*4096*64,dtype=torch.float32).reshape(1,32,4096,64)%29-14).div(16).to(torch.bfloat16).to('npu:4');v=k.flip([-1])
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='tiny_original_flex_decode_operator_no_model_NN',NPU=4,absolute_position=730,original_causal_mask_and_4096_cache_shape_preserved=True))
 with torch.inference_mode():out=flex_attention(q,k,v,block_mask=sliced)
 torch.npu.synchronize();actual=out.cpu();assert torch.isfinite(actual).all()
 # Independent dense mask is a diagnostic reference, never a replacement model mask.
 dense=(torch.arange(4096)<=730).reshape(1,1,1,4096)
 reference=torch.nn.functional.scaled_dot_product_attention(q.cpu().float(),k.cpu().float(),v.cpu().float(),attn_mask=dense).to(torch.bfloat16)
 arr=lambda x:x.contiguous().view(torch.uint16).numpy()
 np.savez_compressed(H/'all_unlearned_operator_inputs_outputs_and_CPU_reference.npz',query_BF16_bits=arr(q.cpu()),key_BF16_bits=arr(k.cpu()),value_BF16_bits=arr(v.cpu()),actual_NPU_BF16_bits=arr(actual),CPU_dense_reference_BF16_bits=arr(reference),absolute_causal_mask=dense.numpy())
 assert before=={p:sha(p) for p in before}
 review=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,scope='Unlearned original FlexAttention decode operator only; zero model forwards, not numerical model acceptance or a grade',NPU=4,
  actual_NPU_dtype='BF16',actual_output_shape=list(actual.shape),max_abs_dense_CPU_reference=float((actual.float()-reference.float()).abs().max()),all_canonical_unchanged=True,
  no_model_math_or_precision_or_budget_or_mask_or_cache_change=True,checkpoint_tensors_loaded=0,actual_NEW_model_forwards=0)
 put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,diagnostic_only=True,no_model_NN=True));print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,no_model_NN=True,no_G_write=True))
  raise

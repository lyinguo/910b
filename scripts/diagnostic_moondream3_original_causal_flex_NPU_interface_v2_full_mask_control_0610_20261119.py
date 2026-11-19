"""Actual NPU kernel versus original CPU FlexAttention, unlearned operands only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,importlib,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Moondream3_original_causal_flex_NPU_interface_v2_full_mask_operator_control_0610_20261119'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,v):(H/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 H.mkdir(exist_ok=False);os.chdir(R);sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p.get('role') in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 ptr=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());before={p:sha(p) for p in ptr['canonical_SHA256']};assert before==ptr['canonical_SHA256']
 guard=(D/'runner.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert [p['PID'] for p in runner_lock_owners()]==[os.getpid()]
 check=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert check.returncode==0 and 'No running processes found in NPU 4' in check.stdout
 put('actual_admission.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),actual_cwd=str(Path.cwd()),actual_argv=(Path('/proc')/str(os.getpid())/'cmdline').read_bytes().replace(b'\0',b' ').decode(),actual_kernel_FLOCK=runner_lock_owners(),NPU=4,npu_smi_before=check.stdout,checkpoint_tensors_loaded=0))
 import torch,torch_npu,numpy as np
 from torch.nn.attention.flex_attention import create_block_mask,flex_attention
 module=importlib.import_module('universal_eval.model.moondream3_own_original_source_20261119.moondream')
 interface=importlib.import_module('universal_eval.model.adapters.moondream3_own_original_causal_flex_NPU_interface_v1_0610_20261119')
 torch.set_num_threads(4);torch.npu.set_device('npu:4');torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
 full_npu=create_block_mask(module.causal_mask,B=None,H=None,Q_LEN=4096,KV_LEN=4096,device='npu:4')
 full_cpu=create_block_mask(module.causal_mask,B=None,H=None,Q_LEN=4096,KV_LEN=4096,device='cpu')
 gen=torch.Generator(device='cpu').manual_seed(1937);reviews=[]
 for case,pos_value in enumerate([730,731,1023,2047,4095]):
  pos=torch.tensor([pos_value],dtype=torch.long,device='npu:4');position_cpu=torch.tensor([pos_value],dtype=torch.long)
  mask=full_npu[:,:,pos//full_npu.BLOCK_SIZE[0]];mask.seq_lengths=(1,mask.seq_lengths[1]);mask.mask_mod=module.get_mask_mod(full_npu.mask_mod,pos[0])
  reference_mask=full_cpu[:,:,position_cpu//full_cpu.BLOCK_SIZE[0]];reference_mask.seq_lengths=(1,reference_mask.seq_lengths[1]);reference_mask.mask_mod=module.get_mask_mod(full_cpu.mask_mod,position_cpu[0])
  q=torch.randn((1,32,1,64),generator=gen).to(torch.bfloat16);k=torch.randn((1,32,4096,64),generator=gen).to(torch.bfloat16);v=torch.randn((1,32,4096,64),generator=gen).to(torch.bfloat16)
  put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='unlearned_original_CPU_Flex_and_actual_NPU_same_mask_operator_control',case=case,position=pos_value,no_model_forward=True))
  with torch.inference_mode():
   reference=flex_attention(q,k,v,block_mask=reference_mask)
   qn,kn,vn=[t.to('npu:4') for t in [q,k,v]];actual=interface.original_causal_flex_attention(qn,kn,vn,block_mask=mask);dense=interface.original_dense_mask(qn,kn,vn,mask)
  torch.npu.synchronize();actual=actual.cpu();dense=dense.cpu();idx=torch.arange(4096).reshape(1,1,1,4096);independent_mask=(idx<=pos_value).expand(1,32,1,4096)
  assert torch.equal(dense,independent_mask) and torch.isfinite(actual).all() and torch.isfinite(reference).all()
  delta=actual.float()-reference.float();rmse=float(delta.square().mean().sqrt());maximum=float(delta.abs().max());
  bits=lambda x:x.contiguous().view(torch.uint16).numpy()
  np.savez_compressed(H/f'all_unlearned_case_{case:02d}.npz',query_BF16_bits=bits(q),key_BF16_bits=bits(k),value_BF16_bits=bits(v),actual_NPU_BF16_bits=bits(actual),original_CPU_Flex_BF16_bits=bits(reference),actual_original_mask=dense.numpy(),independent_absolute_mask=independent_mask.numpy())
  reviews.append(dict(case=case,position=pos_value,max_abs=maximum,RMSE=rmse,all_131072_mask_values_equal_original_absolute_causal_predicate=True,all_input_and_output_values_preserved=True))
  assert maximum<=.03125 and rmse<=.004,'Fixed operator-diagnostic tolerance only, never a model acceptance threshold'
 assert before=={p:sha(p) for p in before}
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,diagnostic_only=True,actual_NEW_model_forwards=0,checkpoint_tensors_loaded=0,
  original_source_file_unmodified=True,shared_Torch_Flex_not_patched=True,NPU=4,actual_dtype='BF16',physical_KV_length=4096,all_original_causal_mask_values_preserved=True,cases=reviews,
  original_CPU_Flex_and_actual_NPU_SDPA_compared=True,not_a_model_NF_or_score_or_original_protocol_qualification=True,model_loading_precision_and_full_NPU_observer_NF_YAML_still_required=True,all_canonical_unchanged=True)
 put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,diagnostic_only=True,no_model_NN=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.is_dir() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,no_model_NN=True,no_G_write=True))
  raise

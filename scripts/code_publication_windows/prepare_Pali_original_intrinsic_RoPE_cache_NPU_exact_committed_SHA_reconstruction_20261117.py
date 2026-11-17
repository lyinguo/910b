import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
s='''"""Reconstruct only the own native intrinsic RoPE table and bind its old SHA.
No learned model, score, changed parameter, alternate checkpoint or output.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Pali10Pt448_original_intrinsic_RoPE_NPU_table_exact_old_committed_SHA_reconstruction_diagnostic_20261117';sys.path[:0]=[str(R),str(R/'scripts')]
os.environ.update(ASCEND_RT_VISIBLE_DEVICES='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):(H/n).write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','independent_verifier']]
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  import torch,torch_npu,numpy as np
  torch.set_num_threads(4);assert torch.npu.device_count()==1;torch.npu.set_device(0);start=time.monotonic()
  protected=[D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];before={str(p):sha(p) for p in protected}
  source=Path('/vllm-workspace/vllm/vllm/model_executor/layers/rotary_embedding/base.py');assert sha(source)=='c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5'
  index=R/'workspace/diagnostics/model_runs/pali10_pt448_native_vllm_a5ix9omo/full_numeric_group_index.json';idx=json.loads(index.read_bytes());group=Path(idx['groups'][0]['path']);g=json.loads(group.read_bytes());expected={json.dumps(row['native_cos_sin_cache'],sort_keys=True) for f in g['frames'] for row in f['rotary']};assert len(expected)==1;expected=json.loads(next(iter(expected)));assert expected==dict(shape=[8192,256],dtype='torch.bfloat16',SHA256='38b3b2b0fd8653fd69f783079f14897e9213f2ede95a8de1d08412772bb721e6')
  # Exact own native base.py recipe, explicit original F32 arange/inverse
  # frequency and einsum/cos/sin followed by the recorded BF16 table cast.
  # Device follows the real native model construction, physical4 isolated.
  with torch.device('npu:0'):
   inv_freq=1.0/10000.0**(torch.arange(0,256,2,dtype=torch.float32)/256)
   t=torch.arange(8192,dtype=torch.float32)
   freqs=torch.einsum('i,j -> ij',t,inv_freq)
   cache=torch.cat((freqs.cos(),freqs.sin()),dim=-1).to(torch.bfloat16)
  torch.npu.synchronize();value=cache.cpu().contiguous();raw=value.view(torch.uint8).numpy().tobytes();actual=hashlib.sha256(raw).hexdigest();file=H/'NPU_source_reconstructed_original_intrinsic_cos_sin_BF16_uint16.npy';np.save(file,value.view(torch.uint16).numpy(),allow_pickle=False)
  assert before=={str(p):sha(p) for p in protected}
  result=dict(UTC=datetime.now(timezone.utc).isoformat(),exact_old_recorded_table_SHA_match=actual==expected['SHA256'],recorded_real_NPU_operand=expected,reconstructed_operand=dict(shape=list(value.shape),dtype=str(value.dtype),SHA256=actual),values_file=str(file),values_file_SHA256=sha(file),source_path=str(source),source_SHA256=sha(source),old_group_manifest=str(group),old_group_manifest_SHA256=sha(group),all42_layers_all10_forwards_old_cache_commit_identical=True,physical_NPU=4,no_learned_model_or_model_score=True,no_original_report_or_failed_job_rewrite=True,old_original_full_cache_values_not_claimed_captured_then=True,only_exact_SHA_identity_can_bind_reconstructed_values_as_old_operand=True,all_canonical_SHA_unchanged=True,canonical_SHA256=before,elapsed_seconds=time.monotonic()-start)
  put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0 if result['exact_old_recorded_table_SHA_match'] else 1,operator_diagnostic_only_not_formal_NPU_evaluation=True));print(json.dumps(result,ensure_ascii=False));assert result['exact_old_recorded_table_SHA_match'],'No SHA identity: keep as diagnostic only, do not replace original operand or widen any threshold'
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:
  if not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),operator_diagnostic_only_not_formal_NPU_evaluation=True))
  raise
'''
ast.parse(s);name='scripts/diagnostic_Pali_original_intrinsic_RoPE_NPU_cache_exact_committed_SHA_20261117.py';(P/Path(name).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';name=__NAME__;raw=base64.b64decode(__DATA__);assert hashlib.sha256(raw).hexdigest()==__SHA__;ast.parse(raw);p=R/name
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
I=B/'Pali10Pt448_intrinsic_RoPE_cache_exact_SHA_NPU_diagnostic_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',name],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',ASCEND_RT_VISIBLE_DEVICES='4'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=name,source_SHA256=__SHA__,unweighted_operator_table_reconstruction_only_not_model_NPU3_or200=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__NAME__',repr(name)).replace('__DATA__',repr(base64.b64encode(s.encode()).decode())).replace('__SHA__',repr(h));ast.parse(remote);(P/'deploy_launch_Pali_original_intrinsic_RoPE_NPU_cache_exact_SHA_diagnostic_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(source=name,SHA256=h)))

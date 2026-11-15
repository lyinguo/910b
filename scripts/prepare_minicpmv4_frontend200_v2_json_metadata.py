"""Recover only the unsaved own input stage; preserve the original failed JSON."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'MiniCPMV4_own_native_frontend200_v2_tensor_metadata_recovery_user_continue_20261115';OLD=B/'MiniCPMV4_own_native_frontend200_source_meta_preflight_user_continue_20261115';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MiniCPMV/openbmb/MiniCPM-V-4')
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');assert not p.exists()
 t.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,p)
def main():
 H.mkdir(exist_ok=False)
 import torch
 from universal_eval.model.adapters.minicpmv4_own_frontend_v1 import OwnMiniCPMV4Inputs,tensor_sha,flatten_tensors
 from universal_eval.model.adapters.minicpmv4_original_pixel_check_v1 import check_pixels
 from universal_eval.algorithm.inference import GreedyInference
 torch.set_num_threads(4);front=[];inputs=OwnMiniCPMV4Inputs(W);dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=[json.loads(s) for s in dataset.read_text().splitlines()];assert len(rows)==4329
 for i,r in enumerate(rows[:200]):
  path=Path(r['media_path']);path=path if path.is_absolute() else dataset.parent/path;path=str(path.resolve());prompt=GreedyInference._add_answer_instruction(r['prompt']);rendered,v=inputs.prepare(prompt,path);tensors=flatten_tensors(v)
  check_pixels(inputs.image_processor,path,v['pixel_values'][0])
  def plain(x):
   if torch.is_tensor(x):return x.tolist()
   if isinstance(x,(list,tuple)):return [plain(t) for t in x]
   return x
  front.append(dict(index=i,source_prompt=r['prompt'],prompt=r['prompt'],actual_model_prompt=prompt,media_path=path,media_file_SHA256=sha(path),reference=r['reference'],text=rendered,tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in tensors.items()},image_sizes=plain(v['image_sizes']),image_bound=plain(v['image_bound']),input_ids=v['input_ids'][0].tolist(),patches=len(v['pixel_values'][0]),independent_uint8_normalization_and_patch_layout_exact=True))
 put('own_frontend200.json',front)
 source=json.loads((OLD/'own_source_tensor_manifest.json').read_bytes());files=json.loads((OLD/'own_weight_file_SHA256.json').read_bytes())
 out=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id='openbmb/MiniCPM-V-4',own_source_tensor_count=len(source),own_weight_files=files,own_source_manifest=str(OLD/'own_source_tensor_manifest.json'),own_source_manifest_SHA256=sha(OLD/'own_source_tensor_manifest.json'),own_frontend200=str(H/'own_frontend200.json'),own_frontend200_SHA256=sha(H/'own_frontend200.json'),max_input_length=max(len(x['input_ids']) for x in front),max_patches=max(x['patches'] for x in front),original_source_copy_review=str(OLD/'own_source_copy_review.json'),original_source_copy_review_SHA256=sha(OLD/'own_source_copy_review.json'),all200_original_Greedy_input_and_independent_pixel_norm_layout_checked=True,old_serialization_failure_preserved=True,no_repeated_learned_CPU_or_NPU_inference=True)
 put('readiness.json',out);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc()))
  raise

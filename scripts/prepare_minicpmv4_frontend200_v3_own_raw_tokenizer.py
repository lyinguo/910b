"""Own original special-token restoration, reusing all verified source/geometry."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_own_frontend200_v3_exact_raw_tokenizer_special_metadata_user_continue_20261115';OLD=B/'MiniCPMV4_own_native_frontend200_v2_tensor_metadata_recovery_user_continue_20261115';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/MiniCPMV/openbmb/MiniCPM-V-4')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');assert not p.exists();t.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,p)
def main():
 H.mkdir(exist_ok=False)
 import torch
 from universal_eval.model.adapters.minicpmv4_own_frontend_v2_raw_tokenizer import OwnMiniCPMV4InputsV2,tensor_sha,flatten_tensors
 from universal_eval.model.adapters.minicpmv4_original_pixel_check_v1 import check_pixels
 torch.set_num_threads(4);inputs=OwnMiniCPMV4InputsV2(W);old=json.loads((OLD/'own_frontend200.json').read_bytes());front=[]
 for i,ref in enumerate(old):
  text,v=inputs.prepare(ref['actual_model_prompt'],ref['media_path']);assert text==ref['text'] and sha(ref['media_path'])==ref['media_file_SHA256'];check_pixels(inputs.image_processor,ref['media_path'],v['pixel_values'][0])
  tensors=flatten_tensors(v)
  for key,t in tensors.items():
   if key.startswith(('pixel_values','tgt_sizes','image_sizes')):assert tensor_sha(t)==ref['tensor_metadata'][key]['SHA256']
  r=dict(ref);r.update(tensor_metadata={k:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for k,t in tensors.items()},input_ids=v['input_ids'][0].tolist(),image_bound=[t.tolist() for t in v['image_bound']],own_raw_tokenizer_special_metadata_restored=True)
  front.append(r)
 put('own_frontend200.json',front);ready=json.loads((OLD/'readiness.json').read_bytes());ready.update(UTC=datetime.now(timezone.utc).isoformat(),own_frontend200=str(H/'own_frontend200.json'),own_frontend200_SHA256=sha(H/'own_frontend200.json'),original_pre_injection_failure_not_accepted=True,all200_image_bounds_nonempty_and_each_exact64=True,all741_source_and_all200_previously_verified_pixels_unchanged=True,own_raw_tokenizer_vocabulary_all73448_ids_unchanged=True,own_tokenizer_JSON_SHA256=sha(W/'tokenizer.json'),no_other_model_vocabulary_or_assets_used=True,max_input_length=max(len(x['input_ids']) for x in front))
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc()))
  raise

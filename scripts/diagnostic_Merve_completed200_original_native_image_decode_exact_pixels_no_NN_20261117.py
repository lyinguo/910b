import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
import numpy as np
import torch
from PIL import Image
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';F=R/'workspace/reports/merve_paligemma_vqav2/image-text-to-text/20261117_125717';H=B/'Merve_completed200_native_media_decode_pixel_reference_diagnostic_only_20261117';torch.set_num_threads(4)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 assert not H.exists();H.mkdir();before={str(p):sha(p) for p in F.iterdir()};p=js(js(F/'eval_report.json')['pipeline_config']['model']['kwargs']['protocol_file']);line=js(F/'operator_lineage.json')['actual_execution'];idx=js(line['full_numeric_group_index']);actual=[];source={}
 for item in idx['groups']:
  assert sha(item['path'])==item['SHA256'];g=js(item['path']);assert sha(g['archive'])==g['archive_SHA256']
  for n in ['/vllm-workspace/vllm/vllm/multimodal/image.py','/vllm-workspace/vllm/vllm/multimodal/media/image.py']:
   assert sha(n)==g['all_actual_worker_imported_sources_and_mapped_shared_libraries_SHA256'][n];source[n]=sha(n)
  with np.load(g['archive'],allow_pickle=False) as z:
   for v in g['vision']:
    ref=v['pixels'];t=torch.from_numpy(z[ref['array']].copy());t=t.view(torch.bfloat16) if ref['dtype']=='torch.bfloat16' else t;assert hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()==ref['SHA256'];actual.extend(t.unbind(0))
 assert len(actual)==200;rows=[json.loads(s) for s in (F/'sample_inputs.jsonl').read_text().splitlines()];assert len(rows)==200
 from vllm.multimodal.media.image import ImageMediaIO
 from transformers import AutoImageProcessor
 io=ImageMediaIO();reviews=[]
 for fast in [False,True]:
  proc=AutoImageProcessor.from_pretrained(p['weight_path'],local_files_only=True,use_fast=fast);bad_direct=[];bad_native=[];different_modes=[]
  for i,row in enumerate(rows):
   with Image.open(row['media_path']) as image:
    mode=image.mode;exif_orientation=image.getexif().get(274);transparent='transparency' in image.info;direct=image.convert('RGB');a=proc(images=direct,return_tensors='pt')['pixel_values'][0].to(actual[i].dtype)
   native=io.load_file(Path(row['media_path'])).media;b=proc(images=native,return_tensors='pt')['pixel_values'][0].to(actual[i].dtype)
   if not torch.equal(a,actual[i]):bad_direct.append(dict(index=i,mode=mode,exif_orientation=exif_orientation,transparency=transparent,max_abs=float((a.float()-actual[i].float()).abs().max())))
   if not torch.equal(b,actual[i]):bad_native.append(dict(index=i,mode=mode,exif_orientation=exif_orientation,transparency=transparent,max_abs=float((b.float()-actual[i].float()).abs().max())))
   if not torch.equal(a,b):different_modes.append(dict(index=i,mode=mode,exif_orientation=exif_orientation,transparency=transparent))
  reviews.append(dict(use_fast=fast,processor_class=type(proc).__module__+'.'+type(proc).__name__,direct_PIL_RGB_mismatch=bad_direct,original_native_ImageMediaIO_mismatch=bad_native,direct_vs_original_native_differences=different_modes))
 assert all(sha(f)==h for f,h in before.items());result=dict(UTC=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,NEW_NPU_inference=False,no_learned_CPU_or_NPU_model_forward=True,no_model_input_image_math_or_runtime_modified=True,original_complete200_five_unchanged=True,original_native_input_decoder_source_SHA256=source,all_actual_200_input_pixels_compared_bitwise=True,reviews=reviews,source_SHA256=sha(__file__))
 (H/'review.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps(dict(result,review_SHA256=sha(H/'review.json')),ensure_ascii=False))
if __name__=='__main__':main()

"""Own Falcon OCR CPU resource/dtype/input evidence, not a learned CPU/NPU grade."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct,os,traceback,gc,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='tiiuae/Falcon-OCR';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/FalconOCRForCausalLM')/MID;H=B/'FalconOCR_own_original512_source_dtype_CPU_frontend200_v3_actual_original_dataset_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def json_set_members(value):
 if isinstance(value,(set,frozenset)):return sorted(value)
 raise TypeError(type(value).__name__)
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=json_set_members)+'\n',encoding='utf-8');os.replace(t,p)
def tensor(t):
 import torch
 t=t.detach().cpu().contiguous();return dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
def main():
 import torch,numpy as np
 from PIL import Image
 from safetensors import safe_open
 from transformers import AutoTokenizer
 from universal_eval.model.falconocr_own_original_20261116.configuration_falcon_ocr import FalconOCRConfig
 from universal_eval.model.falconocr_own_original_20261116.modeling_falcon_ocr import FalconOCRForCausalLM,CATEGORY_PROMPTS
 from universal_eval.model.falconocr_own_original_20261116.processing_falcon_ocr import process_batch
 torch.set_num_threads(4);cfg=FalconOCRConfig(**js(W/'config.json'));cfg._name_or_path=str(W)
 sf=W/'model.safetensors';whole=dict(bytes=sf.stat().st_size,SHA256=sha(sf))
 probe=Path('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/FalconOCR_own_original512_source_dtype_CPU_frontend200_v2_json_set_recording_20261116/own_source_and_saved_loading_dtype_diagnostic.json');assert sha(probe)=='13264f4fede48cc65134d36c1ed6ff7f2d5517ac36d884191a6dcc40b01fd4a2'
 prior=js(probe);assert prior['passed'] and prior['all_source_values_cast_exact'] and prior['no_learned_CPU_forward']
 headers=prior['source'];put('own_source_dtype_probe_reuse.json',dict(path=str(probe),SHA256=sha(probe),no_repeat_learned_CPU_loading=True,all_115_own_source_casts_verified=True))
 tokenizer=AutoTokenizer.from_pretrained(str(W),trust_remote_code=True,local_files_only=True);assert len(tokenizer)==cfg.vocab_size==65536
 data=R/'workspace/datasets/image-to-text-ocr/eval.jsonl';rows=[json.loads(s) for s in data.read_text().splitlines()];assert len(rows)==3000 and sha(data)=='49aa4d96f9c24c16d31f8cb092ddb0ee839f93a0507c8588675cdd5801b9efd5'
 instruction=CATEGORY_PROMPTS['plain'];image_prompt=f'<|image|>{instruction}\n<|OCR_PLAIN|>';output=H/'frontend200.jsonl';records=[]
 with output.open('x',encoding='utf-8') as out:
  for i,row in enumerate(rows[:200]):
   media=Path(row['media_path']);media=(data.parent/media).resolve() if not media.is_absolute() else media.resolve();assert media.is_relative_to(data.parent)
   image=Image.open(media).convert('RGB');got=process_batch(tokenizer,cfg,[(image,image_prompt)],max_length=4096,min_dimension=64,max_dimension=1024)
   tensors={k:tensor(v) for k,v in got.items() if torch.is_tensor(v)};ids=got['tokens'][0].tolist();L=len(ids);S=((L+512+127)//128)*128;assert S<=8192
   rec=dict(index=i,prompt=row.get('prompt',''),media_path=str(media),image_SHA256=sha(media),reference=row['reference'],original_ignored_dataset_prompt=True,actual_category='plain',actual_image_prompt=image_prompt,input_ids=ids,tensors=tensors,original_max_new_tokens=512,original_allocation_S=S,original_effective_decode_slot_limit=S-L,original_temperature=0.0,original_compile=False,original_seed=42,original_stop_ids=[cfg.eos_id,tokenizer.convert_tokens_to_ids('<|end_of_query|>')],non_tensor_input_fields={k:v for k,v in got.items() if not torch.is_tensor(v)})
   out.write(json.dumps(rec,ensure_ascii=False,default=str)+'\n');out.flush();records.append(rec);put('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=i+1,target=200,no_NPU_or_learned_CPU_forward=True))
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.py','.jinja']}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_weight_path=str(W),own_weight_files={'model.safetensors':whole},own_assets_SHA256=assets,source_tensor_count=len(headers),own_frontend200=str(output),own_frontend200_SHA256=sha(output),own_source_dtype_probe=str(probe),own_source_dtype_probe_SHA256=sha(probe),dataset=str(data),dataset_SHA256=sha(data),samples=200,max_input_length=max(len(r['input_ids']) for r in records),source_original_hardware_boundary='Own current raw files contain documented existing MLU-compatible real rotary/eager branches; not claimed as historical A100 source byte capture. Saved original HF generate(image, category=plain,512,temp0,compileFalse) and workbook D .0909 are the comparison protocol.',source_math_or_checkpoint_not_modified=True,no_NPU_adapter_or_true3_ready_claim=True,independent_pixel_and_NPU_backend_controls_still_pending=True,no_NPU_or_learned_CPU_forward=True,no_CPU_score_or_G_write=True)
 put('review.json',result);put('actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,no_CPU_or_NPU_grade=True));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_CPU_or_NPU_grade=True));raise

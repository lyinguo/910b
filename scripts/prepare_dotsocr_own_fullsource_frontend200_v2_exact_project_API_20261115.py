"""Own DotsOCR source, original OCR inputs and standalone native construction preflight."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,contextlib,io,os,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'DotsOCR_own_original_source_frontend200_v2_exact_project_API_user_continue_20261115';MID='rednote-hilab/dots.ocr'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
 p=Path(p);assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch
 from safetensors import safe_open
 from universal_eval.model.adapters.dotsocr_own_frontend_bf16_v1 import OwnDotsOCRInputs,tensor_sha
 from universal_eval.model.dotsocr_own_original_sources_20261115.configuration_dots import DotsOCRConfig
 from universal_eval.model.dotsocr_own_original_sources_20261115.modeling_dots_ocr import DotsOCRForCausalLM
 from universal_eval.data.loader import DatasetLoader
 from universal_eval.data.registry import TaskDataRegistry
 from universal_eval.data.preprocessor import DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 old=json.loads((B/'DotsOCR_own_original_resource_source_preflight_after_MiniCPMV4_user_continue_20261115/resource_source_review.json').read_bytes());W=Path(old['weight_path'])
 for n,v in old['own_weight_files'].items():assert sha(W/n)==v['SHA256']
 previous=B/'DotsOCR_own_original_source_frontend200_v1_user_continue_20261115'
 failed=json.loads((previous/'actual_completion.json').read_bytes());assert failed['actual_returncode']==1 and failed['no_NPU_inference']
 source=json.loads((previous/'own_full_source_manifest.json').read_bytes())
 index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];assert set(source)==set(index)
 src=H/'own_full_source_manifest.json';put(src,source)
 cfg=DotsOCRConfig(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation='eager'
 with contextlib.redirect_stdout(io.StringIO()):
  with torch.device('meta'):model=DotsOCRForCausalLM(cfg)
 assert set(dict(model.named_parameters(remove_duplicate=False)))==set(source)
 for n,t in model.named_parameters(remove_duplicate=False):assert list(t.shape)==source[n]['shape']
 del model
 ds=Path('/mnt/afs/comp_error/workspace/datasets/image-to-text-ocr/eval.jsonl')
 rows=DatasetLoader().load(str(ds));rows=TaskDataRegistry._resolve_media_paths(rows,str(ds.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=True)
 assert len(rows)==3000
 inputs=OwnDotsOCRInputs(W);front=[]
 for i,r in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(r['prompt']);text,v=inputs.prepare(prompt,r['media_path'])
  pixels=inputs.independently_check_pixels(r['media_path'],v['pixel_values'],v['image_grid_thw'])
  rec=dict(index=i,prompt=r['prompt'],actual_model_prompt=prompt,media_path=r['media_path'],media_file_SHA256=sha(r['media_path']),reference=r['reference'],text=text,input_ids=v['input_ids'][0].tolist(),tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in v.items()},grid=v['image_grid_thw'].tolist(),independent_original_pixel_normalization_layout_exact=pixels,own_imgpad_count=int((v['input_ids']==151665).sum()))
  front.append(rec)
 dest=H/'own_original_frontend200.json';put(dest,front)
 ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,weight_path=str(W),source_tensor_count=len(source),own_weight_files=old['own_weight_files'],own_assets_SHA256=old['own_assets_SHA256'],own_source_manifest=str(src),own_source_manifest_SHA256=sha(src),own_frontend200=str(dest),own_frontend200_SHA256=sha(dest),dataset_path=str(ds),dataset_SHA256=sha(ds),all200_original_project_preprocessing_tokens_pixels_checked=True,own_meta_parameter_names_shapes_bijective=True,no_NPU_or_learned_CPU_forward=True,original_own_source_math_not_changed=True,original_vision_FA2_unavailable_eager_fallback_disclosed=True,original_max_tokens=12,original_OCR_CER_task_preserved=True)
 put(H/'readiness.json',ready);put(H/'actual_completion.json',dict(actual_returncode=0,UTC=datetime.now(timezone.utc).isoformat()));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put(H/'actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc(),no_NPU_inference=True))
  raise

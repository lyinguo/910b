"""Prepare own MiniCPM-V-2_6 parameter SHA and exact original input evidence, no learned CPU."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,contextlib,io,traceback,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV26_own_original_fullsource_frontend200_v1_user_continue_20261115'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import torch
 from safetensors import safe_open
 from universal_eval.model.adapters.minicpmv26_own_frontend_bf16_v1 import OwnMiniCPMV26Inputs,tensor_sha,flatten_tensors
 from universal_eval.model.minicpmv26_own_original_20261115.configuration_minicpm import MiniCPMVConfig
 from universal_eval.model.minicpmv26_own_original_20261115.modeling_minicpmv import MiniCPMV
 from universal_eval.model.adapters.minicpmv4_original_pixel_check_v1 import check_pixels
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4);rr=json.loads((B/'MiniCPMV26_own_original_resources_source_preflight_v1_after_DotsOCR_user_continue_20261115/resource_source_review.json').read_bytes());assert rr['passed'];W=Path(rr['weight_path'])
 source={};idx=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map']
 for n,r in rr['own_weight_files'].items():assert sha(W/n)==r['SHA256']
 for file in sorted(set(idx.values())):
  with safe_open(str(W/file),framework='pt',device='cpu') as f:
   for n in f.keys():
    t=f.get_tensor(n);assert t.dtype==torch.bfloat16;h=tensor_sha(t);source[n]=dict(runtime_name=n,source_file=str(W/file),source_dtype=str(t.dtype),shape=list(t.shape),source_SHA256=h,cast_SHA256=h)
 assert set(source)==set(idx);put('own_full_source_manifest.json',source)
 cfg=MiniCPMVConfig(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation=cfg.vision_config._attn_implementation='eager'
 with contextlib.redirect_stdout(io.StringIO()):
  with torch.device('meta'):model=MiniCPMV(cfg)
 params=dict(model.named_parameters(remove_duplicate=False));assert set(params)==set(source)
 for n,t in params.items():assert list(t.shape)==source[n]['shape']
 del model,params
 inputs=OwnMiniCPMV26Inputs(W)
 g=json.loads((W/'generation_config.json').read_bytes());stop=g.get('eos_token_id',cfg.eos_token_id);stop=stop if isinstance(stop,list) else [stop];assert 151645 in stop and all(isinstance(n,int) for n in stop)
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False);assert len(rows)==4329
 front=[]
 for i,row in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(row['prompt']);text,v=inputs.prepare(prompt,row['media_path']);check_pixels(inputs.image_processor,row['media_path'],v['pixel_values'][0])
  front.append(dict(index=i,source_prompt=row['prompt'],prompt=row['prompt'],actual_model_prompt=prompt,media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),reference=row['reference'],text=text,input_ids=v['input_ids'][0].tolist(),image_bound=[t.tolist() for t in v['image_bound']],patches=len(v['pixel_values'][0]),tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in flatten_tensors(v).items()},independent_uint8_normalization_and_patch_layout_exact=True))
 put('own_original_frontend200.json',front)
 ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='openbmb/MiniCPM-V-2_6',weight_path=str(W),own_weight_files=rr['own_weight_files'],own_assets_SHA256=rr['own_assets_SHA256'],source_tensor_count=len(source),own_source_manifest=str(H/'own_full_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_full_source_manifest.json'),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),dataset_path=str(dataset),dataset_SHA256=sha(dataset),all_own_source_and_meta_parameter_names_shapes_bijective=True,all200_own_original_tokens_pixels_boundaries_exact=True,original_stop_ids=stop,no_NPU_or_learned_CPU_inference_yet=True)
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_inference_yet=True))
  raise

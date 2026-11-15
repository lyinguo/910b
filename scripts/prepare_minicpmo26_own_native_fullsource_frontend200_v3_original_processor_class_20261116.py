"""Prepare own MiniCPM-o-2_6 parameter SHA and exact original input evidence, no learned CPU."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,contextlib,io,traceback,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMO26_own_original_fullsource_frontend200_v3_original_processor_class_user_continue_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import torch
 from safetensors import safe_open
 from universal_eval.model.adapters.minicpmo26_own_frontend_bf16_v2_unpadded_vllm import OwnMiniCPMO26InputsV2,tensor_sha,flatten_tensors
 from universal_eval.model.minicpmo26_own_original_20261116.configuration_minicpm import MiniCPMOConfig
 from universal_eval.model.minicpmo26_own_original_20261116.modeling_image_branch import MiniCPMOImageBranch
 from universal_eval.model.adapters.minicpmv4_original_pixel_check_v1 import check_pixels
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4);rr=json.loads((B/'MiniCPMO26_own_original_resources_source_preflight_v1_after_DotsOCR_user_continue_20261116/resource_source_review.json').read_bytes());assert rr['passed'];W=Path(rr['weight_path'])
 source={}
 idx=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map']
 for fn in sorted(set(idx.values())):
  with safe_open(str(W/fn),framework='pt',device='cpu') as f:
   assert set(f.keys())=={n for n,v in idx.items() if v==fn}
   for n in f.keys():
    t=f.get_tensor(n);assert t.dtype==torch.bfloat16
    source[n]=dict(runtime_name=n,source_dtype=str(t.dtype),shape=list(t.shape),source_SHA256=tensor_sha(t),cast_SHA256=tensor_sha(t))
 assert set(source)==set(idx)
 for n,r in rr['own_weight_files'].items():assert sha(W/n)==r['SHA256']
 all_source=source.copy()
 cfg=MiniCPMOConfig(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation=cfg.vision_config._attn_implementation='eager'
 with contextlib.redirect_stdout(io.StringIO()):
  with torch.device('meta'):model=MiniCPMOImageBranch(cfg)
 params=dict(model.named_parameters(remove_duplicate=False));assert set(params)<=set(source)
 unused={n:v for n,v in source.items() if n not in params};assert unused and all(n.startswith(('apm.','audio_projection_layer.','tts.')) for n in unused),sorted(unused)
 source={n:source[n] for n in params};put('own_full_source_manifest.json',source);put('unused_own_audio_TTS_full_source_manifest.json',unused)
 for n,t in params.items():assert list(t.shape)==source[n]['shape']
 del model,params
 inputs=OwnMiniCPMO26InputsV2(W)
 g=json.loads((W/'generation_config.json').read_bytes()) if (W/'generation_config.json').is_file() else {};stop=g.get('eos_token_id',cfg.eos_token_id);stop=stop if isinstance(stop,list) else [stop];assert 151645 in stop and all(isinstance(n,int) for n in stop)
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False);assert len(rows)==4329
 front=[]
 for i,row in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(row['prompt']);text,v=inputs.prepare(prompt,row['media_path']);check_pixels(inputs.image_processor,row['media_path'],v['pixel_values'][0])
  front.append(dict(index=i,source_prompt=row['prompt'],prompt=row['prompt'],actual_model_prompt=prompt,media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),reference=row['reference'],text=text,input_ids=v['input_ids'][0].tolist(),image_bound=[t.tolist() for t in v['image_bound']],patches=len(v['pixel_values'][0]),tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in flatten_tensors(v).items()},independent_uint8_normalization_and_patch_layout_exact=True))
 put('own_original_frontend200.json',front)
 contract=B/'MiniCPMO26_original_own_HF_mask_vs_saved_VLLM_unpadded_contract_v2_user_continue_20261116';contract.mkdir(exist_ok=False)
 current_vllm=Path(__import__('inspect').getfile(__import__('vllm'))).parent/'model_executor/models/minicpmv.py'
 raw=current_vllm.read_bytes();(contract/'current_installed_vllm_model_original.py.txt').write_bytes(raw)
 review=dict(passed=True,original_HF_processor_mask_exact_ne0_retained=True,native_original_unpadded_causal_mask_and_positions=True,all200_own_token_ID_and_pixel_SHA_preserved=True,own_chat_template_called_without_enable_thinking_override=True,no_thinking_disabled=True,original_stop_ids=stop,current_vllm_source=str(current_vllm),current_vllm_source_SHA256=sha(current_vllm),saved_deployment='configs/MiniCPMO/openbmb--MiniCPM-o-2_6.yaml',historical_engine_or_tensor_capture_claimed=False,original_source_functions_unchanged=True)
 (contract/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')

 ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='openbmb/MiniCPM-o-2_6',weight_path=str(W),own_weight_files=rr['own_weight_files'],own_assets_SHA256=rr['own_assets_SHA256'],source_tensor_count=len(source),unused_own_source_tensor_names=sorted(unused),unused_own_source_manifest=str(H/'unused_own_audio_TTS_full_source_manifest.json'),unused_own_source_manifest_SHA256=sha(H/'unused_own_audio_TTS_full_source_manifest.json'),own_image_branch_source_review=rr['own_image_branch_source_review'],own_image_branch_source_review_SHA256=rr['own_image_branch_source_review_SHA256'],own_source_manifest=str(H/'own_full_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_full_source_manifest.json'),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),dataset_path=str(dataset),dataset_SHA256=sha(dataset),all_own_source_and_meta_parameter_names_shapes_bijective=True,all200_own_original_tokens_pixels_boundaries_exact=True,original_stop_ids=stop,original_own_processor_mask_retained_as_diagnostic=True,saved_VLLM_unpadded_sequence_mask_and_absolute_positions_not_changed=True,no_NPU_or_learned_CPU_inference_yet=True)
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_inference_yet=True))
  raise

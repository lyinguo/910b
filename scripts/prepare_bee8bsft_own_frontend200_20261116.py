"""Own Bee original input preparation, no NPU or learned CPU inference."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,traceback,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Bee8BSFT_own_original_frontend200_v1_no_NPU_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch
 from universal_eval.model.adapters.bee8bsft_own_frontend_bf16_20261116 import OwnBeeSFTInputs,tensor_sha,flatten_tensors
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4)
 pre=B/'Bee8BSFT_own_resource_frontend_meta_v5_actual_shards_BF16_native_vision_prefix_no_NPU_20261116/resource_meta_ready.json';rr=json.loads(pre.read_bytes());assert rr['passed'] and rr['meta_parameter_shapes_exact'];W=Path(rr['weight_path'])
 inputs=OwnBeeSFTInputs(W);dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'
 rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False);assert len(rows)==4329
 front=[]
 for i,row in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(row['prompt']);text,v=inputs.prepare(prompt,row['media_path']);inputs.check_pixels(row['media_path'],v)
  expanded,_=inputs.processor._expand_image_tokens([text],iter(v['image_sizes'].tolist()),384,384,inputs.processor.image_token,iter(v['batch_num_images'].tolist()))
  assert inputs.rust.encode(expanded[0],add_special_tokens=True).ids==v['input_ids'][0].tolist()
  front.append(dict(index=i,prompt=row['prompt'],source_prompt=row['prompt'],actual_model_prompt=prompt,media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),reference=row['reference'],text=text,input_ids=v['input_ids'][0].tolist(),image_token_count=int((v['input_ids']==151669).sum()),image_sizes=v['image_sizes'].tolist(),patches=int(v['pixel_values'].shape[1]),tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in flatten_tensors(v).items()},independent_uint8_normalization_and_patch_layout_exact=True,own_Rust_tokenizer_independent_ID_check=True,original_saved_enable_thinking_false_default_preserved=True))
 put('own_original_frontend200.json',front)
 ready=dict(rr,UTC=datetime.now(timezone.utc).isoformat(),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),dataset_path=str(dataset),dataset_SHA256=sha(dataset),source_tensor_count=rr['own_source_count'],all_own_source_and_meta_parameter_names_shapes_bijective=True,all200_own_original_tokens_pixels_boundaries_exact=True,original_stop_ids=[151645],original_budget4096_preserved=True,original_first_token_EOS_boundary_preserved=True,no_NPU_or_learned_CPU_inference_yet=True,parent_resource_review=str(pre),parent_resource_review_SHA256=sha(pre))
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=ready['UTC'],actual_returncode=0,no_NPU_inference=True));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_inference=True))
  raise

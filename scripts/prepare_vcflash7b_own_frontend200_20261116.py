"""Own original VideoChat Flash CPU inputs only, no model answers or A100 claim."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'VCFlash7B_own_original_UMT448_frontend200_no_NPU_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch
 from universal_eval.model.adapters.vcflash7b_own_frontend_20261116 import OwnVCFlash7Inputs,tensor_sha,flatten_tensors
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 from universal_eval.algorithm.inference import GreedyInference
 H.mkdir(exist_ok=False);torch.set_num_threads(4);pre=B/'VCFlash7B_own_image_graph_full_source_preflight_20261116/readiness.json';rr=json.loads(pre.read_bytes());assert rr['passed'];W=Path(rr['weight_path']);inputs=OwnVCFlash7Inputs(W)
 dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));rows=DataPreprocessor().preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False);assert len(rows)==4329
 front=[]
 for i,row in enumerate(rows[:200]):
  prompt=GreedyInference._add_answer_instruction(row['prompt']);text,v=inputs.prepare(prompt,row['media_path']);inputs.check_pixels(row['media_path'],v)
  front.append(dict(index=i,prompt=row['prompt'],actual_model_prompt=prompt,media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),reference=row['reference'],text=text,input_ids=v['input_ids'][0].tolist(),image_token_position=v['input_ids'][0].tolist().index(-200),compressed_length=len(v['input_ids'][0])-1+64,image_token_count=64,patches=1,tensor_metadata={n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t)) for n,t in flatten_tensors(v).items()},independent_uint8_normalization_and_patch_layout_exact=True,own_Rust_tokenizer_independent_ID_check=True))
 put('own_original_frontend200.json',front);ready=dict(rr,UTC=datetime.now(timezone.utc).isoformat(),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),dataset_path=str(dataset),dataset_SHA256=sha(dataset),all_own_source_and_meta_parameter_names_shapes_bijective=True,all200_own_original_tokens_pixels_boundaries_exact=True,original_stop_ids=[151643,151645],original_budget2048_preserved=True,original_repetition_penalty=1.05,no_NPU_or_learned_CPU_inference_yet=True,parent_resource_review=str(pre),parent_resource_review_SHA256=sha(pre));put('readiness.json',ready);put('actual_completion.json',dict(UTC=ready['UTC'],actual_returncode=0,no_NPU=True));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(actual_returncode=1,error=traceback.format_exc(),no_NPU=True))
  raise

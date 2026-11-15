import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,copy
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path.insert(0,str(R))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
H=B/'DotsOCR_own_original_source_frontend200_v4_exact_target_mirror_user_continue_20261115';H.mkdir(exist_ok=False)
old=B/'DotsOCR_own_original_source_frontend200_v3_exact_original_rescale_user_continue_20261115/readiness.json';ready=js(old);assert ready['passed']
dataset=R/'workspace/datasets/image-to-text-ocr/eval.jsonl'
assert sha(dataset)==ready['dataset_SHA256']=='49aa4d96f9c24c16d31f8cb092ddb0ee839f93a0507c8588675cdd5801b9efd5'
from universal_eval.data.loader import DatasetLoader
from universal_eval.data.registry import TaskDataRegistry
from universal_eval.data.preprocessor import DataPreprocessor
rows=DatasetLoader().load(str(dataset));rows=TaskDataRegistry._resolve_media_paths(rows,str(dataset.parent));pre=DataPreprocessor()
a=pre.preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=True);b=pre.preprocess(rows,task_type='image-text-to-text',apply_template=False,filter_empty=False)
assert len(rows)==len(a)==len(b)==3000 and a==b
front=js(ready['own_frontend200']);assert len(front)==200
for i,(rec,row) in enumerate(zip(front,b[:200])):
 assert row['prompt']==rec['prompt'] and row['reference']==rec['reference']
 assert sha(row['media_path'])==sha(rec['media_path'])==rec['media_file_SHA256']
 rec.update(original_AFS_media_path=rec['media_path'],media_path=str(Path(row['media_path']).resolve()),exact_existing_target_dataset_mirror_verified=True)
out=H/'own_original_frontend200.json';put(out,front)
review=dict(passed=True,source_ready=str(old),source_ready_SHA256=sha(old),dataset_path=str(dataset),dataset_SHA256=sha(dataset),all3000_rows_byte_identical_original_data_and_preprocessing_no_filter_change=True,all200_refs_images_prompt_and_all_pixel_token_metadata_unchanged=True,only_existing_target_dataset_and_exact_identical_media_mirror_paths_rebound=True,no_download_or_borrowed_data_or_NPU_inference=True,no_full_pixel200_rerun=True)
rp=H/'target_mirror_review.json';put(rp,review)
ready.update(dataset_path=str(dataset),own_frontend200=str(out),own_frontend200_SHA256=sha(out),exact_target_mirror_review=str(rp),exact_target_mirror_review_SHA256=sha(rp))
put(H/'readiness.json',ready);print(json.dumps(review,ensure_ascii=False))

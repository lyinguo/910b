"""Prove own original task inputs unchanged under the frozen fixed-set runner."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'AutoGLMPhone9B_own_original_AFS_R_identical_fixed200_no_filter_input_alias_no_NN_20261119';FRONT=B/'AutoGLMPhone9B_own_original_BF16_2048_frontend200_meta_v1_20261119'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
 own=Path('/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl');target=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';expected='2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451';assert sha(own)==sha(target)==expected
 load=lambda path,filtering:DataPreprocessor().preprocess(TaskDataRegistry._resolve_media_paths(DatasetLoader().load(str(path)),str(path.parent)),task_type='image-text-to-text',apply_template=False,filter_empty=filtering)
 before=load(own,True);after=load(target,False);assert len(before)==len(after)==4329
 source=[json.loads(x) for x in (FRONT/'own_original_frontend200.jsonl').read_text().splitlines()];assert sha(FRONT/'own_original_frontend200.jsonl')=='21e7a17a36e7ca4a0a5689c5be4e981704a00cb91b463017f759b2f6fc6e1dd4'
 for i,(a,b) in enumerate(zip(before,after)):
  assert a['prompt']==b['prompt'] and a['reference']==b['reference'] and Path(a['media_path']).name==Path(b['media_path']).name,i
 rows=[]
 for i in range(200):
  a,b=before[i],after[i];f=source[i];assert f['index']==i and a['prompt']==f['prompt'] and a['reference']==f['reference'] and a['media_path']==f['media_path'];assert sha(a['media_path'])==sha(b['media_path'])==f['image_SHA256'];rows.append(dict(index=i,own_original_media_path=a['media_path'],actual_target_media_path=b['media_path'],full_image_SHA256=f['image_SHA256'],prompt=a['prompt'],reference=a['reference'],existing_own_complete_input_ids=f['own_complete_input_ids'],existing_own_all_actual_input_tensor_metadata=f['all_actual_input_tensor_metadata']))
 H.mkdir(exist_ok=False)
 value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id='zai-org/AutoGLM-Phone-9B-Multilingual',own_original_dataset=str(own),actual_target_dataset=str(target),original_and_target_dataset_SHA256=expected,own_original_filter_empty=True,actual_fixed_set_filter_empty=False,all4329_original_preprocessed_prompt_reference_records_equal=True,original_and_target_length=4329,all200_original_fixed_input_order_prompt_ref_and_full_image_bytes_equal=True,original_no_rows_removed_or_added_or_selected=True,existing_own_frontend200=str(FRONT/'own_original_frontend200.jsonl'),existing_own_frontend200_SHA256=sha(FRONT/'own_original_frontend200.jsonl'),rows=rows,no_learned_CPU_or_NPU=True,no_original_frontend_pixel_or_token_replay=True,no_checkpoint_or_other_model_input_control_borrowed=True)
 (H/'own_fixed200_no_filter_input_alias_proof.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');(H/'actual_completion.json').write_text(json.dumps(dict(UTC=value['UTC'],actual_returncode=0,proof_SHA256=sha(H/'own_fixed200_no_filter_input_alias_proof.json'),actual_NEW_model_forwards=0),indent=2)+'\n',encoding='utf8');print(json.dumps(dict(passed=True,proof=str(H/'own_fixed200_no_filter_input_alias_proof.json'),proof_SHA256=sha(H/'own_fixed200_no_filter_input_alias_proof.json'),all200_original_inputs_unchanged=True,actual_NEW_model_forwards=0)))
if __name__=='__main__':main()

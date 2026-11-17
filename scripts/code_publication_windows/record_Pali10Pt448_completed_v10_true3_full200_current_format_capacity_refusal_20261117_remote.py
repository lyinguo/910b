"""Read complete ZIP metadata and own fixed200 shape lower bounds, no NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,math
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117';done=js(J/'completion.json');assert done['actual_returncode']==0;a,=done['accepted'];assert sha(a['proof'])==a['proof_sha256']=='03f7e0fa3798b61836ac126f7980b2d53743aaf6b02f5c4f5caf8671a65286d9';proof=js(a['proof']);assert proof['passed'] and proof['numeric_review_passed'] and proof['sample_count']==3
folder=Path(a['report_dir']);before={str(folder/n):sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']};line=js(folder/'operator_lineage.json')['actual_execution'];idx=js(line['full_numeric_group_index']);group=js(idx['groups'][0]['path']);archive=Path(group['archive']);assert archive.stat().st_size==group['archive_bytes'];parts=[];rates=[]
with zipfile.ZipFile(archive) as z:
 for v in z.infolist():parts.append(dict(name=v.filename,uncompressed=v.file_size,compressed=v.compress_size))
 members={v['name']:v for v in parts}
 for f in group['frames']:
  for row in f['cache']:
   for field in ['incoming_K','incoming_V']:
    ref=row[field];v=members[ref['array']+'.npy'];count=math.prod(ref['shape']);assert ref['dtype']=='torch.bfloat16' and ref['shape'][-2:]==[8,256];rates.append(v['compressed']/(count*2))
data=js(R/js(J/'job.json')['spec']['config'])['data']['dataset_path'];rows=[json.loads(s) for s in Path(data).read_text().splitlines()];assert len(rows)==1459
small_input=group['frames'][0]['request_state_before_forward'];assert all(len(v['prompt_token_ids'])>=1024 for v in small_input.values())
kv_bytes_per_position=42*8*256*2*2;minimum_image_positions=1024;component_raw=200*minimum_image_positions*kv_bytes_per_position;minimum_observed_compression=min(rates);component_empirical=math.ceil(component_raw*minimum_observed_compression);needed=4*component_empirical+6*1024**3;free=shutil.disk_usage(B).free
H=B/'Pali10Pt448_completed_v10_true3_current_full_value_format_200_capacity_refused_no_stage_replay_20261117';H.mkdir(exist_ok=False)
review=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id='google/paligemma2-10b-pt-448',passed_source_and_complete_ZIP_readonly_review=True,capacity_passed=free>=needed,formal200_approved=False,NEWfresh3_or_200_started=False,true3_accepted_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],actual_full3_sample_count=3,actual_native_forward_calls=proof['actual_native_model_forward_calls'],actual_full3_compressed_bytes=archive.stat().st_size,actual_all_ZIP_parts=parts,minimum_observed_compressed_bytes_per_BF16_KV_byte=minimum_observed_compression,all200_own_shape_component_bounds=[dict(index=i,media_path=rows[i]['media_path'],original_image448_positions_at_least=1024,original_full42_layers_KV8_head256=True) for i in range(200)],single_KV_copy_raw_component_bytes=component_raw,conservative_observed_compression_component_bytes=component_empirical,factor=4,extra_budget_bytes=6*1024**3,required_bytes_from_this_component_alone=needed,actual_free_bytes=free,estimate_is_empirical_not_worst_case_guarantee=True,full200_would_also_need_QKV_heads_masks_vision_and_metadata_not_in_component=True,complete_count200_numeric_verifier_still_required_if_capacity_restored=True,no_threshold_budget_precision_or_retention_change=True,no_matrix_deleted=True,old_small3_compat3_B10_not_to_be_replayed=True,full_five_reports_SHA256=before)
assert before=={str(folder/n):sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']}
(H/'review.json').write_bytes(json.dumps(review,ensure_ascii=False,indent=2).encode());(H/'actual_completion.json').write_bytes(json.dumps(dict(UTC=review['UTC'],actual_returncode=0,capacity_passed=review['capacity_passed'],no_NN=True,review_SHA256=sha(H/'review.json')),indent=2).encode());print(json.dumps({k:v for k,v in review.items() if k not in ['actual_all_ZIP_parts','all200_own_shape_component_bounds','full_five_reports_SHA256']},ensure_ascii=False))

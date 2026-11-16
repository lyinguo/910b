"""Measure existing completed3 ZIP members; no array/weight/NN recomputation.

The adapter retains full activation-prefill values only for sample indices <3.
Inputs, heads, masks, positions and every native KV value remain mandatory for
all samples. This changes the forecast, never inference or retained evidence.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,math
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'RedHatGemma4E2B_own_original_BF16_true3_v1_saved_HF_F32_real_FLOCK_20261116'
H=B/'RedHatGemma4E2B_own_original_BF16_complete_true3_capacity_review_v1_saved_HF_F32_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 completion=js(J/'completion.json');assert completion['actual_returncode']==0
 accepted=completion['accepted'][-1];assert accepted['sample_count']==3
 proof=js(accepted['proof']);assert proof['passed'] and sha(accepted['proof'])==accepted['proof_sha256']
 folder=Path(accepted['report_dir']);line=js(folder/'operator_lineage.json')['actual_execution']
 front_protocol=js(line['protocol_file']);source=R/'universal_eval/model/adapters/redhat_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py'
 assert sha(source)==front_protocol['implementation_source_SHA256'][str(source.relative_to(R))]
 observer=R/'universal_eval/model/redhat_gemma4_e2b_own_20261116/observer.py'
 assert sha(observer)==front_protocol['implementation_source_SHA256'][str(observer.relative_to(R))]
 assert 'self.index<3' in source.read_text() # actual inherited _install stage observer
 trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];entries=[]
 for row in trace:
  sample=js(row['actual_execution_file']);archive=Path(sample['archive']);assert archive.stat().st_size==sample['archive_bytes']
  with zipfile.ZipFile(archive) as z:
   retained={name.replace('.','_')+'.npy' for name in sample['activation_prefill']}
   members={rec.filename:rec for rec in z.infolist()};assert retained.issubset(members)
   prefix_payload=sum(members[name].compress_size for name in retained)
  common=sample['archive_bytes']-prefix_payload # conservatively includes all ZIP headers
  assert common>0
  entries.append(dict(index=sample['index'],prompt_length=sample['prompt_length'],forwards=sample['actual_native_forwards'],archive=str(archive),archive_SHA256=sample['archive_SHA256'],archive_bytes=sample['archive_bytes'],first3_only_prefill_payload_bytes=prefix_payload,all_sample_mandatory_payload_and_headers_bytes=common,first3_only_members=sorted(retained)))
 full3=sum(e['archive_bytes'] for e in entries);base=max(e['all_sample_mandatory_payload_and_headers_bytes'] for e in entries)
 # Two new first3 sets: approved fresh3, then the first3 of fixed200. The other
 # 197 keep all mandatory values but omit the same prefill-only members as the
 # unchanged implementation. Keep the original factor4 and reserve6GiB.
 front=[json.loads(s) for s in Path(front_protocol['own_frontend200']).read_text().splitlines()];assert len(front)==200
 prompt_lengths=[x['inputs']['input_ids']['shape'][1] for x in front];min_observed=min(e['prompt_length'] for e in entries);forecast_common=[math.ceil(base*max(1.,(N/min_observed)**2)) for N in prompt_lengths[3:]]
 expected=4*(2*full3+sum(forecast_common))+6*1024**3
 assert expected>=4*(2*full3+197*base)+6*1024**3
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=front_protocol['model_id'],source_SHA256=sha(source),true3_proof_SHA256=accepted['proof_sha256'],entries=entries,forecast_formula='4 * (2 * actual_full3_bytes + sum197(max_common * max(1,(own_prompt/min_observed_prompt)**2)))) + 6 GiB',all200_own_original_input_shapes_checked=True,all200_original_prompt_lengths=prompt_lengths,all197_forecast_mandatory_common_bytes=forecast_common,original2048_budget_preserved_not_a_worst_case_guarantee=True,empirical_estimate_bytes=expected,actual_free_bytes=shutil.disk_usage(B).free,capacity_passed=shutil.disk_usage(B).free>=expected,empirical_not_worst_case_guarantee=True,unchanged_hard_reserve_bytes=front_protocol['hard_reserve_bytes'],unchanged_log_reserve_bytes=front_protocol['log_reserve_bytes'],unchanged_sample_guard_bytes=front_protocol['per_sample_lossless_space_estimate_bytes'],all_head_cache_input_mask_position_values_retained=True,no_matrix_deleted=True,no_model_math_precision_budget_or_retention_change=True)
 H.mkdir(exist_ok=False);(H/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

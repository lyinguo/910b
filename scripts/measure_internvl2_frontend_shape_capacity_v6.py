"""Forecast mandatory complete archives from measured ZIP bytes and all own input shapes.

No NN, array recomputation, pruning, guard reduction, or new qualification.
The factor4 and reserve6GiB remain; this is empirical, not a worst-case guarantee.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,math,shutil
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'InternVL2_2B_own_native_true3_v5_real_FLOCK_1535_20261115';H=B/'InternVL2_2B_actual_all_frontend_shape_capacity_v6_1535_20261115'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 done=js(J/'completion.json');assert done['actual_returncode']==0;a=done['accepted'][-1];proof=js(a['proof']);assert sha(a['proof'])==a['proof_sha256'] and proof['passed'] and proof['sample_count']==3
 line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];assert len(front)==200
 rows=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];entries=[];rates={};max_steps=0
 for row in rows:
  e=js(row['actual_execution_file']);index=e['index'];N=len(front[index]['input_ids']);tiles=front[index]['tiles'];steps=e['actual_native_forwards'];max_steps=max(max_steps,steps);parts={};first3=0
  with zipfile.ZipFile(e['archive']) as z:
   for v in z.infolist():
    n=v.filename
    if n.startswith('LM_layer_'):first3+=v.compress_size;continue
    if n in ['all_full_head_logits.npy','native_cache_appends.npy']:kind='steps';denom=steps if n=='all_full_head_logits.npy' else max(steps-1,1)
    elif n in ['pixel_values.npy','vision_features.npy']:kind='tiles';denom=tiles
    elif n.startswith('mask_'):kind='prompt_square';denom=N*N
    elif n=='generated_tokens.npy':kind='total_tokens';denom=N+steps
    else:kind='prompt';denom=N
    assert denom>0;parts[n]=dict(compressed_bytes=v.compress_size,kind=kind,denominator=denom);rec=rates.setdefault(n,dict(kind=kind,max_ratio=0.0));assert rec['kind']==kind;rec['max_ratio']=max(rec['max_ratio'],v.compress_size/denom)
  overhead=Path(e['archive']).stat().st_size-first3-sum(v['compressed_bytes'] for v in parts.values());assert overhead>=0
  entries.append(dict(index=index,archive=e['archive'],archive_SHA256=e['archive_SHA256'],archive_bytes=e['archive_bytes'],N=N,tiles=tiles,steps=steps,first3_only_payload=first3,ZIP_headers_bytes=overhead,parts=parts))
 forecasts=[];overhead=max(e['ZIP_headers_bytes'] for e in entries)
 for row in front[3:]:
  N=len(row['input_ids']);factors=dict(steps=max_steps,tiles=row['tiles'],prompt=N,prompt_square=N*N,total_tokens=N+max_steps);parts={n:math.ceil(rec['max_ratio']*factors[rec['kind']]) for n,rec in rates.items()};forecasts.append(dict(index=row['index'],N=N,tiles=row['tiles'],empirical_observed_max_steps=max_steps,estimated_mandatory_bytes=sum(parts.values())+overhead))
 full3=sum(e['archive_bytes'] for e in entries);forecast=4*(2*full3+sum(e['estimated_mandatory_bytes'] for e in forecasts))+6*1024**3
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=p['model_id'],true3_proof_SHA256=a['proof_sha256'],frontend_SHA256=sha(p['own_frontend200']),entries=entries,component_max_measured_compression_ratios=rates,all_remaining197_forecasts=forecasts,empirical_estimate_bytes=forecast,actual_free_bytes=shutil.disk_usage(B).free,capacity_passed=shutil.disk_usage(B).free>=forecast,forecast_formula='4 * (2 * measured_full3 + sum197(max_component_measured_ratio * own_frozen_input_shape)) + 6 GiB',empirical_not_worst_case_guarantee=True,observed_generation_max_not_a_2048_worst_case_claim=max_steps,original_hard_log_preflight_and_per_sample_guards_unchanged=True,no_matrix_deleted=True,no_model_math_precision_budget_or_retention_change=True)
 H.mkdir(exist_ok=False);(H/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

"""Observed lossless archive/own200 shapes, unchanged factor4 + reserve6GiB."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,math,shutil,numpy as np
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'Eagle2_1B_own_native_true3_v1_real_FLOCK_1905_20261115';H=B/'Eagle2_1B_actual_all_frontend_shape_capacity_v1_1910_20261115'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 done=js(J/'completion.json');assert done['actual_returncode']==0;a=done['accepted'][-1];proof=js(a['proof']);assert sha(a['proof'])==a['proof_sha256'] and proof['passed'] and proof['sample_count']==3
 line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);front=js(p['own_frontend200']);assert len(front)==200;entries=[];rates={};max_steps=0
 for row in [json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]:
  e=js(row['actual_execution_file']);i=e['index'];N=len(front[i]['input_ids']);steps=e['actual_native_forwards'];tiles=front[i]['tiles'];max_steps=max(max_steps,steps);parts={};only_first3=0
  with zipfile.ZipFile(e['archive']) as z:
   for rec in z.infolist():
    n=rec.filename
    if n.startswith('stage_'):only_first3+=rec.compress_size;continue
    with z.open(rec) as f:
     version=np.lib.format.read_magic(f);shape,_,_=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
    if n in ['all_original_full_head_logits.npy','all_head_inputs.npy']:kind='steps';denom=steps
    elif n=='native_cache_decode_delta.npy':kind='decode_steps';denom=max(steps-1,1)
    elif n in ['pixels_BF16.npy','own_visual_features.npy']:kind='tiles';denom=tiles
    elif len(shape)==4 and shape[-2:]==(N,N):kind='prompt_square';denom=N*N
    elif N in shape:kind='prompt';denom=N
    else:kind='constant';denom=1
    parts[n]=dict(compressed_bytes=rec.compress_size,kind=kind,denominator=denom,shape=shape);rate=rates.setdefault(n,dict(kind=kind,max_ratio=0.0));assert rate['kind']==kind;rate['max_ratio']=max(rate['max_ratio'],rec.compress_size/denom)
  overhead=Path(e['archive']).stat().st_size-only_first3-sum(v['compressed_bytes'] for v in parts.values());assert overhead>=0;entries.append(dict(index=i,archive=e['archive'],archive_SHA256=e['archive_SHA256'],archive_bytes=e['archive_bytes'],N=N,tiles=tiles,steps=steps,first3_only_payload=only_first3,ZIP_headers_bytes=overhead,parts=parts))
 forecasts=[];overhead=max(e['ZIP_headers_bytes'] for e in entries)
 for row in front[3:]:
  N=len(row['input_ids']);factors=dict(steps=max_steps,decode_steps=max(max_steps-1,1),tiles=row['tiles'],prompt=N,prompt_square=N*N,constant=1);size=sum(math.ceil(v['max_ratio']*factors[v['kind']]) for v in rates.values())+overhead;forecasts.append(dict(index=row['index'],N=N,tiles=row['tiles'],observed_max_steps=max_steps,estimated_mandatory_bytes=size))
 full3=sum(e['archive_bytes'] for e in entries);estimate=4*(2*full3+sum(v['estimated_mandatory_bytes'] for v in forecasts))+6*1024**3;free=shutil.disk_usage(B).free
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=p['model_id'],true3_proof_SHA256=a['proof_sha256'],frontend_SHA256=sha(p['own_frontend200']),entries=entries,component_max_measured_compression_ratios=rates,all_remaining197_forecasts=forecasts,empirical_estimate_bytes=estimate,actual_free_bytes=free,capacity_passed=free>=estimate,forecast_formula='4 * (2 * measured_full3 + sum197(max_measured_component_ratio * own_input_shapes)) + 6 GiB',empirical_not_worst_case_guarantee=True,observed_generation_max_not_a_256_worst_case_claim=max_steps,original_hard_log_preflight_and_per_sample_guards_unchanged=True,no_matrix_deleted=True,no_model_math_precision_budget_or_retention_change=True)
 H.mkdir(exist_ok=False);(H/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

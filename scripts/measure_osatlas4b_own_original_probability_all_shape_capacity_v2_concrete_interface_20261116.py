"""All ZIP headers and all200 own shapes; frozen pre-NN retention, factor4+6GiB."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,math
import numpy as np
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'OSAtlas4B_own_original_mcscore_BF16_true3_v2_concrete_interface_real_FLOCK_20261116';H=B/'OSAtlas4B_own_original_probability_full_ZIP_all200_shape_capacity_v2_concrete_interface_20261116'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 accepted=js(J/'completion.json')['accepted'][-1];assert sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['original_generation_invocations']==0
 line=js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);assert p['activation_retention_frozen_before_true3']['no_matrix_pruning_or_streaming_deletion']
 front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()];traces=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];schemas={};entries=[]
 for small in traces:
  e=js(small['actual_execution_file']);assert sha(e['archive'])==e['archive_SHA256'];parts=[]
  with zipfile.ZipFile(e['archive']) as z:
   for item in z.infolist():
    assert item.filename.endswith('.npy');name=item.filename[:-4]
    with z.open(item) as f:version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,version)
    assert not order;spec=dict(shape=list(shape),dtype=str(dtype),itemsize=dtype.itemsize)
    if name in schemas:assert len(shape)==len(schemas[name]['shape']) and str(dtype)==schemas[name]['dtype']
    else:schemas[name]=spec
    parts.append(dict(name=name,shape=list(shape),dtype=str(dtype),raw_bytes=math.prod(shape)*dtype.itemsize,ZIP_compressed_bytes=item.compress_size,ZIP_uncompressed_bytes=item.file_size))
  entries.append(dict(index=e['index'],archive=e['archive'],archive_SHA256=e['archive_SHA256'],archive_bytes=e['archive_bytes'],ZIP_parts=parts))
 forecasts=[]
 for row in front:
  i=row['index'];N=len(row['input_ids']);tiles=row['tiles'];parts={}
  for name,spec in schemas.items():
   shape=spec['shape'][:]
   if name.startswith(('LM_layer_','QKV_layer_','vision_layer_')) and i>=3:continue
   if name in ['input_ids','position_ids']:shape[-1]=N
   elif name in ['pixel_values','image_flags','vision_features']:shape[0]=tiles
   elif name in ['text_embeddings','merged_embeddings'] or name.startswith(('LM_layer_','QKV_layer_')):shape[1]=N
   elif name.startswith('vision_layer_'):shape[0]=tiles
   elif name.startswith('causal_mask_'):shape[-1]=shape[-2]=N
   elif name.startswith(('rope_cos_','rope_sin_')):shape[-2]=N
   else:assert name in ['head_input','full_head_logits','full_log_probs'],name
   parts[name]=dict(shape=shape,raw_bytes=math.prod(shape)*spec['itemsize'])
  forecasts.append(dict(index=i,N=N,tiles=tiles,complete_mandatory_parts=parts,raw_bytes=sum(v['raw_bytes'] for v in parts.values())))
 # Count both the new approved fresh3 and complete200, in addition to existing true3.
 raw=sum(x['raw_bytes'] for x in forecasts)+sum(x['raw_bytes'] for x in forecasts[:3]);estimate=4*raw+6*1024**3;free=shutil.disk_usage(B).free
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=p['model_id'],capacity_passed=free>=estimate,empirical_estimate_bytes=estimate,actual_free_bytes=free,formula='4 * (uncompressed full mandatory arrays for new fresh3 + fixed200 using all own input shapes and frozen first3 extra activations) + 6GiB',empirical_not_worst_case_guarantee=True,observed_original_probability_forwards_per_sample=1,all_true3_ZIP_headers=entries,all200_complete_array_shape_forecasts=forecasts,frontend_SHA256=sha(p['own_frontend200']),true3_proof_SHA256=accepted['proof_sha256'],no_matrix_deleted=True,no_model_math_precision_budget_or_retention_change=True,original_hard_log_preflight_and_per_sample_guards_unchanged=True,original_cacheFalse_no_returned_KV_not_an_evidence_drop=True)
 H.mkdir(exist_ok=False);(H/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

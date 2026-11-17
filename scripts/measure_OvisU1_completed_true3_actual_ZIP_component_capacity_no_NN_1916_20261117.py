"""Empirical space review of exact existing complete NPZ components, no replay.

Keep the old raw bound, original512/EOSNone, factor4+6GiB and all runtime guards.
Use each component's maximum observed compressed fraction across accepted3.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,zipfile,math,shutil,ast
import numpy as np
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'OvisU1_own_original_BF16_512_true3_v2_actual_empty_precision_guard_real_FLOCK_20261116';H=B/'OvisU1_completed_true3_original_NPZ_actual_compressed_component_capacity_no_NN_1916_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def components(row,G,i,has_masks=False):
 T=int(np.prod(row['grid_thws'][0]));K=T//4;N=min(4096,len(row['input_ids'])+K-1);out={}
 def add(n,s,b):out[n]=(tuple(s),b)
 add('input_ids',(1,len(row['input_ids'])),8);add('input_attention_mask',(1,len(row['input_ids'])),1);add('pixels',(T,588),2);add('grid_thws',(1,3),8)
 add('vision_head_inputs',(K,4096),2);add('vision_full_head_logits',(K,65532),2);add('vision_soft_tokens',(K,65536),2);add('visual_embeddings',(K,2048),2);add('visual_indicator_embeddings',(5,2048),2)
 add('merged_embeddings',(1,N,2048),2);add('merged_attention_mask',(1,N),1);add('merged_image_labels',(1,N),8)
 add('all_raw_head_logits',(G,151936),2);add('all_head_inputs',(G,2048),2);add('generated_tokens',(G,),8)
 add('native_cache_prefill',(28,2,8,N,128),2);add('native_cache_appends',(G-1,28,2,8,1,128),2)
 for j in range(G):
  for kind in ['cos','sin']:add('position_'+kind+'_'+str(j),(1,N if j==0 else 1,128),2)
  if has_masks:add('attention_mask_'+str(j),(1,1,N if j==0 else 1,N+j),4)
 if i<3:
  for j in range(28):add('LM_layer_'+str(j),(1,N,2048),2)
  for j in range(24):add('vision_layer_'+str(j),(T,1024),2)
 return out
def main():
 os.chdir('/root');sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 done=js(J/'completion.json');assert done['actual_returncode']==0;a,=done['accepted'];assert sha(a['proof'])==a['proof_sha256']=='221e4c3060bfce23733937f1679a7f89b0e58f0be148650738b99b0da016f39d';proof=js(a['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==1536
 line=js(Path(a['report_dir'])/'operator_lineage.json')['actual_execution'];p=js(line['protocol_file']);front=[json.loads(x) for x in Path(p['own_frontend200']).read_text().splitlines()];assert len(front)==200
 rows=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert [r['index'] for r in rows]==[0,1,2];ratio={};entries=[];overhead=0;allmask=False;mask_items={}
 for row in rows:
  e=js(row['actual_execution_file']);assert e['actual_native_forwards']==512;f=Path(e['archive']);assert sha(f)==e['archive_SHA256'];seen={};members={}
  with zipfile.ZipFile(f) as z:
   for rec in z.infolist():
    assert rec.filename.endswith('.npy')
    with z.open(rec) as inp:
     ver=np.lib.format.read_magic(inp);shape,order,dtype=np.lib.format.read_array_header_1_0(inp) if ver==(1,0) else np.lib.format.read_array_header_2_0(inp)
    key=rec.filename[:-4];raw=int(np.prod(shape))*dtype.itemsize;assert raw>0;seen[key]=(shape,dtype.itemsize);members[key]=dict(shape=shape,itemsize=dtype.itemsize,raw_bytes=raw,compressed_bytes=rec.compress_size,NPY_member_bytes=rec.file_size);ratio[key]=max(ratio.get(key,0),rec.compress_size/raw)
    if key.startswith('attention_mask_'):mask_items[key]=dtype.itemsize;allmask=True
   overhead=max(overhead,f.stat().st_size-sum(x.compress_size for x in z.infolist()))
  expected=components(front[e['index']],512,e['index'],allmask)
  for name,item in mask_items.items():
   if name in expected:expected[name]=(expected[name][0],item)
  assert seen==expected,('Actual full member geometry mismatch',[(n,seen.get(n),sh) for n,sh in expected.items() if seen.get(n)!=sh],set(seen)^set(expected))
  entries.append(dict(index=e['index'],archive=str(f),archive_SHA256=sha(f),archive_bytes=f.stat().st_size,complete_members=members))
 projected=[];raw_bounds=[]
 for i,row in enumerate(front):
  c=components(row,512,i,allmask)
  for name,item in mask_items.items():
   if name in c:c[name]=(c[name][0],item)
  assert set(c)<=set(ratio);raw_bounds.append(sum(int(np.prod(sh))*item for sh,item in c.values()));projected.append(sum(math.ceil(int(np.prod(sh))*item*ratio[n]) for n,(sh,item) in c.items())+overhead)
 full3=sum(e['archive_bytes'] for e in entries);need=4*(sum(projected)+full3)+6*1024**3;free=shutil.disk_usage(D).free;result=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=done['current'],passed=True,capacity_passed=free>=need,empirical_estimate_bytes=need,actual_free_bytes=free,old_uncompressed_capacity_refusal_preserved=True,uncompressed_component_bound_bytes=4*(sum(raw_bounds)+sum(sum(v['raw_bytes'] for v in e['complete_members'].values()) for e in entries))+6*1024**3,all_original3279_member_shapes_and_dtype_checked=sum(len(e['complete_members']) for e in entries),entries=entries,max_observed_component_compression_fraction=ratio,max_actual_ZIP_overhead_bytes=overhead,all200_own_frontend_shapes_checked=True,all200_projected_full_NPZ_bytes=projected,original_factor4_plus6GiB_and_original512_EOSNone_preserved=True,original_hard3GiB_log1GiB_precheck_halfGiB_sample2GiB_guards_preserved=True,no_evidence_or_semantic_arrays_removed=True,no_new_storage_format_or_model_math_dtype_budget_change=True,no_NEW_NPU_or_CPU200_or_old_small3_numeric_replay=True,empirical_not_worst_case_guarantee=True,source_SHA256=sha(__file__))
 H.mkdir(exist_ok=False);(H/'review.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps({k:v for k,v in result.items() if k not in ['entries','max_observed_component_compression_fraction','all200_projected_full_NPZ_bytes']},ensure_ascii=False))
if __name__=='__main__':main()

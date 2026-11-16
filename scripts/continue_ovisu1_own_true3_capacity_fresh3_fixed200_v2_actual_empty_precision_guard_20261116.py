"""Immediately run own OvisU1 true3, review full shapes, NEWfresh3/fixed200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,os,fcntl,traceback,time,shutil,subprocess,zipfile
import numpy as np
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'OvisU1_own_true3_full_shape_capacity_fresh3_fixed200_continuation_v2_actual_empty_precision_guard_20261116';J3=B/'OvisU1_own_original_BF16_512_true3_v2_actual_empty_precision_guard_real_FLOCK_20261116';J=B/'OvisU1_own_original_BF16_512_approved_fresh3_fixed200_v2_actual_empty_precision_guard_real_FLOCK_20261116'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def components(row,G,i,has_masks=False):
 T=int(np.prod(row['grid_thws'][0]));K=T//4;N=min(4096,len(row['input_ids'])+K-1);out={}
 def add(n,s,b):out[n]=(tuple(s),b)
 add('input_ids',(1,len(row['input_ids'])),8);add('input_attention_mask',(1,len(row['input_ids'])),1);add('pixels',(T,588),2);add('grid_thws',(1,3),8)
 add('vision_head_inputs',(K,4096),2);add('vision_full_head_logits',(K,65532),2);add('vision_soft_tokens',(K,65536),2);add('visual_embeddings',(K,2048),2);add('visual_indicator_embeddings',(5,2048),2)
 add('merged_embeddings',(1,N,2048),2);add('merged_attention_mask',(1,N),1);add('merged_image_labels',(1,N),8)
 add('all_raw_head_logits',(G,151936),2);add('all_head_inputs',(G,2048),2);add('generated_tokens',(G,),8)
 add('native_cache_prefill',(28,2,8,N,128),2);add('native_cache_appends',(G-1,28,2,8,1,128),2)
 for j in range(G):
  for kind in ['cos','sin']:add('position_'+kind+'_'+str(j),(1,N if j==0 else 1,128),4)
  if has_masks:add('attention_mask_'+str(j),(1,1,N if j==0 else 1,N+j),4)
 if i<3:
  for j in range(28):add('LM_layer_'+str(j),(1,N,2048),2)
  for j in range(24):add('vision_layer_'+str(j),(T,1024),2)
 return out
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists() and not J.exists() and not (J3/'state.json').exists() and not (J3/'completion.json').exists()
 parent=js(J3/'job.json');batch.check_sources(parent['spec']);unchanged(parent['protected_canonical_hashes']);roles=['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in roles] and batch.device_idle(parent['spec']) and shutil.disk_usage(D).free>=12*1024**3
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (H/'true3_runner_stdout.log').open('xb') as out,(H/'true3_runner_stderr.log').open('xb') as err:c3=subprocess.Popen(['python3','-B',str(runner),str(J3)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=env)
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_own_original_OvisU1_true3_then_independent_verification',job=str(J3),child_PID=c3.pid,no_G_or_canonical_write=True));until=time.monotonic()+14400
 while not (J3/'completion.json').exists() or runner_lock_owners() or [p for p in project_processes() if p['role'] in roles]:
  assert time.monotonic()<until,'Original true3/verification deadline exceeded; no signal sent';time.sleep(10)
 assert c3.wait()==0;done=js(J3/'completion.json');assert done['actual_returncode']==done['verifier_returncode']==0
 accepted,=done['accepted'];assert accepted['sample_count']==3 and sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['actual_NPU_inference'] and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['all_native_KV_values_losslessly_reconstructed']
 score=validate_report(R,accepted['report_dir'],parent['model_id'],'image-text-to-text',3,proof);assert score==accepted['score'];batch.check_sources(parent['spec']);unchanged(parent['protected_canonical_hashes'])
 for p,h in proof['external_evidence_sha256'].items():assert sha(p)==h
 cfg=js(R/parent['spec']['config']);proto=js(cfg['model']['kwargs']['protocol_file']);front=[json.loads(x) for x in Path(proto['own_frontend200']).read_text().splitlines()];line=js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution'];trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()]
 actual_sizes=[];checked=0;gs=[]
 for row in trace:
  e=js(row['actual_execution_file']);G=e['actual_native_forwards'];gs.append(G);expected=components(front[e['index']],G,e['index']);seen={}
  with zipfile.ZipFile(e['archive']) as z:
   for name in z.namelist():
    assert name.endswith('.npy')
    with z.open(name) as f:
     v=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format.read_array_header_1_0(f) if v==(1,0) else np.lib.format.read_array_header_2_0(f)
    seen[name[:-4]]=(shape,dtype.itemsize);checked+=1
  N=e['merged_length']
  for key,(shape,item) in seen.items():
   if key.startswith('attention_mask_'):assert item in [1,2,4] and shape[:2]==(1,1) and (shape==(1,1,N,N) or (shape[2]==1 and N<shape[3]<=N+G-1))
  actual_shapes={k:v for k,v in seen.items() if not k.startswith('attention_mask_')};expected_shapes={k:v for k,v in expected.items() if not k.startswith('attention_mask_')}
  assert actual_shapes==expected_shapes,('Unreviewed or mis-sized lossless component',set(actual_shapes)^set(expected_shapes),[(n,actual_shapes.get(n),v) for n,v in expected_shapes.items() if actual_shapes.get(n)!=v]);actual_sizes.append(sum(int(np.prod(sh))*item for sh,item in seen.values()))
 assert all(all(row['attention_mask']) for row in front), 'Full200 padding contract needs explicit mask budget review'
 desired=max(gs);has_masks=any(str(n).startswith('attention_mask_') for n in seen);sizes=[sum(int(np.prod(sh))*item for sh,item in components(row,desired,i,has_masks).values()) for i,row in enumerate(front)]
 need=(sum(sizes)+sum(actual_sizes))*4+6*1024**3;free=shutil.disk_usage(D).free
 cap=dict(passed=free>=need,UTC=now(),empirical_estimate_bytes=need,actual_free_bytes=free,complete_lossless_ZIP_array_member_shapes_checked=checked,all200_original_frontend_shapes_checked=True,per_sample_projected_uncompressed_bytes=sizes,NEWfresh3_full_observed_components_included=True,observed_generated_token_counts=gs,head_decode_estimate_generated_tokens=desired,original_configured512_and_original_EOS_None_preserved=True,original_factor4_plus6GiB_preserved=True,all_full_head_KV_input_pixels_masks_positions_and_first3_layers_preserved=True,original_3GiB_hard_1GiB_log_halfGiB_preflight_2GiB_sample_guards_preserved=True,empirical_not_worst_case_guarantee=True,no_evidence_deleted=True)
 put('capacity_review.json',cap);assert cap['passed'],'Insufficient actual project disk; retain evidence and continue another eligible model'
 approval=dict(UTC=now(),approved=True,model_id=parent['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),first200_is_measurement_not_predetermined_5pct_qualification=True,no_model_math_precision_budget_or_current_evidence_deletion=True,small_sample_Accuracy=score,small_sample_error_to_full200_D_not_an_extra_first200_gate=True);put('first200_measurement_approval.json',approval)
 newproto=copy.deepcopy(proto);newproto['static_evidence_SHA256'].update({str(H/'capacity_review.json'):sha(H/'capacity_review.json'),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')});pp=R/'configs/protocols/ovisu1_own_original_understanding_BF16_512_v2_actual_empty_precision_guard_approved_measurement_20261116.json';create(pp,newproto)
 config=R/'configs/OvisU1/AIDC-AI--Ovis-U1-3B-own-original-BF16-512-native-v2-actual-empty-precision-guard-approved-first200-20261116.yaml';cfg['name']=config.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(config,cfg)
 job=copy.deepcopy(parent);job.update(stages=[3,200],created_at=now());spec=job['spec'];spec.update(approved=True,config=str(config.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));spec['source_sha256'].update({str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)})
 validate_config(cfg,parent['model_id']);batch.check_sources(spec);entry=js(D/'state.json')['models'][parent['model_id']]
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,parent['model_id'])
 scope(spec,[3,200],False);unchanged(job['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(spec);create(J/'job.json',job)
 with (H/'runner_stdout.log').open('xb') as out,(H/'runner_stderr.log').open('xb') as err:c=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=env)
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200_original_OvisU1_BF16_512',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=c.pid,no_G_or_canonical_write=True));rc=c.wait();final=js(J/'completion.json');assert rc==final['actual_returncode']==0
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=final,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),original_evidence_and_G_unchanged=True,disk_alert_is_not_chat_notification=True))
  raise

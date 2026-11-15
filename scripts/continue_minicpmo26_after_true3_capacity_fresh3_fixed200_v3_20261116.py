"""Continue own MiniCPM-o-2_6 from accepted true3 to approved NEWfresh3/fixed200; never publish G."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,os,fcntl,traceback,time,shutil,subprocess,zipfile,math
import numpy as np
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMO26_true3_then_complete_shape_capacity_fresh3_fixed200_v3_continuation_user_continue_20261116'
J3=B/'MiniCPMO26_own_original_BF16_4096_true3_v2_real_FLOCK_user_continue_20261116'
J=B/'MiniCPMO26_own_original_BF16_4096_approved_fresh3_fixed200_v3_real_FLOCK_user_continue_20261116'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists() and not J.exists()
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_same_true3_actual_completion_and_original_lock_release',job=str(J3),no_NPU_or_G_write_by_waiter=True))
 until=time.monotonic()+14400
 while not (J3/'completion.json').exists() or runner_lock_owners() or [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]:
  assert time.monotonic()<until,'Original true3/verification or lock did not finish before prospective controller deadline'
  time.sleep(10)
 done=js(J3/'completion.json');assert done['actual_returncode']==0 and done['verifier_returncode']==0
 accepted,=done['accepted'];assert accepted['sample_count']==3 and sha(accepted['proof'])==accepted['proof_sha256']
 proof=js(accepted['proof']);assert proof['passed'] and proof['actual_NPU_inference'] and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['all_native_KV_values_losslessly_reconstructed']
 score=validate_report(R,accepted['report_dir'],'openbmb/MiniCPM-o-2_6','image-text-to-text',3,proof);assert score==accepted['score']
 parent=js(J3/'job.json');batch.check_sources(parent['spec']);unchanged(parent['protected_canonical_hashes'])
 for p,h in proof['external_evidence_sha256'].items():assert sha(p)==h
 cfg=js(R/parent['spec']['config']);proto=js(cfg['model']['kwargs']['protocol_file']);front=js(proto['own_frontend200']);assert len(front)==200
 line=js(Path(accepted['report_dir'])/'operator_lineage.json')['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]
 templates=[];first_stage_bytes=0;shape_members=0
 for row in trace:
  e=js(row['actual_execution_file']);N=len(e['input_ids']);P=front[row['index']]['patches'];members=[];stage_bytes=0
  with zipfile.ZipFile(e['archive']) as z:
   for name in z.namelist():
    assert name.endswith('.npy')
    with z.open(name) as f:
     version=np.lib.format.read_magic(f)
     shape,order,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
    count=int(np.prod(shape))*dtype.itemsize;key=name[:-4];shape_members+=1
    if key.startswith('stage_'):stage_bytes+=count
    else:members.append(dict(key=key,shape=shape,dtype_bytes=dtype.itemsize))
  first_stage_bytes+=stage_bytes;templates.append(dict(N=N,P=P,steps=e['actual_native_forwards'],members=members))
 desired_steps=max(t['steps'] for t in templates)
 def estimate(frontrow,t):
  N=len(frontrow['input_ids']);P=frontrow['patches'];total=sum(int(np.prod(v['shape']))*6 for k,v in frontrow['tensor_metadata'].items() if k.startswith('pixel_values.'))
  for m in t['members']:
   key=m['key'];shape=list(m['shape']);scale=1.
   if key.startswith(('own_pixels_F32_','own_pixels_BF16_')):continue
   elif key=='own_visual_features':shape[0]=P
   elif key=='prefill_inputs_embeds' or key=='input_ids':shape[1]=N
   elif key=='native_cache_prefill':shape[3]=N
   elif key=='native_cache_decode_delta':shape[3]=desired_steps-1
   elif key in ['all_original_full_head_logits','all_head_inputs']:shape[0]=desired_steps
   elif key.startswith('f'):
    import re
    match=re.match(r'f(\d+)_',key)
    if match:
     step=int(match.group(1))-1
     shape=[N+step if v==t['N']+step else N if v==t['N'] else v for v in shape]
     scale=desired_steps/max(1,t['steps'])
   total+=math.ceil(int(np.prod(shape))*m['dtype_bytes']*scale)
  return total
 estimates=[max(estimate(r,t) for t in templates) for r in front]
 total_bytes=sum(estimates)+first_stage_bytes
 need=total_bytes*4+6*1024**3;free=shutil.disk_usage(D).free
 cap=dict(UTC=now(),passed=free>=need,empirical_estimate_bytes=need,actual_free_bytes=free,complete_lossless_ZIP_array_member_shapes_checked=shape_members,all200_original_frontend_shapes_checked=True,per_sample_projected_uncompressed_bytes=estimates,original_factor4_plus6GiB_preserved=True,all_full_head_KV_input_pixels_masks_positions_preserved=True,all_first3_55_layer_values_and_all200_layer_metadata_preserved=True,original_3GiB_hard_1GiB_log_halfGiB_preflight_2GiB_sample_guards_preserved=True,empirical_not_worst_case_guarantee=True,observed_true3_forward_counts=[t['steps'] for t in templates],head_decode_estimate_steps=desired_steps,original_runtime_budget4096_not_lowered=True,no_evidence_deleted=True)
 put('capacity_review.json',cap);assert cap['passed'],'Actual capacity insufficient; preserve all prior evidence and continue another eligible model'
 approval=dict(UTC=now(),approved=True,model_id=parent['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),first200_is_measurement_not_predetermined_5pct_qualification=True,no_model_math_precision_budget_or_current_evidence_deletion=True,small_sample_Strict_MC_Accuracy=score,small_sample_error_to_full200_D_not_an_extra_first200_gate=True)
 put('first200_measurement_approval.json',approval)
 newproto=copy.deepcopy(proto);newproto['static_evidence_SHA256'].update({str(H/'capacity_review.json'):sha(H/'capacity_review.json'),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')})
 pp=R/'configs/protocols/minicpmo26_own_original_BF16_4096_native_v3_approved_measurement.json';create(pp,newproto)
 config=R/'configs/MiniCPMO/openbmb--MiniCPM-o-2_6-own-original-BF16-4096-native-v3-approved-first200-20261116.yaml'
 cfg['name']=config.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(config,cfg)
 job=copy.deepcopy(parent);job.update(stages=[3,200],created_at=now());spec=job['spec'];spec.update(approved=True,config=str(config.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
 spec['source_sha256'].update({str(Path(__file__).relative_to(R)):sha(__file__),str(config.relative_to(R)):sha(config),str(pp.relative_to(R)):sha(pp)})
 validate_config(cfg,parent['model_id']);batch.check_sources(spec);entry=js(D/'state.json')['models'][parent['model_id']]
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,parent['model_id'])
 scope(spec,[3,200],False);unchanged(job['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(spec)
 create(J/'job.json',job);runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('xb') as out,(H/'runner_stderr.log').open('xb') as err:
  c=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200_original_own_BF16_4096',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=c.pid,no_G_or_canonical_write=True))
 rc=c.wait();final=js(J/'completion.json');assert rc==final['actual_returncode']==0
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=final,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),original_evidence_and_G_unchanged=True,disk_alert_is_not_chat_notification=True))
  raise

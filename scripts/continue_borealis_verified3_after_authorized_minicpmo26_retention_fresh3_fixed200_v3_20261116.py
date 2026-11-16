"""Reuse completed Borealis3, authorized verified-only retention, then NEWfresh3/200.
No completed true3 replay, model/source/math change, new retention rule or relaxed
space guard. The exact approved MiniCPMo26 nonexample plan is independently
checked before the existing retention executor is allowed to remove any file.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,copy,subprocess,fcntl,shutil,re,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Borealis4B_authorized_verified_MiniMo26_capacity_recovered_same_completed_true3_fresh3_fixed200_v3_20261116'
RET=B/'authorized_MiniCPMO26_complete_verified200_nonexample_matrix_retention_for_Borealis_20261116';PLAN_SHA='fe13117811ce8c6fb7834bedabd2d224bd55bcb05f2d26fde22c200cb562d29e'
J3=B/'Borealis4B_own_original_BF16_true3_v2_original_prediction_schema_real_FLOCK_20261116';J200=B/'Borealis4B_own_original_BF16_capacity_recovered_approved_fresh3_fixed200_v3_real_FLOCK_20261116';CAP=B/'Borealis4B_original_BF16_complete_true3_capacity_review_v2_original_prediction_schema_20261116/review.json'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def idle():
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
 assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier']]
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (H/'actual_completion.json').exists() and not J200.exists();idle()
 assert sha(RET/'plan.json')==PLAN_SHA;plan=js(RET/'plan.json');assert plan['remove_count']==170 and plan['remove_bytes']==13145347994 and plan['protected_count']==8536 and len(plan['keep'])==30 and len(plan['scopes'])==1
 own=Path('/root/projects/comperror/workspace/diagnostics/model_runs/MiniCPMO26_own_BF16_20261116_052700_205073_1329207').resolve();keep=set(range(10))|set(range(90,100))|set(range(190,200));expected=set(range(200))-keep;indices=set()
 proof=js(plan['scopes'][0]['proof']);assert sha(plan['scopes'][0]['proof'])=='52ad4c86137ee6b45f9ddee54c9a8a94c8481bf00d86254773268f71fbf7415c' and proof['passed'] and proof['sample_count']==200 and proof['original_protocol_final_qualification_complete'] and proof['formal_alignment_within_5pct']
 for rec in plan['remove']:
  p=Path(rec['path']);assert p.parent.resolve()==own and p.resolve().is_relative_to(own) and not p.is_symlink() and re.fullmatch(r'(?:sample_)?\d+\.npz',p.name) and rec['index'] in expected and rec['SHA256']==proof['external_evidence_sha256'][str(p)] and str(p) not in plan['protected_SHA256'];indices.add(rec['index'])
 assert indices==expected and {x['index'] for x in plan['keep']}==keep
 extra={}
 for F in [R/'workspace/reports/nicoboss_MedraN-E4B-Uncensored-EP7/image-text-to-text/20261116_154356',R/'workspace/reports/NbAiLab_borealis-4b-instruct-preview/image-text-to-text/20261116_162651',R/'workspace/reports/NbAiLab_borealis-4b-instruct-preview/image-text-to-text/20261116_163810']:
  assert {p.name for p in F.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
  extra.update({str(p):sha(p) for p in F.iterdir()})
 put('authorized_exact_retention_review.json',dict(passed=True,UTC=now(),plan_SHA256=PLAN_SHA,only_complete_verified_MiniMo26_NPZ170=True,exact_first_middle_last30_kept=True,all8536_protected_plan_files=True,additional15_recent_full_report_files_SHA256=extra,no_weight_data_source_config_log_or_unverified_matrix_in_delete_scope=True))
 executor=R/'scripts/retain_minicpmo26_verified200_nonexample_matrices_for_borealis_20261116.py';assert sha(executor)=='69b9e276c70fc06f797291046b618cd4c4ed6f6b4b229ebcdd7482abd4979455';put('state.json',dict(UTC=now(),PID=os.getpid(),phase='authorized_exact_verified_MiniMo26_nonexample_matrix_retention_before_NPU'))
 with (H/'retention_stdout.json').open('xb') as out,(H/'retention_stderr.log').open('xb') as err:rc=subprocess.run(['python3','-B',str(executor),'apply',PLAN_SHA],cwd=R,env=ENV,stdout=out,stderr=err).returncode
 assert rc==0;ret=js(RET/'actual_completion.json');assert ret['actual_returncode']==0 and ret['all_protected_SHA256_unchanged'] and ret['removed_files']==170 and all(sha(p)==h for p,h in extra.items());idle()
 done=js(J3/'completion.json');assert done['actual_returncode']==0;accepted=done['accepted'][-1];pf=js(accepted['proof']);assert sha(accepted['proof'])==accepted['proof_sha256'] and pf['passed'] and pf['sample_count']==3 and pf['all883_own_source_cast_actual_NPU_SHA256_verified'] and pf['all_full_head_rows_independently_recomputed'] and pf['all_native_KV_values_losslessly_reconstructed']
 oldjob=js(J3/'job.json');batch.check_sources(oldjob['spec']);validate_report(R,accepted['report_dir'],oldjob['model_id'],'image-text-to-text',3,pf)
 cap=js(CAP);assert cap['passed'] and cap['empirical_estimate_bytes']==65933333720 and cap['unchanged_hard_reserve_bytes']==3*1024**3 and cap['unchanged_sample_guard_bytes']==2*1024**3 and shutil.disk_usage(D).free>=cap['empirical_estimate_bytes']
 newcap=dict(cap,UTC=now(),capacity_passed=True,actual_free_bytes=shutil.disk_usage(D).free,parent_original_capacity_review=str(CAP),parent_original_capacity_review_SHA256=sha(CAP),no_completed_true3_replay=True,authorized_retention_receipt=str(RET/'actual_completion.json'),authorized_retention_receipt_SHA256=sha(RET/'actual_completion.json'));put('actual_recovered_capacity_review.json',newcap)
 approval=dict(approved=True,UTC=now(),model_id=oldjob['model_id'],real_NPU3_independent_proof_verified=True,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],numerical_review_verified=True,capacity_review=str(H/'actual_recovered_capacity_review.json'),capacity_review_SHA256=sha(H/'actual_recovered_capacity_review.json'),actual_free_bytes=shutil.disk_usage(D).free,empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],original_hard_log_and_per_sample_guards_preserved=True,no_retention_or_model_math_change=True,first200_is_measurement_not_predetermined_5pct_qualification=True);put('first200_measurement_approval.json',approval)
 cfg=js(R/oldjob['spec']['config']);proto=js(cfg['model']['kwargs']['protocol_file']);parent_runtime={k:v for k,v in proto.items() if k not in ['implementation_source_SHA256','static_evidence_SHA256']};proto['implementation_source_SHA256'][str(Path(__file__).relative_to(R))]=sha(__file__);proto['static_evidence_SHA256'].update({str(RET/'actual_completion.json'):sha(RET/'actual_completion.json'),str(H/'actual_recovered_capacity_review.json'):sha(H/'actual_recovered_capacity_review.json'),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')});assert parent_runtime=={k:v for k,v in proto.items() if k not in ['implementation_source_SHA256','static_evidence_SHA256']}
 pp=R/'configs/protocols/borealis_own_original_BF16_capacity_recovered_approved_first200_v3_20261116.json';create(pp,proto);cf=R/'configs/Gemma3ForConditionalGeneration/NbAiLab--borealis-4b-instruct-preview-own-original-BF16-2048-capacity-recovered-approved-first200-v3-20261116.yaml';cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 new=copy.deepcopy(oldjob);new['stages']=[3,200];new['created_at']=now();new['protected_canonical_hashes']={n:sha(n) for n in oldjob['protected_canonical_hashes']};spec=new['spec'];spec.update(approved=True,config=str(cf.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));spec['source_sha256'].update({n:sha(R/n) for n in [str(Path(__file__).relative_to(R)),str(pp.relative_to(R)),str(cf.relative_to(R))]});scope(spec,[3,200],False);create(J200/'job.json',new)
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';idle();assert batch.device_idle(spec)
 with (H/'approved_fresh3_fixed200_runner.log').open('xb') as out:
  child=subprocess.Popen(['python3','-B',str(runner),str(J200)],cwd=R,env=ENV,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_fixed200_same_verified_true3_no_replay',child_PID=child.pid,job=str(J200),job_SHA256=sha(J200/'job.json')));rc=child.wait()
 done=js(J200/'completion.json');assert rc==done['actual_returncode']==0;put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=done,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise

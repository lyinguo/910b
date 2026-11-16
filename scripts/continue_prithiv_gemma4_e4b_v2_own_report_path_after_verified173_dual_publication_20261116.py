import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,copy,subprocess,time,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX';H=B/'PrithivGemma4E4B_v2_exact_report_path_after_Coder173_dual_sync_recovery_20261116';OLD=B/'PrithivGemma4E4B_own_original_BF16_true3_v1_own_config_real_FLOCK_20261116';J=B/'PrithivGemma4E4B_own_original_BF16_true3_v2_own_config_real_FLOCK_20261116';PUB=B/'CoderGemma4E2B_original_HF_BF16_verified173_blank_G_publication_20261116';DIAG=B/'PrithivGemma4E4B_existing_failed_true3_full_value_readonly_diagnostic_v2_exact_report_path_20261116';BOUND={'scripts/continue_prithiv_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v2_own_config_20261116.py': '5e0235e73c28d5920568702049fee17915fd31d07785ff45dbf880bd4f7e8475', 'scripts/measure_prithiv_gemma4_e4b_own_original_BF16_actual_capacity_v2_own_config_20261116.py': '9387e799be5f8facf9adbce2fcf36944da9c265165d887c657e901fbbf7262a8', 'scripts/verify_prithiv_gemma4_e4b_own_original_BF16_native_v2_exact_own_report_path_20261116.py': '975ec3d59987d55ffb76c5555aaa9fe7fe94b0ead74889f0288a612102407b07'}
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 while True:
  ready=(PUB/'full_target_dual_publication_audit.json').exists() and js(PUB/'full_target_dual_publication_audit.json')['passed'];locks=runner_lock_owners();procs=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
  if ready and not locks and not procs:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_verified173_actual_dual_publication_and_original_lock_release',publication_full_audit_ready=ready,actual_processes=procs,runner_lock=locks,no_new_NPU=True));time.sleep(2)
 assert js(OLD/'completion.json')['actual_returncode']==1 and js(DIAG/'actual_completion.json')['actual_returncode']==0 and sha(DIAG/'diagnostic_full_value_review.json')=='f74ad84baceb721a77cdb78501cab41f2574aa686fb53de45ac4530209e5dd77'
 for n,h in BOUND.items():assert sha(R/n)==h,n
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')['local_workbook_sha256']=='78bb01b2b821e6a2508661aabca551904403130937ff377870f4b52db1196f1d'
 parent=js(OLD/'job.json');cfg=js(R/parent['spec']['config']);oldcfg=copy.deepcopy(cfg);oldproto=Path(cfg['model']['kwargs']['protocol_file']);p=js(oldproto);p.update(reference_workbook_SHA256=sha(book));p['implementation_source_SHA256'].update(BOUND);p['implementation_source_SHA256'][str(Path(__file__).relative_to(R))]=sha(__file__)
 for f in [OLD/'job.json',DIAG/'diagnostic_full_value_review.json',PUB/'publication_receipt.json',PUB/'full_target_dual_publication_audit.json']:p['static_evidence_SHA256'][str(f)]=sha(f)
 pp=R/'configs/protocols/prithiv_gemma4_e4b_own_original_BF16_native_v2_exact_own_report_path_20261116.json';create(pp,p);cf=R/'configs/Gemma4ForConditionalGeneration/prithivMLmods--gemma-4-E4B-it-Uncensored-MAX-own-original-BF16-2048-native-v2-exact-own-report-path-20261116.yaml';cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 assert {k:v for k,v in cfg.items() if k not in ['name','model']}=={k:v for k,v in oldcfg.items() if k not in ['name','model']} and {k:v for k,v in cfg['model'].items() if k!='kwargs'}=={k:v for k,v in oldcfg['model'].items() if k!='kwargs'}
 a=dict(cfg['model']['kwargs']);b=dict(oldcfg['model']['kwargs'])
 for k in ['protocol_file','protocol_sha256']:a.pop(k);b.pop(k)
 assert a==b
 spec=copy.deepcopy(parent['spec']);spec.update(config=str(cf.relative_to(R)),verifier='scripts/verify_prithiv_gemma4_e4b_own_original_BF16_native_v2_exact_own_report_path_20261116.py',review_revision='Prithiv_only_exact_own_report_path_v2_all_original_math_preserved');spec['source_sha256'].update(BOUND);spec['source_sha256'].update({str(Path(__file__).relative_to(R)):sha(__file__),str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});batch.check_sources(spec);entry=js(D/'state.json')['models'][MID]
 for count in [3,200]:_,v=batch.config_for(spec,entry,count);validate_config(v,MID)
 scope(spec,[3],False);job=copy.deepcopy(parent);job.update(spec=spec,created_at=now(),protected_canonical_hashes={n:sha(n) for n in parent['protected_canonical_hashes']});create(J/'job.json',job);put('new_source_binding_review.json',dict(passed=True,UTC=now(),job=str(J),job_SHA256=sha(J/'job.json'),all_original_nonstatic_model_runtime_fields_unchanged=True,only_verifier_own_report_path_and_authorized_Coder173_canonical_binding_updated=True,old_failure_and_diagnostic_only_proof_preserved=True,no_old_job_promoted=True))
 with (H/'controller_stdout.log').open('xb') as out,(H/'controller_stderr.log').open('xb') as err:
  child=subprocess.Popen([sys.executable,'-B',str(R/'scripts/continue_prithiv_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v2_own_config_20261116.py')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_NEW_v2_true3_full_numeric_capacity_then_approved_fresh3_fixed200',child_PID=child.pid));rc=child.wait()
 put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,no_G_or_canonical_write=True));assert rc==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():
   e=dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True);put('NOTIFY_error.json',dict(e,chat_notification_not_claimed=True))
   if not (H/'actual_completion.json').exists():put('actual_completion.json',e)
  raise

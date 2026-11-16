"""Resume already passed Gemma4 true3 after explicit verified-only retention.
Keep every original report, source/config/log, small-sample and original first,
middle, last batch. No completed true3 replay or model/numeric/space-guard edit.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,copy,subprocess,fcntl,shutil,re,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Gemma4E2B_authorized_verified_only_capacity_recovered_same_passed_true3_fresh3_fixed200_v5_20261116'
RET=B/'authorized_eight_complete_numeric_verified200_nonexample_retention_v2_original_batch_examples_for_Gemma4_20261116';PLAN_SHA='97dcc8575fa00a46c3ca4efbf9a0c652e5cf24b886be94dd187fa0c80ce92e52'
J3=B/'Gemma4E2B_own_original_BF16_true3_v4_exact_verifier_binding_real_FLOCK_20261116';J200=B/'Gemma4E2B_own_original_BF16_capacity_recovered_approved_fresh3_fixed200_v5_real_FLOCK_20261116'
CAP=B/'Gemma4E2B_own_original_BF16_complete_true3_capacity_review_v4_exact_verifier_binding_20261116/review.json'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
EXPECTED={'OpenGVLab/VideoChat-Flash-Qwen2-7B_res448': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/VCFlash7B_own_BF16_20261116_072815_638258_1340865', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/VCFlash7B_own_BF16_20261116_072815_638258_1340865/independent_report_verification.json', 'proof_SHA256': '2aa6e901244a9155120a3e0f0a9b33649e461747b3ca22f3b2c779bebcc410a5', 'report_dir': '/root/projects/comperror/workspace/reports/OpenGVLab_VideoChat-Flash-Qwen2-7B_res448/image-text-to-text/20261116_072639'}, 'zai-org/GLM-OCR': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRZai_own_BF16_20261116_093600_043164_1358760', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRZai_own_BF16_20261116_093600_043164_1358760/independent_report_verification.json', 'proof_SHA256': 'fa0fb98eba41aff4b22758aff9a07469ef123d79e7e3b00f639bf17af88d639c', 'report_dir': '/root/projects/comperror/workspace/reports/zai-org_GLM-OCR/image-text-to-text/20261116_093518'}, 'unsloth/GLM-OCR': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRUnsloth_own_BF16_20261116_094349_211852_1362341', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRUnsloth_own_BF16_20261116_094349_211852_1362341/independent_report_verification.json', 'proof_SHA256': '1c6d34c12861a0b3200f557ffff526c5bbfd4d4d25c003b8a6b0bb32113b7885', 'report_dir': '/root/projects/comperror/workspace/reports/unsloth_GLM-OCR/image-text-to-text/20261116_094307'}, 'OS-Copilot/OS-Atlas-Base-4B': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/OSAtlas4B_own_original_MCScore_BF16_20261116_114832_670686_1372084', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/OSAtlas4B_own_original_MCScore_BF16_20261116_114832_670686_1372084/independent_report_verification.json', 'proof_SHA256': '0d0d866149d2fb03058ba30af446c8aa4a1724caf0eeabee69d143d30c8fa84f', 'report_dir': '/root/projects/comperror/workspace/reports/OS-Copilot_OS-Atlas-Base-4B/image-text-to-text/20261116_114723'}, 'NbAiLab/borealis-4b-instruct-preview': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/Borealis4B_own_original_BF16_20261116_165524_141859_1401142', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/Borealis4B_own_original_BF16_20261116_165524_141859_1401142/independent_report_verification.json', 'proof_SHA256': 'dbdf8562c1ed83dec4c4656381e35d3776faf84163d635a5180803c749303c09', 'report_dir': '/root/projects/comperror/workspace/reports/NbAiLab_borealis-4b-instruct-preview/image-text-to-text/20261116_165521'}, 'tiiuae/Falcon-OCR': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/FalconOCR_own_BF16_native512_original_Triton_v2_20261116_130134_591506_1380562', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/FalconOCR_own_BF16_native512_original_Triton_v2_20261116_130134_591506_1380562/independent_report_verification.json', 'proof_SHA256': '48ebec275a96de88906bee2c5aba7ccd39d6b953077b03115529727cec1decde', 'report_dir': '/root/projects/comperror/workspace/reports/tiiuae_Falcon-OCR/image-text-to-text/20261116_130126'}, 'google/pix2struct-base': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/Pix2StructBase_A100_FP16_1527_20261107_235631_167149_741561/native_samples', 'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Pix2StructBase_A100_FP16_formal_v1_1527_20261107/formal200/verification_stdout.json', 'proof_SHA256': 'bcb5be9c4159e823275af195006197dd3c52f7d7f2b15fe5edbb05006242f193', 'report_dir': '/root/projects/comperror/workspace/reports/google_pix2struct-base/image-to-text/20261107_235631'}, 'google/pix2struct-textcaps-large': {'home': '/root/projects/comperror/workspace/diagnostics/model_runs/TextCapsLarge_A100_BF16_1025_20261107_191147_462200_732795/native_samples', 'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/TextCapsLarge_A100_BF16_formal_v1_1025_20261107/formal200/verification_stdout.json', 'proof_SHA256': '05828e76e2b650cbdeb8a3bdb4cfbe9ab3ca415be40d62dd62ab3ff217ba4f14', 'report_dir': '/root/projects/comperror/workspace/reports/google_pix2struct-textcaps-large/image-to-text/20261107_191147'}}
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
 assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier','verifier']]
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (H/'actual_completion.json').exists() and not J200.exists();idle()
 job=js(J3/'job.json');assert sha(J3/'job.json')=='390ddabdd27a6fb475b6487d9f62a47185daa1619787fa58a16a4c7044f08290'
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec'])
 done=js(J3/'completion.json');assert done['actual_returncode']==0;a=done['accepted'][-1];assert a['sample_count']==3 and a['proof_sha256']=='f4bb24ca1bce35189e490f165611a9ae49f5334319008a5feb4648aa3e319f68';pf=js(a['proof'])
 assert sha(a['proof'])==a['proof_sha256'] and pf['passed'] and pf['sample_count']==3 and pf['all2011_own_source_actual_NPU_and_unused_SHA256_verified'] and pf['all60_unused_shared_KV_source_preserved'] and pf['all_full_head_rows_independently_recomputed'] and pf['all_native_KV_values_losslessly_reconstructed']
 cfg=js(R/job['spec']['config']);assert cfg['task_type']=='any-to-any';validate_report(R,a['report_dir'],job['model_id'],cfg['task_type'],3,pf)
 assert sha(RET/'plan.json')==PLAN_SHA;plan=js(RET/'plan.json');assert plan['remove_count']==1356 and plan['protected_count']==10494 and len(plan['keep'])==244 and len(plan['scopes'])==8
 assert {s['model_id'] for s in plan['scopes']}==set(EXPECTED)
 assert plan['remove_bytes']==sum(x['bytes'] for x in plan['remove']) and plan['remove_bytes']+shutil.disk_usage(D).free>=79630599752
 base=set(range(10))|set(range(90,100))|set(range(190,200));by_model={}
 for rec in plan['remove']:
  mid=rec['model_id'];e=EXPECTED[mid];home=Path(e['home']).resolve();p=Path(rec['path']);keep=base|(set(range(96,104)) if mid=='tiiuae/Falcon-OCR' else set())
  assert home.is_relative_to(R/'workspace/diagnostics/model_runs') and p.parent.resolve()==home and p.resolve().is_relative_to(home) and not p.is_symlink() and re.fullmatch(r'(?:sample_)?\d+\.npz',p.name)
  assert rec['index'] not in keep and str(p) not in plan['protected_SHA256'] and rec['proof']==e['proof'] and rec['proof_SHA256']==e['proof_SHA256']
  by_model.setdefault(mid,set()).add(rec['index'])
 for mid,e in EXPECTED.items():
  proof=js(e['proof']);assert sha(e['proof'])==e['proof_SHA256'] and proof['passed'] and proof['sample_count']==200 and not proof.get('error_indices')
  assert all(sha(Path(e['report_dir'])/n)==h for n,h in proof['report_sha256'].items())
  keep=base|(set(range(96,104)) if mid=='tiiuae/Falcon-OCR' else set());assert by_model[mid]==set(range(200))-keep and {x['index'] for x in plan['keep'] if x['model_id']==mid}==keep
  assert all(x['SHA256']==proof['external_evidence_sha256'][x['path']] for x in plan['remove'] if x['model_id']==mid)
 put('authorized_exact_retention_independent_review.json',dict(passed=True,UTC=now(),plan_SHA256=PLAN_SHA,remove_count=1356,keep_count=244,protected_files=10494,all_eight_full_numeric200_proofs_and_original_full_reports_verified=True,original_Falcon_batch8_middle96_to103_protected=True,all_original_alignment_qualification_flags_and_G_unchanged=True,no_unverified_small_AA_fresh3_failed160_arrays_in_delete_scope=True))
 executor=R/'scripts/retain_eight_numeric_verified200_nonexample_v2_original_batch_examples_for_gemma4_20261116.py';assert sha(executor)=='db6c646d3e8860ac7f5f7da50c781784af1c457d6046c35d525980d796b5e9f3'
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='authorized_exact_completed_numeric200_nonexample_retention_before_NEWfresh3'))
 with (H/'retention_stdout.json').open('xb') as out,(H/'retention_stderr.log').open('xb') as err:rc=subprocess.run(['python3','-B',str(executor),'apply',PLAN_SHA],cwd=R,env=ENV,stdout=out,stderr=err).returncode
 assert rc==0;ret=js(RET/'actual_completion.json');assert ret['actual_returncode']==0 and ret['removed_files']==1356 and ret['all_protected_SHA256_unchanged'];idle()
 changed={n:dict(before=h,after=sha(n)) for n,h in job['protected_canonical_hashes'].items() if sha(n)!=h};assert set(changed).issubset({str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')})
 put('canonical_unchanged_except_authorized_retention_ledger_review.json',dict(passed=True,UTC=now(),changes=changed,no_G_state_registry_workbook_change=True))
 cap=js(CAP);assert cap['passed'] and cap['empirical_estimate_bytes']==79630599752 and cap['unchanged_hard_reserve_bytes']==3*1024**3 and cap['unchanged_log_reserve_bytes']==1024**3 and cap['unchanged_sample_guard_bytes']==2*1024**3
 assert shutil.disk_usage(D).free>=cap['empirical_estimate_bytes']
 newcap=dict(cap,UTC=now(),capacity_passed=True,actual_free_bytes=shutil.disk_usage(D).free,parent_original_capacity_review=str(CAP),parent_original_capacity_review_SHA256=sha(CAP),no_completed_true3_replay=True,authorized_retention_receipt=str(RET/'actual_completion.json'),authorized_retention_receipt_SHA256=sha(RET/'actual_completion.json'));put('actual_recovered_capacity_review.json',newcap)
 approval=dict(approved=True,UTC=now(),model_id=job['model_id'],real_NPU3_independent_proof_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],numerical_review_verified=True,capacity_review=str(H/'actual_recovered_capacity_review.json'),capacity_review_SHA256=sha(H/'actual_recovered_capacity_review.json'),actual_free_bytes=shutil.disk_usage(D).free,empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],empirical_not_worst_case_guarantee=True,original_hard_log_and_per_sample_guards_preserved=True,first200_is_measurement_not_predetermined_5pct_qualification=True,no_model_math_dtype_budget_retention_rule_or_report_content_change=True);put('first200_measurement_approval.json',approval)
 proto=js(cfg['model']['kwargs']['protocol_file']);runtime={k:v for k,v in proto.items() if k not in ['implementation_source_SHA256','static_evidence_SHA256']};proto['implementation_source_SHA256'][str(Path(__file__).relative_to(R))]=sha(__file__)
 proto['static_evidence_SHA256'].update({str(H/n):sha(H/n) for n in ['authorized_exact_retention_independent_review.json','actual_recovered_capacity_review.json','first200_measurement_approval.json']});proto['static_evidence_SHA256'][str(RET/'actual_completion.json')]=sha(RET/'actual_completion.json');assert runtime=={k:v for k,v in proto.items() if k not in ['implementation_source_SHA256','static_evidence_SHA256']}
 pp=R/'configs/protocols/gemma4_e2b_own_original_BF16_capacity_recovered_approved_first200_v5_20261116.json';create(pp,proto)
 cf=R/'configs/Gemma4ForConditionalGeneration/google--gemma-4-E2B-it-own-original-BF16-2048-capacity-recovered-approved-first200-v5-20261116.yaml';cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 new=copy.deepcopy(job);new['created_at']=now();new['stages']=[3,200];new['protected_canonical_hashes']={n:sha(n) for n in job['protected_canonical_hashes']};spec=new['spec'];spec.update(approved=True,config=str(cf.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));spec['source_sha256'].update({n:sha(R/n) for n in [str(Path(__file__).relative_to(R)),str(pp.relative_to(R)),str(cf.relative_to(R))]});batch.check_sources(spec);scope(spec,[3,200],False);create(J200/'job.json',new)
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90';idle();assert batch.device_idle(spec)
 with (H/'approved_NEWfresh3_fixed200_runner.log').open('xb') as out:
  child=subprocess.Popen(['python3','-B',str(runner),str(J200)],cwd=R,env=ENV,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_fixed200_same_verified_true3_no_replay',child_PID=child.pid,job=str(J200),job_SHA256=sha(J200/'job.json')));rc=child.wait()
 done=js(J200/'completion.json');assert rc==done['actual_returncode']==0;put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=done,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise

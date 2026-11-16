"""Explicit retention consent: only nonexample NPZ from eight complete numerically verified200 runs (six aligned and two measured outside tolerance).
All complete reports, per-sample records, source/config/snapshots/logs, original
first/middle/last groups, every small-sample/AA/fresh3 run and failures remain.
This never edits an existing proof to imply pruned arrays remain available.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,shutil,re
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'authorized_eight_complete_numeric_verified200_nonexample_retention_v2_original_batch_examples_for_Gemma4_20261116';POL=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
SCOPES=[{'model_id': 'OpenGVLab/VideoChat-Flash-Qwen2-7B_res448', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/VCFlash7B_own_BF16_20261116_072815_638258_1340865/independent_report_verification.json', 'proof_SHA256': '2aa6e901244a9155120a3e0f0a9b33649e461747b3ca22f3b2c779bebcc410a5', 'report_dir': '/root/projects/comperror/workspace/reports/OpenGVLab_VideoChat-Flash-Qwen2-7B_res448/image-text-to-text/20261116_072639', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/VCFlash7B_own_BF16_20261116_072815_638258_1340865'}, {'model_id': 'zai-org/GLM-OCR', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRZai_own_BF16_20261116_093600_043164_1358760/independent_report_verification.json', 'proof_SHA256': 'fa0fb98eba41aff4b22758aff9a07469ef123d79e7e3b00f639bf17af88d639c', 'report_dir': '/root/projects/comperror/workspace/reports/zai-org_GLM-OCR/image-text-to-text/20261116_093518', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRZai_own_BF16_20261116_093600_043164_1358760'}, {'model_id': 'unsloth/GLM-OCR', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRUnsloth_own_BF16_20261116_094349_211852_1362341/independent_report_verification.json', 'proof_SHA256': '1c6d34c12861a0b3200f557ffff526c5bbfd4d4d25c003b8a6b0bb32113b7885', 'report_dir': '/root/projects/comperror/workspace/reports/unsloth_GLM-OCR/image-text-to-text/20261116_094307', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/GLMOCRUnsloth_own_BF16_20261116_094349_211852_1362341'}, {'model_id': 'OS-Copilot/OS-Atlas-Base-4B', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/OSAtlas4B_own_original_MCScore_BF16_20261116_114832_670686_1372084/independent_report_verification.json', 'proof_SHA256': '0d0d866149d2fb03058ba30af446c8aa4a1724caf0eeabee69d143d30c8fa84f', 'report_dir': '/root/projects/comperror/workspace/reports/OS-Copilot_OS-Atlas-Base-4B/image-text-to-text/20261116_114723', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/OSAtlas4B_own_original_MCScore_BF16_20261116_114832_670686_1372084'}, {'model_id': 'NbAiLab/borealis-4b-instruct-preview', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/Borealis4B_own_original_BF16_20261116_165524_141859_1401142/independent_report_verification.json', 'proof_SHA256': 'dbdf8562c1ed83dec4c4656381e35d3776faf84163d635a5180803c749303c09', 'report_dir': '/root/projects/comperror/workspace/reports/NbAiLab_borealis-4b-instruct-preview/image-text-to-text/20261116_165521', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/Borealis4B_own_original_BF16_20261116_165524_141859_1401142'}, {'model_id': 'tiiuae/Falcon-OCR', 'proof': '/root/projects/comperror/workspace/diagnostics/model_runs/FalconOCR_own_BF16_native512_original_Triton_v2_20261116_130134_591506_1380562/independent_report_verification.json', 'proof_SHA256': '48ebec275a96de88906bee2c5aba7ccd39d6b953077b03115529727cec1decde', 'report_dir': '/root/projects/comperror/workspace/reports/tiiuae_Falcon-OCR/image-text-to-text/20261116_130126', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/FalconOCR_own_BF16_native512_original_Triton_v2_20261116_130134_591506_1380562'}, {'model_id': 'google/pix2struct-base', 'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Pix2StructBase_A100_FP16_formal_v1_1527_20261107/formal200/verification_stdout.json', 'proof_SHA256': 'bcb5be9c4159e823275af195006197dd3c52f7d7f2b15fe5edbb05006242f193', 'report_dir': '/root/projects/comperror/workspace/reports/google_pix2struct-base/image-to-text/20261107_235631', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/Pix2StructBase_A100_FP16_1527_20261107_235631_167149_741561/native_samples'}, {'model_id': 'google/pix2struct-textcaps-large', 'proof': '/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/TextCapsLarge_A100_BF16_formal_v1_1025_20261107/formal200/verification_stdout.json', 'proof_SHA256': '05828e76e2b650cbdeb8a3bdb4cfbe9ab3ca415be40d62dd62ab3ff217ba4f14', 'report_dir': '/root/projects/comperror/workspace/reports/google_pix2struct-textcaps-large/image-to-text/20261107_191147', 'home': '/root/projects/comperror/workspace/diagnostics/model_runs/TextCapsLarge_A100_BF16_1025_20261107_191147_462200_732795/native_samples'}]
FIVE=['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'];KEEP=set(range(10))|set(range(90,100))|set(range(190,200))
def now():return datetime.now(timezone.utc).isoformat()
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(name,v):
 p=H/name;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def idle():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier','verifier']]
def main():
 idle();assert sha(POL)=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7';policy=js(POL);assert policy['explicit_user_consent'] and policy['permanently_preserve_full_five_and_all_records_source_YAML_logs'];assert sha(policy['authorization'])==policy['authorization_SHA256']=='06eba6a16df1559793352231cc9527955f5f13d58eb5e96929a110695afa9ee3'
 if sys.argv[1]=='plan':
  H.mkdir(exist_ok=False);removed=[];kept=[];proofs=[];protected=set();scope_rows=[]
  critical={D/n for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']};critical.add(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'));protected.update(critical)
  for mid,e in js(D/'state.json')['models'].items():
   if e.get('status')=='done':protected.update(Path(e['report_dir'])/n for n in FIVE)
  for scope in SCOPES:
   home=Path(scope['home']).resolve();assert home.is_relative_to(R/'workspace/diagnostics/model_runs') and not home.is_symlink()
   pf=Path(scope['proof']);assert sha(pf)==scope['proof_SHA256'];p=js(pf);assert p['passed'] and p['sample_count']==200 and not p.get('error_indices');protected.add(pf);proofs.append(dict(path=str(pf),SHA256=sha(pf)))
   assert set(p['report_sha256'])==set(FIVE)
   for n,h in p['report_sha256'].items():assert sha(Path(scope['report_dir'])/n)==h;protected.add(Path(scope['report_dir'])/n)
   actual=[]
   for name,h in p['external_evidence_sha256'].items():
    f=Path(name)
    if f.suffix!='.npz' or f.parent.resolve()!=home:continue
    m=re.fullmatch(r'(?:sample_)?(\d+)\.npz',f.name);assert m,name;i=int(m.group(1));actual.append(i);assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(home)
    rec=dict(path=str(f),SHA256=h,bytes=f.stat().st_size,index=i,home=str(home),model_id=scope['model_id'],proof=str(pf),proof_SHA256=sha(pf))
    if i in KEEP or (scope['model_id']=='tiiuae/Falcon-OCR' and 96<=i<=103):kept.append(rec);protected.add(f)
    else:removed.append(rec)
   assert sorted(actual)==list(range(200));scope_rows.append(dict(scope,keep_indices=sorted(KEEP | (set(range(96,104)) if scope["model_id"]=="tiiuae/Falcon-OCR" else set())),original_full200_independent_verification_unchanged=True))
  candidates={x['path'] for x in removed};assert len(candidates)==8*170-4
  for scope in SCOPES:
   # Protect parent metadata as well as all retained values in nested native_samples.
   h=Path(scope['home']);root=R/'workspace/diagnostics/model_runs'/h.relative_to(R/'workspace/diagnostics/model_runs').parts[0]
   protected.update(p for p in root.rglob('*') if p.is_file() and str(p) not in candidates)
  # The latest passed Gemma4 true3 and earlier failed attempt are outside deletion scopes.
  for root in [R/'workspace/reports/google_gemma-4-E2B-it/any-to-any/20261116_174006',R/'workspace/reports/google_gemma-4-E2B-it/any-to-any/20261116_173112',R/'workspace/diagnostics/model_runs/Gemma4E2B_own_original_BF16_20261116_174009_260199_1404478']:
   assert root.is_dir();protected.update(p for p in root.rglob('*') if p.is_file())
  # Only primary code trees, not a workspace artifact discovery/upload scan.
  for folder in [R/'universal_eval',R/'scripts',R/'configs']:protected.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.ir'])
  # Full reports and source snapshots/logs in selected homes stay byte-exact.
  owners={s['home']:s['model_id'] for s in SCOPES}
  for mid,e in js(D/'state.json')['models'].items():
   if e.get('status')!='done':continue
   docs=[Path(e['report_dir'])/'eval_report.json',Path(e['report_dir'])/'operator_lineage.json'];cfg=e.get('active_config')
   if cfg and Path(cfg).is_file():docs.append(Path(cfg))
   for doc in docs:
    text=doc.read_text()
    for home,owner in owners.items():
     if owner!=mid and home in text:
      assert not any(x['path'] in text for x in removed if x['home']==home),('Other active model directly needs a candidate',mid,str(doc))
  assert not candidates.intersection(map(str,protected));before={str(p):sha(p) for p in sorted(protected)};ledger=LED.read_bytes();(H/'ledger_before_original_bytes.json').write_bytes(ledger)
  plan=dict(UTC=now(),authorization=policy['authorization'],authorization_SHA256=policy['authorization_SHA256'],policy_SHA256=sha(POL),ledger_before_SHA256=hashlib.sha256(ledger).hexdigest(),scopes=scope_rows,remove=removed,keep=kept,remove_count=len(removed),remove_bytes=sum(x['bytes'] for x in removed),protected_SHA256=before,protected_count=len(before),critical_shared_paths=sorted(map(str,critical)),free_before_bytes=shutil.disk_usage(D).free,no_unverified_failed160_small_AA_fresh3_arrays_in_scope=True,no_original_alignment_or_qualification_flags_changed=True,two_outside_tolerance_measurements_remain_unpublished=True,all_full_reports_all_records_source_YAML_snapshots_logs_preserved=True)
  put('plan.json',plan);print(json.dumps(dict(plan=str(H/'plan.json'),plan_SHA256=sha(H/'plan.json'),remove_count=len(removed),logical_GiB=plan['remove_bytes']/2**30,protected_files=len(before),retained_original_first_middle_last_matrices=len(kept))))
 else:
  assert sys.argv[1]=='apply' and sha(H/'plan.json')==sys.argv[2];plan=js(H/'plan.json');assert sha(LED)==plan['ledger_before_SHA256'];idle()
  with (D/'runner.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for p,h in plan['protected_SHA256'].items():assert sha(p)==h,p
   for i,x in enumerate(plan['remove']):
    f=Path(x['path']);home=Path(x['home']).resolve();assert home.is_relative_to(R/'workspace/diagnostics/model_runs') and f.parent.resolve()==home and f.resolve().is_relative_to(home) and not f.is_symlink() and f.suffix=='.npz' and f.stat().st_size==x['bytes'] and sha(f)==x['SHA256'],str(f)
    if i%25==0:put('progress.json',dict(UTC=now(),phase='full_SHA_before_any_removal',checked=i+1,total=plan['remove_count']))
   manifest=H/'removed_exact_original_paths_SHA256.jsonl';assert not manifest.exists()
   with manifest.open('x',encoding='utf-8') as out:
    for i,x in enumerate(plan['remove']):
     f=Path(x['path']);assert f.parent.resolve()==Path(x['home']).resolve() and f.stat().st_size==x['bytes'];f.unlink();out.write(json.dumps(x,ensure_ascii=False)+'\n');out.flush()
     if i%25==0:put('progress.json',dict(UTC=now(),phase='removing_exact_preverified_nonexample_NPZ_only',removed=i+1,total=plan['remove_count']))
   assert all(not Path(x['path']).exists() for x in plan['remove'])
   for p,h in plan['protected_SHA256'].items():assert sha(p)==h,p
   assert sha(LED)==plan['ledger_before_SHA256'];actual=dict(UTC=now(),actual_returncode=0,plan_SHA256=sys.argv[2],removed_files=plan['remove_count'],removed_bytes=plan['remove_bytes'],removed_manifest=str(manifest),removed_manifest_SHA256=sha(manifest),protected_files=plan['protected_count'],all_protected_SHA256_unchanged=True,all_full_five_reports_input_output_reference_error_token_source_YAML_snapshots_logs_and_small_examples_preserved=True,retained_original_first_middle_last_complete_values=len(plan['keep']),free_before_bytes=plan['free_before_bytes'],free_after_bytes=shutil.disk_usage(D).free,old_proofs_unchanged_and_pruned_arrays_explicitly_no_longer_available=True,no_NN_or_canonical_score_write=True);put('actual_completion.json',actual)
   ledger=js(LED);ledger['events'].append(dict(UTC=actual['UTC'],authorization=plan['authorization'],authorization_SHA256=plan['authorization_SHA256'],receipt=str(H/'actual_completion.json'),receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=actual['removed_files'],removed_bytes=actual['removed_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True));tmp=LED.with_name(LED.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,LED);print(json.dumps(dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED))))
if __name__=='__main__':main()

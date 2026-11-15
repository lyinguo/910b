"""Apply explicit user retention consent to two fully independently verified runs only.

Permanent reports, all records/source/YAML/logs and original first/middle/last
groups are immutable. Old proofs remain unchanged; removed arrays are declared
unavailable by exact path/SHA in the append-only retention ledger.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,fcntl,os,re,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'authorized_four_verified_runs_nonexample_retention_v3_1535_20261115';POL=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
FIVE=['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']
SCOPES=[dict(home=R/'workspace/diagnostics/model_runs/Gemma3nE4B_actual_original_Greedy_HighTen_BF16_20261114_005219_865961_1188034',proof_SHA256='a4a8d5c7b9792c3dbadf30e6bfc1274ba2ebd1133735413c4e7e264378719215',pattern=r'sample_(\d{5})\.npz',keep=set(range(10))|set(range(90,110))|set(range(190,200)),runner_deadline_exception_preserved=True),dict(home=R/'workspace/diagnostics/model_runs/FigCodifier_original_manual_910b_20261102_125404_286460_399684',proof_SHA256='0696dd91dd7b1a38f00e6a878d7570b7af6957de6317ee7f2c83e8d2ec5afe97',pattern=r'samples/(\d{3})\.npz',keep=set(range(10))|set(range(90,100))|set(range(190,200)),runner_deadline_exception_preserved=False)]
def now():return datetime.now(timezone.utc).isoformat()
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def idle():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners();assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier']]
def protect_paths():
 paths={D/n for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']};paths.add(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'))
 state=js(D/'state.json')
 for mid,e in state['models'].items():
  if e['status']=='done':paths.update(Path(e['report_dir'])/n for n in FIVE)
 for s in SCOPES:
  proof=js(s['home']/'independent_report_verification.json');paths.update(Path(proof['report_dir'])/n for n in FIVE)
 for name in ['20261115_161859','20261115_162701']:
  p=R/'workspace/reports/OpenGVLab_InternVL2-2B/image-text-to-text'/name;paths.update(p/n for n in FIVE)
 for d in [R/'universal_eval',R/'scripts',R/'configs']:
  paths.update(p for p in d.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json'])
 return paths
def main():
 idle();assert sha(POL)=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7';policy=js(POL);assert policy['explicit_user_consent'] and policy['permanently_preserve_full_five_and_all_records_source_YAML_logs'];assert sha(policy['authorization'])==policy['authorization_SHA256']=='06eba6a16df1559793352231cc9527955f5f13d58eb5e96929a110695afa9ee3'
 if sys.argv[1]=='--legacy-plan-disabled-use-reviewed-extension':
  H.mkdir(exist_ok=False);removed=[];protected=protect_paths();scopes=[]
  for s in SCOPES:
   home=s['home'].resolve();pf=home/'independent_report_verification.json';assert sha(pf)==s['proof_SHA256'];proof=js(pf);assert proof['passed'] and proof['sample_count']==200 and not proof['error_indices'];actual=[]
   for name,h in proof['external_evidence_sha256'].items():
    p=Path(name)
    if p.suffix!='.npz' or not p.is_relative_to(home):continue
    match=re.fullmatch(s['pattern'],str(p.relative_to(home)))
    if not match:continue
    i=int(match.group(1));assert 0<=i<200 and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(home);actual.append(i)
    if i not in s['keep']:removed.append(dict(path=str(p),SHA256=h,bytes=p.stat().st_size,index=i,home=str(home),proof=str(pf),proof_SHA256=s['proof_SHA256']))
   assert sorted(actual)==list(range(200));scopes.append(dict(home=str(home),keep_indices=sorted(s['keep']),all_200_complete_values_previously_independently_verified=True,proof=str(pf),proof_SHA256=s['proof_SHA256'],runner_deadline_exception_preserved=s['runner_deadline_exception_preserved']))
  candidate={v['path'] for v in removed}
  # Check direct references in every other active model's report/config and
  # protocol. These sources and all full reports remain protected regardless.
  crossrefs=set()
  for mid,e in js(D/'state.json')['models'].items():
   if e['status']!='done' or mid=='MathLLMs/FigCodifier':continue
   folder=Path(e['report_dir']);docs=[folder/'eval_report.json',folder/'operator_lineage.json'];cfg=Path(e['active_config']) if e.get('active_config') else None
   if cfg and cfg.is_file():docs.append(cfg)
   for p in docs:
    text=p.read_text(errors='strict')
    for n in candidate:
     if n in text:crossrefs.add(n)
    if p.name=='operator_lineage.json':
     line=js(p).get('actual_execution',{});pf=line.get('protocol_file')
     if pf and Path(pf).is_file():
      text=Path(pf).read_text()
      for n in candidate:
       if n in text:crossrefs.add(n)
  assert not crossrefs,sorted(crossrefs)
  for s in SCOPES:protected.update(p for p in s['home'].rglob('*') if p.is_file() and str(p) not in candidate)
  assert all(Path(n).suffix=='.npz' for n in candidate) and not candidate.intersection(map(str,protected));before={str(p):sha(p) for p in sorted(protected)}
  ledger_raw=LED.read_bytes();(H/'retention_ledger_before_original_bytes.json').write_bytes(ledger_raw)
  result=dict(UTC=now(),authorization=policy['authorization'],authorization_SHA256=policy['authorization_SHA256'],policy_SHA256=sha(POL),ledger_before_SHA256=hashlib.sha256(ledger_raw).hexdigest(),scopes=scopes,remove=removed,remove_count=len(removed),remove_bytes=sum(x['bytes'] for x in removed),protected_SHA256=before,protected_count=len(before),free_before_bytes=shutil.disk_usage(D).free,only_two_exact_verified_run_scopes=True,all_reports_records_code_YAML_logs_and_small_examples_kept=True,no_weights_data_workbooks_or_unverified_failed160_values_in_scope=True)
  put('plan.json',result);print(json.dumps(dict(plan=str(H/'plan.json'),SHA256=sha(H/'plan.json'),count=result['remove_count'],logical_GiB=result['remove_bytes']/1024**3,protected=result['protected_count'],free_GiB=result['free_before_bytes']/1024**3)))
 else:
  assert sys.argv[1]=='--apply-sha' and sha(H/'plan.json')==sys.argv[2];plan=js(H/'plan.json');assert sha(LED)==plan['ledger_before_SHA256'];idle();lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for p in plan['critical_shared_paths']:assert sha(p)==plan['protected_SHA256'][p],p
  # Validate every candidate's original complete-file SHA before any removal.
  for i,x in enumerate(plan['remove']):
   p=Path(x['path']);home=Path(x['home']).resolve();assert p.resolve().is_relative_to(home) and not p.is_symlink() and p.suffix=='.npz';assert p.stat().st_size==x['bytes'] and sha(p)==x['SHA256'],str(p)
   if i%20==0:put('progress.json',dict(UTC=now(),phase='verifying_all_original_candidate_SHA_before_any_delete',verified=i+1,target=plan['remove_count']))
  manifest=H/'removed_original_paths_SHA256.jsonl';assert not manifest.exists()
  with manifest.open('x',encoding='utf-8') as f:
   for i,x in enumerate(plan['remove']):
    p=Path(x['path']);assert p.stat().st_size==x['bytes'];p.unlink();f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
    if i%20==0:put('progress.json',dict(UTC=now(),phase='removing_only_preverified_exact_matrix_paths',removed=i+1,target=plan['remove_count']))
  for p,h in plan['protected_SHA256'].items():assert sha(p)==h,p
  assert all(not Path(x['path']).exists() for x in plan['remove']);assert sha(LED)==plan['ledger_before_SHA256'];receipt=dict(UTC=now(),actual_returncode=0,plan_SHA256=sys.argv[2],removed_files=plan['remove_count'],removed_bytes=plan['remove_bytes'],removed_manifest=str(manifest),removed_manifest_SHA256=sha(manifest),protected_files=plan['protected_count'],all_protected_SHA256_unchanged=True,all_full_reports_records_source_YAML_snapshots_logs_and_small_first_middle_last_kept=True,free_before_bytes=plan['free_before_bytes'],free_after_bytes=shutil.disk_usage(D).free,original_proofs_unchanged_and_removed_arrays_explicitly_unavailable=True,no_NN_or_canonical_score_write=True);put('actual_completion.json',receipt)
  ledger=js(LED);ledger['events'].append(dict(UTC=receipt['UTC'],authorization=plan['authorization'],authorization_SHA256=plan['authorization_SHA256'],receipt=str(H/'actual_completion.json'),receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=plan['remove_count'],removed_bytes=plan['remove_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True));tmp=LED.with_name(LED.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,LED);print(json.dumps(dict(receipt=receipt,receipt_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED)),ensure_ascii=False))
if __name__=='__main__':main()

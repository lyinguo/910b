import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timezone
from collections import Counter
import os,copy,json,hashlib,fcntl,shutil,base64,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'OSAtlas4B_original_probability_verified170_blank_G_publication_20261116'
Q=json.loads(Path(sys.argv[1]).read_bytes());TEST_SOURCE_BASE64=base64.b64encode((R/Q['test_template']).read_bytes()).decode()
sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import atomic_json,validate_report,workbook_rows
from A100_alignment_evidence_runner_20261103 import unchanged
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
from local_A100_reference_gate_20261103 import check_local_A100_reference
from verified_A100_G_cell_writer_20261103 import write_verified_A100_G
import batch_910b as batch
def read(p):return json.loads(Path(p).read_bytes())
cache={}
def sha(p):
 p=Path(p);s=p.stat();k=(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if k not in cache:
  with p.open('rb') as f:cache[k]=hashlib.file_digest(f,'sha256').hexdigest()
 return cache[k]
def now():return datetime.now(timezone.utc).isoformat()
roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
assert not H.exists() and not runner_lock_owners() and not [p for p in project_processes() if p['role'] in roles]
policy=read(D/'A100_ALIGNMENT_PUBLICATION_POLICY.json');decision=Path(policy['authorization'])
assert sha(decision)==policy['authorization_sha256']=='b47ae739b279138134f87978bf4a71b3bbcecff149dcdbfc0b26e49ab04234f1'
assert sha(R/'scripts/verified_A100_G_cell_writer_20261103.py')=='83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8'
book=Path(read(D/'POLICY.json')['workbook']);sf=D/'state.json';rf=D/'registry.json';binding=D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
with (D/'runner.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
 assert not [p for p in project_processes() if p['role'] in roles]
 assert sha(book)==Q['expected_before_sha256']=='4c924c83f43ab75dd5fc0e5903e77c072042db1c0383f09a25a37877754bed60'
 state=read(sf);registry=read(rf);old_state=copy.deepcopy(state);old_reg=copy.deepcopy(registry)
 assert state['current'] is None and read(D/'PAUSE')['owner']=='A100_relative5pct_alignment_20261103_explicit_user_resume'
 rows,_,before=workbook_rows(book);assert len(rows)==700 and sum(r['score'] is not None for r in rows)==169
 candidates=[];all_external={}
 # BOTH jobs pin the same valid pre-publication canonical state. Validate both before any authorized G mutation.
 for c in Q['changes']:
  mid=c['model_id'];expected=Q['models'][mid];J=Path(expected['job']);job=read(J/'job.json');spec=copy.deepcopy(job['spec'])
  assert sha(J/'job.json')==expected['job_SHA256'] and job['stages']==[3,200] and spec['approved'] and not job['write_workbook']
  unchanged(job['protected_canonical_hashes']);batch.check_sources(spec)
  a=read(spec['alignment_approval_review']);assert sha(spec['alignment_approval_review'])==spec['alignment_approval_review_sha256']
  assert a['approved'] and a['real_NPU3_independent_proof_verified'] and a['numerical_review_verified'] and a['first200_is_measurement_not_predetermined_5pct_qualification']
  assert sha(a['true3_proof'])==a['true3_proof_SHA256'] and sha(a['capacity_review'])==a['capacity_review_SHA256'] and read(a['capacity_review'])['passed']
  t=read(a['true3_proof']);assert t['passed'] and t['sample_count']==3 and t['actual_NPU_inference'] and not t['historical_record_reuse']
  assert t['all_own_source_actual_NPU_SHA256_verified'] and t['original_cache_false_no_KV_returned'] and t['all_original_probability_scores_and_full_head_verified'] and validate_report(R,t['report_dir'],mid,c['task'],3,t)==expected['true3_score'] and sha(a['true3_proof'])==expected['true3_proof_SHA256']
  done=read(J/'completion.json');sup=read(Path(expected['controller'])/'actual_completion.json')
  assert done['actual_returncode']==done['model_returncode']==done['verifier_returncode']==sup['actual_returncode']==0
  assert [x['sample_count'] for x in done['accepted']]==[3,200]
  run=done['accepted'][-1];assert sha(run['proof'])==run['proof_sha256']==c['formal_proof_sha256'];proof=read(run['proof'])
  assert proof['passed'] and proof['actual_NPU_inference'] and not proof['historical_record_reuse'] and proof['formal_alignment_within_5pct']
  assert proof['original_protocol_final_qualification_complete'] and not proof['A100_historical_engine_or_tensor_capture_claimed'] and not proof['error_indices']
  assert proof['all_own_source_actual_NPU_SHA256_verified'] and proof['source_tensor_count']==541 and proof['original_cache_false_no_KV_returned'] and proof['original_generation_invocations']==0 and proof['all_original_probability_scores_and_full_head_verified']
  assert proof['actual_forward_count']==expected['expected_forward_count'] and len(proof['numeric_sample_checks'])==200 and proof['NPU_profiler_kernel_events']==expected['expected_kernels']
  assert proof['original_cache_false_no_KV_returned'] and proof['all_full_inputs_tokens_raw_references_errors_preserved']
  assert proof['truncated_indices']==expected['expected_truncated'] and proof['empty_indices']==expected['expected_empty']
  assert proof['invalid_indices']==expected['expected_invalid']
  score=validate_report(R,run['report_dir'],mid,c['task'],200,proof);assert Decimal(str(score))==Decimal(str(c['new_value']))
  assert all(Decimal(v)<=Decimal('.05') for v in proof['relative_error_to_current_A100_D'].values())
  gate=check_local_A100_reference(mid,c['task'],score);err=abs(Decimal(c['A100_full_precision_cached_D'])-Decimal(str(score)))/abs(Decimal(c['A100_full_precision_cached_D']));assert err<=Decimal('.05')
  gate['current_user_workbook_D_additional_gate']={'D':c['A100_full_precision_cached_D'],'relative_error':str(err),'passed':True,'original_full_precision_reference_gate_also_retained':True}
  for p in [t]+[read(x['proof']) for x in done['accepted']]:
   validate_report(R,p['report_dir'],mid,c['task'],p['sample_count'],p)
   for path,h in p['external_evidence_sha256'].items():
    assert path not in all_external or all_external[path]==h;all_external[path]=h
  row,=[r for r in rows if r['model_id']==mid];assert row['row']==c['row'] and row['task']==c['task'] and row['score'] is None
  assert state['models'][mid]['status']=='needs_adaptation' and state['models'][mid]['score'] is None and mid not in registry and not (D/'receipts'/(mid.replace('/','_')+'.json')).exists()
  candidates.append(dict(change=c,J=J,job=job,spec=spec,done=done,run=run,proof=proof,score=score,gate=gate))
 for p,h in all_external.items():assert sha(p)==h,(p,'External evidence changed before batch publication')
 H.mkdir();backup=H/'before_verified_write';backup.mkdir();saved={}
 for path in [book,sf,rf,binding]:
  dest=backup/path.name;shutil.copy2(path,dest);assert sha(dest)==sha(path);saved[str(path)]={'backup':str(dest),'sha256':sha(path)}
 atomic_json(H/'backup_manifest.json',saved)
 # Reuse the frozen writer's full unsafe-write/preservation suite independently for both actual targets.
 for v in candidates:
  mid=v['change']['model_id'];home=H/mid.replace('/','--');home.mkdir();v['home']=home
  manifest=home/'verified_candidate.json';v['manifest']=manifest
  atomic_json(manifest,dict(model_id=mid,task=v['change']['task'],new_score=v['score'],expected_old_score=None,expected_book_sha256=before,job_completion=str(v['J']/'completion.json'),report_dir=v['run']['report_dir'],proof=v['run']['proof'],proof_sha256=v['run']['proof_sha256'],new_real_NPU_inference=True,historical_record_reuse=False,authorization=str(decision),authorization_sha256=sha(decision),local_full_precision_A100_gate=v['gate'],artifact_tool_authored_target_numeric_value_verified=v['change']))
  src=base64.b64decode(TEST_SOURCE_BASE64,validate=True).decode().replace('InternVL2_2B_A100_existing_D_verified_blank_G_publication_v2_1805_20261115',str(home.relative_to(B)))
  tp=home/'isolated_writer_tests_source.py';tp.write_text(src,encoding='utf-8')
  with (home/'isolated_11_writer_tests.json').open('xb') as out,(home/'isolated_11_writer_tests_stderr.log').open('xb') as err:
   result=subprocess.run(['python3','-B',str(tp)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONPATH=str(R/'scripts'),PYTHONDONTWRITEBYTECODE='1'))
  assert result.returncode==0,(home/'isolated_11_writer_tests_stderr.log').read_text();tests=read(home/'isolated_11_writer_tests.json');assert tests['passed'] and tests['test_count']==11 and tests['production_workbook_unchanged']
 fixture=H/'isolated_sequential_pair_fixture';fixture.mkdir();tb=fixture/'book.xlsx';shutil.copy2(book,tb);auth=fixture/'authorization.json';atomic_json(auth,dict(test_fixture_only=True,origin='isolated_test_fixture',test_root=str(fixture)))
 predicted=[]
 for v in candidates:predicted.append(write_verified_A100_G(tb,v['change']['row'],v['change']['model_id'],v['score'],fixture/v['change']['model_id'].replace('/','--'),expected_task=v['change']['task'],decision_path=auth,expected_book_hash=sha(tb),manifest_path=v['manifest']))
 assert sha(book)==before
 unchanged({p:v['sha256'] for p,v in saved.items()})
 atomic_json(H/'transaction.json',dict(phase='single_full_evidence_and_11_tests_passed_same_prepublication_baseline',before=saved,expected_final_book=sha(tb)))
 for n,v in enumerate(candidates):
  c=v['change'];mid=c['model_id'];entry=state['models'][mid];spec=v['spec'];done=v['done'];proof=v['proof'];run=v['run'];folder=Path(run['report_dir'])
  write=write_verified_A100_G(book,c['row'],mid,v['score'],v['home']/'writer_backup',expected_task=c['task'],decision_path=decision,expected_book_hash=before if n==0 else candidates[n-1]['write']['after_sha256'],manifest_path=v['manifest'])
  assert write['after_sha256']==predicted[n]['after_sha256'];v['write']=write
  atomic_json(H/'transaction.json',dict(phase='authorized_G_batch_write_in_progress',completed_targets=[x['change']['model_id'] for x in candidates[:n+1]],before=saved,latest_write=write))
  for item in done['accepted']:
   stage='fresh3' if item['sample_count']==3 else 'formal200';entry['attempts'].append(dict(count=item['sample_count'],diagnostics=str(v['J']/stage),report_dir=item['report_dir'],independent_verification=item['proof'],proof_sha256=item['proof_sha256'],score=item['score'],new_real_NPU_inference=True))
  entry.update(status='done',row=write['row'],score=v['score'],score_in_workbook=repr(v['score']),published_score_in_workbook=v['score'],workbook_score_representation='exact',report_dir=str(folder),completed_at=done['completed_at'],active_config=str(R/spec['config']),active_attempt=str(v['J']/'formal200'),original_protocol_final_qualification_complete=True,qualification_basis=proof['qualification_basis'],A100_historical_engine_or_tensor_capture_claimed=False,reason=None,last_error=None,workbook_write=write,active_report_files={name:{'path':str(folder/name),'sha256':h} for name,h in proof['report_sha256'].items()},A100_relative_alignment=dict(metric='Accuracy',A100_D=float(c['A100_full_precision_cached_D']),relative_error=proof['relative_error_to_A100_table_D'],passed=True,sample_count=200,authorization=str(decision),authorization_sha256=sha(decision)))
  spec.update(accepted_formal200_report=str(folder),accepted_formal200_proof=run['proof'],accepted_formal200_proof_sha256=run['proof_sha256'],active_report_files=entry['active_report_files']);registry[mid]=spec
 assert sha(book)==sha(tb);state['updated_at']=now();targets={v['change']['model_id'] for v in candidates}
 assert all(state['models'][k]==val and registry.get(k)==old_reg.get(k) for k,val in old_state['models'].items() if k not in targets)
 counts=Counter(v['status'] for v in state['models'].values());assert counts['done']==170 and counts['needs_adaptation']==220 and counts['skipped_no_complete_weights']==310
 unchanged({str(p):saved[str(p)]['sha256'] for p in [sf,rf,binding]})
 atomic_json(rf,registry);atomic_json(sf,state)
 for v in candidates:atomic_json(D/'receipts'/(v['change']['model_id'].replace('/','_')+'.json'),state['models'][v['change']['model_id']])
 receipt=dict(completed=True,UTC=now(),authorized_single_verified_target=True,targets=[dict(model_id=v['change']['model_id'],new_G=v['score'],A100_D=v['change']['A100_full_precision_cached_D'],exact_three_D_errors=v['proof']['relative_error_to_current_A100_D'],workbook_write=v['write'],new_report=v['run']['report_dir'],formal_proof=v['run']['proof'],formal_proof_sha256=v['run']['proof_sha256'],actual_forward_count=v['proof']['actual_forward_count'],source_tensor_count=v['proof']['source_tensor_count'],invalid_indices=v['proof']['invalid_indices'],empty_indices=v['proof']['empty_indices'],truncated_indices=v['proof']['truncated_indices'],actual_NPU_card=4) for v in candidates],canonical_counts=dict(counts),old169_state_registry_and_G_preserved=True,canonical_before_sha256={p:v['sha256'] for p,v in saved.items()},canonical_after_sha256={str(p):sha(p) for p in [book,sf,rf]},all_new_target_external_SHA_freshly_checked=len(all_external),distinct_files_hashed=len(cache),original_protocol_final_qualification_complete=True,local_CAS_commit_pending=True)
 atomic_json(H/'publication_receipt.json',receipt);atomic_json(H/'transaction.json',dict(phase='single_authorized_G_state_registry_receipts_committed',receipt=str(H/'publication_receipt.json')))
print(json.dumps(receipt,ensure_ascii=False))

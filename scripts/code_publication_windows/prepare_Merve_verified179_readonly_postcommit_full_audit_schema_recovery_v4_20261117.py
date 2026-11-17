import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,json,hashlib
P=Path(__file__).parent;old=(P/'publish_MervePaliCompact_verified179_dual_transaction_v3_exact_negative_fixture_20261117.py').read_text(encoding='utf8');assert hashlib.sha256(old.encode()).hexdigest()=='96b8f91ecbd645f8a4e920e4209c9f840a90c6829c28436796fe2a6265ef4317'
source=old[:old.index('assert not H.exists()')]
a=source.index("    p = D / 'A100_ALIGNMENT_LOCAL_WORKBOOK.json'",source.index('def finish_local_and_full_target_audit'))
b=source.index('    transitions = []',a)
guard='''    p = D / 'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
    assert sha(p)=='3534a1dc1762fcb276d9c074d52bc47b04cf82459bb58424c2893804dfd301d1'
    binding=read(p)
    assert binding['local_workbook_sha256']==lp['after_sha256'] and binding['real_record_count']==binding['real_measured_record_count']==179
    assert sha(H/'before_local_binding_verbatim.json')=='ed4eead6ccd1802f5731266f8f9ba8b180cf0a17ad39b3e1076abd2fc267e38d'
    assert (H/'user_local_publication_receipt.json').read_bytes()==local and binding['last_local_publication_receipt_SHA256']==Q['local_receipt_SHA256']
    dual=read(H/'dual_publication_completed.json')
    assert dual['completed'] and dual['local_and_server_workbook_sha256']==sha(book) and dual['local_publication_receipt_sha256']==Q['local_receipt_SHA256']
'''
source=source[:a]+guard+source[b:]
assert source.count("forward_count=t['actual_native_model_forward_calls']")==1
source=source.replace("forward_count=t['actual_native_model_forward_calls']","forward_count=t['actual_forward_count']")
tail='''
REC=B/'Merve_verified179_committed_dual_result_readonly_final_audit_field_schema_recovery_v4_20261117'
assert not REC.exists();REC.mkdir()
started=now();canonical=[Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]+[D/n for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json']]
before={str(p):sha(p) for p in canonical}
try:
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in roles]
 I=B/'MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117'
 fail=read(I/'publication_actual_completion.json');assert fail['actual_returncode']==1 and fail['source_SHA256']=='96b8f91ecbd645f8a4e920e4209c9f840a90c6829c28436796fe2a6265ef4317'
 assert "KeyError: 'actual_native_model_forward_calls'" in (I/'publication_stderr.log').read_text()
 request=read(H/'reviewed_local_receipt_ready.json')
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
  assert not [p for p in project_processes() if p['role'] in roles]
  result=finish_local_and_full_target_audit(request)
  assert {str(p):sha(p) for p in canonical}==before
  receipt=dict(passed=True,started_UTC=started,completed_UTC=now(),actual_returncode=0,original_failed_publication_completion_preserved=True,canonical_G_state_registry_binding_not_rewritten=True,all_original_full_numeric_and_dual_publication_checks_retained=True,only_final_audit_forward_count_key_corrected_to_actual_existing_receipt_schema=True,original_model_and_five_reports_and_proofs_and_sources_unchanged=True,canonical_before_and_after_SHA256=before,full_target_audit_SHA256=result['audit_SHA256'],verified179_export=result['audit']['verified179_export'],verified179_export_SHA256=result['audit']['verified179_export_SHA256'])
  atomic_json(REC/'actual_completion.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False))
except BaseException as exc:
 atomic_json(REC/'actual_completion.json',dict(passed=False,started_UTC=started,completed_UTC=now(),actual_returncode=1,error=repr(exc),canonical_before_SHA256=before));raise
'''
source+=tail;ast.parse(source);sn='scripts/recover_Merve_verified179_committed_dual_readonly_full_audit_field_schema_v4_20261117.py';sh=hashlib.sha256(source.encode()).hexdigest();(P/Path(sn).name).write_bytes(source.encode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,base64,hashlib,ast,subprocess
R=Path('/root/projects/comperror');p=R/'__SN__';assert not p.exists();raw=base64.b64decode(__RAW__);assert hashlib.sha256(raw).hexdigest()=='__SH__';ast.parse(raw);p.write_bytes(raw)
Q=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117/reviewed_inputs.json'
r=subprocess.run(['python3','-B',str(p),str(Q)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source=str(p),source_SHA256='__SH__')));raise SystemExit(r.returncode)
'''.replace('__SN__',sn).replace('__RAW__',repr(base64.b64encode(source.encode()).decode())).replace('__SH__',sh);ast.parse(remote);(P/'run_Merve179_readonly_postcommit_full_audit_schema_recovery_v4_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(source=sn,source_SHA256=sh,original_failed_publisher_and_canonical_preserved=True)))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64
P=Path(__file__).parent;o=json.loads((P/'R_Donut_empty_raw_report_schema_Deadbydawn_constructor_full_sources_20261117.json').read_text(encoding='utf8'));old=o['sources']['scripts/verify_donut_invoice_own_native_fp16_v5_approved_measurement_full_linear_KV_20261117.py']['text']
helper='''def present_detail_raw(detail):
 values=[]
 if 'raw_output' in detail:values.append(detail['raw_output'])
 extra=detail.get('extra',{})
 assert isinstance(extra,dict)
 if 'raw_output' in extra:values.append(extra['raw_output'])
 assert values and all(isinstance(x,str) for x in values),'A real raw_output field is required even when empty'
 assert all(x==values[0] for x in values),'Conflicting actual raw-output fields'
 return values[0]
'''
new=old.replace('def verify(folder,count,accept_report_metrics=True):',helper+'\ndef verify(folder,count,accept_report_metrics=True):')
new=new.replace("out['prediction']==detail['raw_output']","out['prediction']==present_detail_raw(detail)")
new=new[:new.index("if __name__=='__main__':")]+'''if __name__=='__main__':
 import contextlib
 assert len(sys.argv)==5 and sys.argv[3]=='--diagnostic-existing-complete-report-no-accept'
 folder=Path(sys.argv[1]).resolve();count=int(sys.argv[2]);dest=Path(sys.argv[4]);assert count==200
 # All original report/approval/source/metric checks still execute. Only the
 # resulting review role changes: the failed original runner remains failed.
 with contextlib.redirect_stdout(sys.stderr):p=verify(folder,count,accept_report_metrics=True)
 p.update(passed=False,diagnostic_only=True,NEW_real_NPU_inference=False,historical_record_reuse=True,existing_real_NPU_inference_verified=True,original_failed_runner_completion_not_replaced=True,all_original_report_approval_source_metric_and_numerical_checks_executed=True,only_schema_change='Require equal raw_output in top-level or existing extra field, including the real empty string; neither field missing is rejected')
 p['external_evidence_sha256'][str(Path(__file__).resolve())]=sha(__file__)
 assert p['numeric_review_passed'] and not p['passed'] and not p['NEW_real_NPU_inference'] and not p['original_protocol_final_qualification_complete']
 assert not dest.exists();dest.write_bytes(json.dumps(p,ensure_ascii=False,allow_nan=False,indent=2).encode());print(json.dumps(p,ensure_ascii=False,allow_nan=False))
'''
ast.parse(new);a=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='numeric');b=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='numeric');assert ast.dump(a)==ast.dump(b)
rel='scripts/verify_donut_invoice_existing200_v6_present_raw_field_full_numeric_diagnostic_20261117.py';(P/Path(rel).name).write_bytes(new.encode())
test='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1])]
from verify_donut_invoice_existing200_v6_present_raw_field_full_numeric_diagnostic_20261117 import present_detail_raw
assert present_detail_raw({'raw_output':''})==''
assert present_detail_raw({'extra':{'raw_output':''}})==''
assert present_detail_raw({'raw_output':'real','extra':{'raw_output':'real'}})=='real'
for bad in [{},{'prediction':''},{'raw_output':None},{'raw_output':'x','extra':{'raw_output':''}}]:
 try:present_detail_raw(bad)
 except AssertionError:pass
 else:raise AssertionError('Missing or conflicting raw output was accepted')
print('7 real-presence/empty/conflict schema checks passed; no model forward')
'''
sup='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,fcntl,subprocess,os,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'DonutInvoice_existing_complete200_v6_present_raw_full_numeric_independent_diagnostic_real_FLOCK_20261117';F=R/'workspace/reports/katanaml-org_invoices-donut-model-v1/image-to-text/20261117_045854';J=B/'DonutInvoice_own_original_FP16_poscap2048_approved_fresh3_fixed200_v4_real_FLOCK_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 f=H/n;t=f.with_name(f.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,f)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);assert not runner_lock_owners() and not [x for x in project_processes() if x['PID']!=os.getpid() and x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protected=[f for f in F.iterdir() if f.is_file()]+[J/'job.json',J/'completion.json',D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];before={str(f):sha(f) for f in protected};put('protected_before.json',before)
 assert json.loads((J/'completion.json').read_bytes())['actual_returncode']==1
 source=R/'scripts/verify_donut_invoice_existing200_v6_present_raw_field_full_numeric_diagnostic_20261117.py';started=time.monotonic();timeout=False
 with (H/'stdout.json').open('xb') as out,(H/'stderr.log').open('xb') as err:
  p=subprocess.Popen(['python3','-B',str(source),str(F),'200','--diagnostic-existing-complete-report-no-accept',str(H/'review.json')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
  while p.poll() is None:
   timeout=timeout or time.monotonic()-started>21600;put('state.json',dict(UTC=now(),PID=os.getpid(),phase='independent_full_numeric_review_of_existing_complete200_no_new_NN',child_PID=p.pid,source_SHA256=sha(source),actual_unique_original_FLOCK=True,timeout_review_required_no_signal=timeout,original_verify_budget_seconds=21600));time.sleep(10)
 after={str(f):sha(f) for f in protected};assert after==before;put('protected_after.json',after);rc=p.returncode
 if rc==0:
  review=json.loads((H/'review.json').read_bytes());assert review['numeric_review_passed'] and not review['passed'] and not review['NEW_real_NPU_inference'] and review['sample_count']==200
 put('actual_completion.json',dict(UTC=now(),actual_returncode=rc if not timeout else 1,numeric_process_actual_returncode=rc,timeout_review_required_no_signal=timeout,elapsed_seconds=time.monotonic()-started,original_failed_job_unchanged=True,all_original_five_reports_and_canonical_hashes_unchanged=True,no_NPU_inference_or_G_write=True,review_SHA256=sha(H/'review.json') if (H/'review.json').exists() else None));return rc if not timeout else 1
if __name__=='__main__':
 try:raise SystemExit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),original_job_and_reports_unchanged=True))
  raise
'''
payload={rel:new.encode(),'scripts/tests/test_donut_raw_output_present_even_empty_schema_20261117.py':test.encode(),'scripts/supervise_Donut_existing200_v6_full_numeric_diagnostic_original_FLOCK_20261117.py':sup.encode()}
items={n:dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest()) for n,raw in payload.items()}
for raw in payload.values():ast.parse(raw)
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,json,hashlib,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';ITEMS=__ITEMS__
for rel,item in ITEMS.items():
 raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['SHA256'];ast.parse(raw);f=R/rel
 if f.exists():assert f.read_bytes()==raw
 else:f.write_bytes(raw)
I=B/'DonutInvoice_existing200_v6_full_numeric_schema_recovery_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'schema_tests.log').open('xb') as log:p=subprocess.run(['python3','-B','scripts/tests/test_donut_raw_output_present_even_empty_schema_20261117.py'],cwd=R,stdout=log,stderr=subprocess.STDOUT);assert p.returncode==0
with (I/'stdout.log').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B','scripts/supervise_Donut_existing200_v6_full_numeric_diagnostic_original_FLOCK_20261117.py'],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,sources={n:x['SHA256'] for n,x in ITEMS.items()},schema_tests=(I/'schema_tests.log').read_text(),no_new_NPU_inference=True,original_failed_job_not_replaced=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__ITEMS__',repr(items));ast.parse(remote);(P/'launch_Donut_existing200_v6_present_raw_schema_full_numeric_locked_diagnostic_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in items.items()}))

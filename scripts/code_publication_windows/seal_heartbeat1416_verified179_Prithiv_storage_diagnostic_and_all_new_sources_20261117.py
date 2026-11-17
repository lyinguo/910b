import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast,tomllib
P=Path(__file__).parent;auto=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes();av=tomllib.loads(auto.decode());assert av['id']=='910b' and av['status']=='ACTIVE' and av['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300';assert hashlib.sha256(auto).hexdigest()=='aead3fa15a3de326293146993e3991396e122928e3b249f52a83c2f25f96b919'
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');raw=book.read_bytes();bh=hashlib.sha256(raw).hexdigest();assert bh=='9bd689a4a763bb716c6f9d4c3869aed95bb8d4b1ea9ede4b418591e3226a859c';st=book.stat();LS=dict(SHA256=bh,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
Wnames=['read_verified179_required_full_handoff_then_late_actual_20261117_remote.py','read_Prithiv_completed_card2_numeric3_lossless_storage_layout_only_20261117_remote.py','read_Prithiv_lossless_complete_NPY_diagnostic_actual_progress_no_NN_20261117_remote.py','read_Donut_completed200_existing_full_numeric_scope_and_matrix_capacity_only_20261117_remote.py','prepare_Prithiv_completed3_exact_lossless_standard_NPY_storage_diagnostic_20261117.py','seal_heartbeat1416_verified179_Prithiv_storage_diagnostic_and_all_new_sources_20261117.py']
Qnames=['R_heartbeat1416_verified179_required_full_handoff_and_late_actual_20261117.json','R_heartbeat1416_verified179_frozen_monitor_complete_checkpoint_20261117.json','R_heartbeat1416_Prithiv_completed_card2_numeric3_lossless_layout_20261117.json','R_heartbeat1416_Prithiv_lossless_complete_NPY_storage_isolated_tests_20261117.json','R_heartbeat1416_Prithiv_completed3_exact_lossless_storage_diagnostic_actual_launch_20261117.json','R_heartbeat1416_Prithiv_complete_NPY_lossless_diagnostic_progress1_20261117.json','R_heartbeat1416_Prithiv_complete_NPY_lossless_diagnostic_progress2_20261117.json','R_heartbeat1416_Prithiv_complete_NPY_lossless_diagnostic_progress3_20261117.json','R_heartbeat1416_Donut_completed200_full_numeric_and_matrix_capacity_metadata_only_20261117.json']
def enc(p):
 b=p.read_bytes();return dict(data=base64.b64encode(b).decode(),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b))
W={n:enc(P/n) for n in Wnames};Q={n:enc(P/n) for n in Qnames}
code='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114';H=D/'heartbeat_handoffs/A100_verified179_Merve200_dual_multi16_two_card_pass_capacity_blocked_all_code_preserved_20261117_061053_934942';W=__W__;Q=__Q__;LS=__LS__
sys.path[:0]=[str(R),str(R/'scripts')];from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
pf=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pf.read_bytes();ptr=js(pf);assert ptr['handoff_directory']==str(H) and sha(H/'handoff.txt')=='69ffe482b4766393f21cc05460f7d84aa7bfbe6c9b62432c09e532ff70c7722f';old=Path(ptr['new_primary_pending']);assert sha(old)==ptr['new_primary_pending_SHA256']=='2c675b7a2e601ab14a48eb692198dd6a29eaf1bd34f7d02e4a80bbf5cf1f625c';pending=js(old);records=list(pending['all_current_primary_source_and_true_consumer_configs']);assert len(records)==995;known={v['path']:v for v in records}
for path,v in known.items():assert sha(path)==v['SHA256'],('Preserved primary source changed',path)
N=H/'late_actual_notes/heartbeat1416_Prithiv_complete_NPY_storage_diagnostic_not_adopted_and_source_closure_20261117';N.mkdir(parents=True,exist_ok=False);paths=[];local={}
for n,v in Q.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];p=N/'local_actual_original_bytes'/(n+'.original');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);local[n]=dict(path=str(p),SHA256=v['SHA256'],bytes=len(raw))
for n,v in W.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p=R/'scripts/code_publication_windows'/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
 paths.append(p)
for name,h in [('universal_eval/utils/lossless_complete_alias_npz_fortran_head_20261117.py','11be8afdceb5becc5fde2a098a614f1625ec2b1e2711556ceb22c7e0c8d7ea88'),('scripts/diagnostic_Prithiv_completed_card2_true3_lossless_standard_Fortran_head_no_NN_20261117.py','9b6af404caad9561fab88bf08b22cd9db1e23b8b41573d3ab6a4e1377fb381e7'),('scripts/tests/test_lossless_complete_alias_standard_Fortran_head_20261117.py','9ed2d6f4fd23bb570d3dd1451a796e57ef3033db43e51c13b3405c8237ae4233')]:assert sha(R/name)==h;paths.append(R/name)
added=[]
for p in paths:
 raw=p.read_bytes();ast.parse(raw)
 if str(p) in known:assert known[str(p)]['SHA256']==sha(p);continue
 snap=N/'primary_source_original_bytes'/(str(p.relative_to(R))+'.txt');snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(raw);v=dict(path=str(p),relative=str(p.relative_to(R)),SHA256=sha(p),bytes=len(raw),snapshot=str(snap),role='Completed-only exact storage feasibility diagnostic or actual Windows code; not adopted into official model/source bindings');records.append(v);known[str(p)]=v;added.append(v)
new=C/'pending_after_daily_20261117_verified179_Prithiv_complete_NPY_storage_diagnostic_all_primary_v18.json';pending.update(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(old),parent_pending_SHA256=sha(old),all_current_primary_source_and_true_consumer_configs=records,CodeDay17_already_pushed_no_second_commit=True,Prithiv_storage_candidate_diagnostic_only_not_adopted=True);put(new,pending)
diag=B/'Prithiv_completed_card2_true3_exact_alias_standard_Fortran_full_head_storage_diagnostic_only_20261117';actual=js(diag/'actual_completion.json');review=js(diag/'review.json');assert actual['actual_returncode']==0 and actual['full_original_numeric_assertions_passed'] and actual['review_SHA256']==sha(diag/'review.json')=='7fa3fd989dd896b1dfa101fad13a001b1a0cf575cd89f11d6a8137a199152d9f';assert review['diagnostic_only'] and not review['passed'] and review['no_NEW_NPU_inference'] and review['original_full_numeric_proof']['actual_forward_count']==475
assert review['capacity_review']['empirical_estimate_bytes']==127394500932 and not review['capacity_review']['passed'];assert review['canonical_SHA256_before_and_after'][str(D/'state.json')]==sha(D/'state.json')
for name in ['review.json','state.json','actual_completion.json']:(N/('Prithiv_'+name+'.original')).write_bytes((diag/name).read_bytes())
statpath=B/'MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117/active179_full_report_stat_baseline.json';stats=js(statpath);assert len(stats['reports'])==179
exp=js(B/'MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117/verified179_export_complete_metadata.json')
for rec in exp['records']:
 mid=rec['model_id'];files=rec.get('active_report_files') or {n:dict(path=str(Path(rec['accepted_report'])/n)) for n in stats['reports'][mid]}
 for name,info in files.items():
  p=Path(info['path']);s=p.stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==stats['reports'][mid][name],(mid,name)
procs=project_processes();locks=runner_lock_owners();assert not procs and not locks;book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==LS['SHA256'];canonical={str(D/n):sha(D/n) for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']};canonical[str(book)]=sha(book)
counts=dict(delivered=179,numeric_within5pct=164,numeric_outside5pct=15,complete_weights_pending_formal=211,confirmed_missing=310,full_original_qualification=65);free=shutil.disk_usage(R).free
o=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,canonical_counts=counts,no_new_formal_NPU200_G_or_full_qualification=True,canonical_SHA256=canonical,all179_complete_five_report_stat_exactly_unchanged=True,actual_project_processes=procs,actual_original_FLOCK=locks,actual_free_bytes=free,original_Prithiv200_budget_bytes=123344571732,original_Prithiv200_budget_shortfall_bytes=123344571732-free,Prithiv_storage_diagnostic=dict(completed_UTC=actual['completed_UTC'],actual_returncode=0,complete_old3_forward_count=475,no_NEW_NPU=True,all_original_complete_values_occurrences_and_assertions_retained=True,review_SHA256=sha(diag/'review.json'),storage_candidate_capacity_bytes=127394500932,capacity_passed=False,not_adopted_into_any_formal_model_job_or_protocol=True,old_original3_values_and_proof_unchanged=True),Donut_stat_only_inventory=dict(all200_complete_NPZ_bytes=28352491845,potential_nonexample170_bytes=24605464565,original_full_NF6_diagnostic_only=True,no_new_acceptance_or_original_failure_promotion_or_deletion=True,not_sufficient_alone_to_satisfy_Prithiv_capacity_shortfall=True),current_severity='严重阻塞｜正式评测未运行',earliest_current_resource_block_proven_BJT='2026-11-17 14:04:18',latest_formal_qualification_BJT='2026-11-17 13:35:10.750282',latest_dual_publication_full_audit_BJT='2026-11-17 14:03:11.137807',new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records),new_primary_original_sources=added,actual_original_local_packets=local,local_fresh_workbook_stat=LS,Day17_already_submitted_no_second_commit=True,no_file_deletion_or_runtime_source_mutation_in_this_heartbeat=True)
put(N/'actual.json',o)
parent=Path(ptr['parent_handoff_directory']);assert sha(parent/'handoff.txt')=='5b72fa88e2be5662ae8b31865c4b80c089e0fa2a3566063e18f7b74d3b0e9af8'
for k in ['parent_handoff','parent_handoff_path','parent_handoff_txt']:ptr[k]=str(parent/'handoff.txt')
for k in ['parent_facts','parent_facts_path','parent_facts_json']:ptr[k]=str(parent/'facts.json')
ptr['canonical_counts']=dict(delivered=179,numeric_within=164,numeric_outside=15,own_complete_weights_pending_delivery=211,confirmed_missing=310,full_original_protocol_qualified=65);ptr['verified_delivered_count']=179;ptr['current_canonical_counts']=counts
for k in ['handoff','handoff_path','handoff_txt']:assert ptr[k]==str(H/'handoff.txt')
for k in ['handoff_SHA256','handoff_sha256','handoff_txt_SHA256','handoff_text_SHA256']:assert ptr[k]==sha(H/'handoff.txt')
for k in ['facts','facts_path','facts_json']:assert ptr[k]==str(H/'facts.json')
for k in ['facts_SHA256','facts_sha256','facts_json_SHA256']:assert ptr[k]==sha(H/'facts.json')
closing=js(H/'closing_addendum.json');(N/'closing_before.original').write_bytes((H/'closing_addendum.json').read_bytes());closing.update(UTC=o['UTC'],latest_snapshot_is_facts=False,latest_actual_notes=str(N/'actual.json'),latest_actual_notes_SHA256=sha(N/'actual.json'),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records),current_Prithiv_storage_diagnostic_completed_not_adopted=True,no_current_NN_or_NF=True)
tmp=H/'closing_addendum.heartbeat1416.tmp';assert not tmp.exists();tmp.write_bytes(json.dumps(closing,ensure_ascii=False,indent=2).encode());os.replace(tmp,H/'closing_addendum.json')
ptr.update(UTC=o['UTC'],closing_addendum_SHA256=sha(H/'closing_addendum.json'),latest_actual_note=str(N/'actual.json'),latest_actual_note_SHA256=sha(N/'actual.json'),latest_actual_snapshot=str(N/'actual.json'),latest_actual_snapshot_SHA256=sha(N/'actual.json'),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records));assert pf.read_bytes()==before;tmp=pf.with_name('A100_ALIGNMENT_HANDOFF_CURRENT.heartbeat1416.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(ptr,ensure_ascii=False,indent=2).encode());os.replace(tmp,pf)
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert npu.returncode==0;put(N/'npu_smi_actual.json',dict(returncode=0,stdout=npu.stdout,stderr=npu.stderr));print(json.dumps({k:v for k,v in o.items() if k not in ['new_primary_original_sources','actual_original_local_packets']}|dict(current_pointer_aliases_and_179_counts_consistent=True,pointer_SHA256=sha(pf),closing_SHA256=sha(H/'closing_addendum.json'),npu_smi_SHA256=sha(N/'npu_smi_actual.json')),ensure_ascii=False))
'''.replace('__W__',repr(W)).replace('__Q__',repr(Q)).replace('__LS__',repr(LS));ast.parse(code);(P/'seal_heartbeat1416_verified179_Prithiv_completed_storage_diagnostic_all_sources_20261117_remote.py').write_bytes(code.encode());print(json.dumps(dict(prepared=True,Windows_code_sources=len(W),actual_packets=len(Q),before_TOML_SHA256=hashlib.sha256(auto).hexdigest(),local_book_fresh_SHA256=bh)))

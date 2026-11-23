"""Reuse reviewed code-only publication, with exact Day24 consumers and tests."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def rename(s):return s.replace('20261123','20261124').replace('2026-11-23','2026-11-24').replace('daily23','daily24').replace('Day23','Day24').replace('Day_23','Day_24').replace('cd23','cd24')
def write(n,s):ast.parse(s);(V/n).write_text(s,encoding='utf8')
extra=[
 'scripts/maintenance/collect_daily24_verified185_explicit_primary_sources_20261124.py',
 'scripts/maintenance/run_daily24_actual_R_source_tests_no_model_20261124_remote.py',
 'scripts/maintenance/verify_daily24_R_source_concurrent_CAS_before_publish_20261124.py',
 'scripts/maintenance/sync_daily24_reviewed_code_policy_and_remote_receipts_only_20261124.py',
 'scripts/code_publication_windows/daily24_reviewed_code_publication_controls_20261124.py',
 'scripts/code_publication_windows/'+Path(__file__).name]
s=rename((V/'collect_daily23_verified185_explicit_primary_sources_20261123.py').read_text(encoding='utf8'))
a="'bc5f4275df7abc884b3f021f096cb0a5934ce061fb6d7843f310dab587059ef8'";assert s.count(a)==1
s=s.replace(a,"'73032ebd63354be27e338950415f9c089e7f044cf0ddffe762b8eca21ee146fd'")
a=s.index('extra=[');b=s.index('\nargs=',a);s=s[:a]+'extra='+repr(extra)+s[b:]
write('collect_daily24_verified185_explicit_primary_sources_20261124.py',s)
s=rename((V/'run_daily23_actual_R_source_tests_no_model_v2_original_source_arg_20261123_remote.py').read_text(encoding='utf8'))
s=s.replace('daily_20261124_actual_unit_checks_no_model_v2_original_source_arg','daily_20261124_actual_unit_checks_no_model')
needle="assert all(sha(p)==h for p,h in canonical.items())\nvalue=dict(UTC="
insert="""assert all(sha(p)==h for p,h in canonical.items())
# Reuse the already completed new producer/independent-reader contract checks.
# Recheck every consumed source digest; do not replay meta/model/frontend data.
completed=[]
base=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for folder,h,count in [
 ('Kanana_complete_native_observer_synthetic_values_and_real_meta_registration_tests_v1_20261123','3ea092c400e335679619c6d951f327a30d62a8bb4933a214b7129da76d76a181',14),
 ('Kanana_independent_complete_capture_and_full32KV_reconstruction_synthetic_only_v1_20261123','91dba8a2df8d919d6f0f559e2bc4341fc0607b43dac130d3735ab35d9d42ef47',12)]:
    rp=base/folder/'review.json';assert sha(rp)==h
    z=json.loads(rp.read_bytes());assert z['passed'] and z['check_count']==count
    assert z['learned_CPU_GPU_NPU_forwards']==0 and not z['formal_NEW_NPU3'] and not z['formal_fullNF_or_200_or_qualification']
    for n,digest in z['source_sha256'].items():assert sha(R/n)==digest,n
    completed.append(dict(review=str(rp),review_SHA256=h,completed_contract_checks_reused=count,all_consumed_source_SHA_fresh=True,no_replay=True))
value=dict(UTC="""
assert s.count(needle)==1;s=s.replace(needle,insert)
needle="explicit_extra_full_value_artifact_modules=extra,";assert s.count(needle)==1
s=s.replace(needle,needle+"completed_Kanana_contract_checks_reused=completed,completed_Kanana_check_count_reused=26,")
write('run_daily24_actual_R_source_tests_no_model_20261124_remote.py',s)
for old,new in [
 ('verify_daily23_R_source_concurrent_CAS_before_publish_20261123.py','verify_daily24_R_source_concurrent_CAS_before_publish_20261124.py'),
 ('sync_daily23_reviewed_code_policy_and_remote_receipts_only_20261123.py','sync_daily24_reviewed_code_policy_and_remote_receipts_only_20261124.py')]:
    write(new,rename((V/old).read_text(encoding='utf8')))
s=rename((V/'daily23_reviewed_code_publication_controls_20261123.py').read_text(encoding='utf8'))
s=s.replace("BASE='43a149bb69510288451c2348159d6e7baa0d8d8e'","BASE='5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9'")
s=s.replace('R_daily24_actual_R_source_unit_checks_v2_original_source_arg_20261124.json','R_daily24_actual_R_source_unit_checks_20261124.json')
needle="assert len(rt['explicit_extra_full_value_artifact_modules'])==2"
assert s.count(needle)==1;s=s.replace(needle,needle+"\n    assert rt['completed_Kanana_check_count_reused']==26 and len(rt['completed_Kanana_contract_checks_reused'])==2")
s=s.replace("heartbeat=js(V/'R_heartbeat1731_verified185_required_v87_latest_v89_current_actual_20261124.json')","heartbeat=js(V/'R_heartbeat0223_required_root_v90_latest_v95_actual_20261124.json')")
s=s.replace("heartbeat['complete_current_pointer']['SHA256']","heartbeat['complete_required_and_later_handoff_records']['/root/projects/comperror/workspace/diagnostics/batch_910b/A100_ALIGNMENT_HANDOFF_CURRENT.json']['SHA256']")
s=s.replace('Preserve own LLaVA adaptations, complete numerical verification and serial resource recovery','Preserve own Kanana source, complete input controls and full-value observation and reconstruction')
s=s.replace('Own More/Xtuner/Hancom/Turkish graphs, dtype/source remap and complete original observers; Base probability graph remains independent-NF-failed, no acceptance claim','Own Kanana original private RegStage/operator graphs and complete native observation plus independent capture reconstruction; input/contract checks are not model acceptance')
s=s.replace("'R_daily24_actual_R_source_unit_checks_20261124.json','R_daily24_actual_R_source_unit_checks_20261124.json.stderr','R_daily24_actual_R_source_unit_checks_20261124.json',","'R_daily24_actual_R_source_unit_checks_20261124.json','R_daily24_actual_R_source_unit_checks_20261124.json.stderr',")
s=s.replace("tests_R=review['tests_passed_R'],tests_local=review['tests_passed_local'],","tests_R=review['tests_passed_R'],completed_Kanana_contract_checks_reused=26,tests_local=review['tests_passed_local'],")
s=s.replace("tests_passed_R=rt['tests_run'],tests_passed_local=local['tests_run'],","tests_passed_R=rt['tests_run'],completed_Kanana_contract_checks_reused=26,tests_passed_local=local['tests_run'],")
assert '43a149bb69510288451c2348159d6e7baa0d8d8e' not in s
assert 'R_heartbeat1731' not in s and 'complete_current_pointer' not in s
write('daily24_reviewed_code_publication_controls_20261124.py',s)
src={}
for n in extra:
    filename=Path(n).name;b=(V/filename).read_bytes();ast.parse(b);src[n]=b
payload={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in src.items()}
(V/'daily24_actual_consumer_explicit_primary_paths_20261124.json').write_text(json.dumps({n:r['SHA256'] for n,r in payload.items()},indent=2)+'\n',encoding='utf8')
c="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/collect_daily24_verified185_explicit_primary_sources_20261124.py')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(c);(V/'stage_collect_daily24_verified185_explicit_primary_sources_20261124_remote.py').write_text(c,encoding='utf8')
print(json.dumps(dict(primary_paths=len(payload),primary_SHA256={n:r['SHA256'] for n,r in payload.items()},explicit_inventory_only=True,new_model_or_NPU_checks_not_claimed=True)))

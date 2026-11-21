from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,subprocess,ast
V=Path(__file__).resolve().parent;repo=Path('D:/code-lv/910b');source=Path('D:/code-lv/.tmp/cd21s2')
sha=lambda b:hashlib.sha256(b).hexdigest()
pub=json.loads((V/'daily_20261121_publish.stdout.json').read_bytes());review=json.loads((V/'daily_20261121_plan_review.json').read_bytes());cas=json.loads((V/'R_daily21_R_source_concurrent_CAS_before_publish_20261121.json').read_bytes());assert pub['status']=='pushed' and pub['day']=='2026-11-21'
commit=pub['commit'];assert commit=='189b3146a82adf0dc9e50f12691007ff31de6b6b' and pub['audited_outgoing_commits']==[commit]
git='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def run(*args):return subprocess.run([git,'-C',str(repo),'-c','http.sslBackend=openssl','-c','credential.interactive=never',*args],capture_output=True,check=True).stdout
assert run('rev-parse','HEAD').decode().strip()==commit and run('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0]==commit and not run('status','--porcelain=v1','-z')
assert run('rev-list','--count',review['remote_expected_before']+'..HEAD').strip()==b'1' and b'Code-Day: 2026-11-21' in run('show','-s','--format=%B','HEAD')
policy=(source/'configs/publication/code_only.json').read_bytes();assert sha(policy)=='c8a705028535eace9acb88fb60fbe45bf6afa3b974a9a3566201d44698786f05' and (repo/'configs/publication/code_only.json').read_bytes()==policy
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,Code_Day='2026-11-21',status='pushed_and_remote_verified',repository='https://github.com/lyinguo/910b.git',branch=pub['branch'],commit=commit,remote_commit=commit,parent=review['remote_expected_before'],explicit_paths=2544,changed_files=206,new_source_or_config_files=205,only_existing_file_update='configs/publication/code_only.json',one_new_Code_Day21_commit=True,worktree_clean=True,no_merge_force_or_Gitee_history=True,only_code_config_necessary_documents=True,no_weights_data_arrays_reports_logs_or_credentials=True,tests_R=23,local_publication_tests=19,javascript_actual_node_syntax_checks=21,reviewed_plan_SHA256=review['reviewed_plan_SHA256'],full_source_diff_SHA256=sha(Path(review['full_diff_path']).read_bytes()),source_content_review_SHA256=sha((V/'daily_20261121_source_content_review_v2.json').read_bytes()),outgoing_full_tree_audited_by_actual_publisher_before_push=True,source_CAS_SHA256=sha((V/'R_daily21_R_source_concurrent_CAS_before_publish_20261121.json').read_bytes()),actual_task_input_protocol_data_excluded_from_Git_only_preserved_R=review['exact_task_data_exclusions'],actual_parameter_consumed_failed_config_preserved_in_code=True,initial_long_Windows_snapshot_path_and_LF_vs_CRLF_source_review_failures_preserved=True,initial_old_fixed_MJS_count_review_rejection_preserved_then_all21_checked=True,original_SyntaxWarnings_preserved=True)
raw=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode();(V/'daily_20261121_receipt.json').write_bytes(raw)
orig=json.loads((V/'R_heartbeat_0355_mandatory_handoff_full_20261121.json').read_bytes());D='/root/projects/comperror/workspace/diagnostics/batch_910b';ptr=orig['files'][D+'/A100_ALIGNMENT_HANDOFF_CURRENT.json'];ptr=ptr.get('complete_JSON') or json.loads(ptr['complete_text']);canonical=ptr['canonical_SHA256'];canonical[D+'/A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json']='6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c'
close=json.loads((V/'R_verified183_Meta_NF4_actual_pin_v2_Ovis_source_closure_20261121.json').read_bytes());q=dict(expected_pointer_SHA256=close['pointer_SHA256'],expected_parent_pending_SHA256=close['pending_SHA256'],canonical_SHA256=canonical,policy_before_SHA256=review['remote_expected_before'] and '7c60205d2f555e5502d50594b388154558a81e22d2e8002bc47244f20f09a550',policy_after_SHA256=sha(policy),policy_base64=base64.b64encode(policy).decode(),commit=commit,original_local_records=[])
for name in ['daily_20261121_receipt.json','daily_20261121_plan_review.json','daily_20261121_source_content_review_v2.json','daily_20261121_snapshot_tests_actual.json','R_daily21_actual_R_source_unit_checks_20261121.json','R_daily21_R_source_concurrent_CAS_before_publish_20261121.json','daily_20261121_publish.stdout.json','daily_20261121_publish.stderr.log','verified183_local_fresh_SHA_stat_during_Ovis_formal200_20261121.json']:
 b=(V/name).read_bytes();q['original_local_records'].append(dict(name=name,SHA256=sha(b),base64=base64.b64encode(b).decode()))
plan=V/'Day21_verified_code_push_R_policy_actual_handoff_CAS_plan_20261121.json';plan.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode())
src=V/'archive_Day21_reviewed_push_policy_transition_and_current_Ovis_results_logs_20261121.py';b=src.read_bytes();compile(b,str(src),'exec');items=[('scripts/maintenance/'+src.name,base64.b64encode(b).decode(),sha(b)),('workspace/diagnostics/batch_910b/code_publication_source_layout_20261114/'+plan.name,base64.b64encode(plan.read_bytes()).decode(),sha(plan.read_bytes()))]
stage="""import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,base64,hashlib,subprocess,os
R=Path('/root/projects/comperror');items=ITEMS
for rel,encoded,h in items:
 p=R/rel;b=base64.b64decode(encoded,validate=True);assert hashlib.sha256(b).hexdigest()==h
 if p.suffix=='.py':compile(b,str(p),'exec')
 if p.exists():assert p.read_bytes()==b
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
r=subprocess.run(['python3','-B',str(R/items[0][0]),str(R/items[1][0]),items[1][2]],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
""".replace('ITEMS',repr(items));ast.parse(stage);(V/'archive_Day21_verified_unique_push_R_policy_current_Ovis_20261121_remote.py').write_bytes(stage.encode())
print(json.dumps(dict(passed=True,commit=commit,verified_remote=True,receipt_SHA256=sha(raw),source_SHA256=items[0][2],plan_SHA256=items[1][2])))

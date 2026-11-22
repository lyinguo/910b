"""Read-only remote/unique commit and local workbook verification after push."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
V=Path(__file__).resolve().parent;repo=Path('D:/code-lv/910b');source=Path('D:/code-lv/.tmp/cd22s1')
review=json.loads((V/'daily_20261122_plan_review.json').read_bytes());published=json.loads((V/'daily_20261122_publish.stdout.json').read_bytes());assert published['status']=='pushed' and published['day']=='2026-11-22' and published['changed_files']==127
env=dict(os.environ,GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.sslBackend',GIT_CONFIG_VALUE_0='openssl');git='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def run(*args):
    result=subprocess.run([git,'-c','credential.interactive=never',*args],cwd=repo,env=env,capture_output=True);assert result.returncode==0,result.stderr.decode(errors='replace');return result.stdout.decode('utf8')
commit=run('rev-parse','HEAD').strip();remote=run('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').split()[0];assert commit==remote==published['commit']=='43a149bb69510288451c2348159d6e7baa0d8d8e'
assert run('status','--porcelain')=='' and run('branch','--show-current').strip()=='codex/910b-source-layout-20261114' and run('remote','get-url','origin').strip()=='https://github.com/lyinguo/910b.git'
assert run('rev-list',review['remote_expected_before']+'..HEAD').split()==[commit] and run('log','-200','--format=%B%x00').count('Code-Day: 2026-11-22')==1
changes=run('diff','--name-status',review['remote_expected_before'],commit).splitlines();assert len(changes)==127 and sum(x.startswith('A\t') for x in changes)==126 and [x for x in changes if x.startswith('M\t')]==['M\tconfigs/publication/code_only.json']
assert published['audited_outgoing_commits']==[commit]
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,status='pushed_and_remote_verified',day='2026-11-22',commit=commit,remote_commit=remote,branch=published['branch'],repository='https://github.com/lyinguo/910b.git',one_new_Code_Day22_commit=True,tests_R_existing=23,tests_R_original_private_cache_full_observer=3,local_publication_tests=19,javascript_actual_node_syntax_checks=23,changed_files=127,explicit_paths=2670,reviewed_plan_SHA256=review['reviewed_plan_SHA256'],full_diff_SHA256=review['full_diff_SHA256'],source_collection_SHA256=review['source_collection_SHA256'],complete_outgoing_commit_tree_audited=published['audited_outgoing_commits'],worktree_clean=True,only_existing_updated_path='configs/publication/code_only.json',policy_before_SHA256='c8a705028535eace9acb88fb60fbe45bf6afa3b974a9a3566201d44698786f05',policy_after_SHA256=hashlib.sha256((source/'configs/publication/code_only.json').read_bytes()).hexdigest(),no_merge_force_Gitee_history_or_R_model_mutation=True)
(V/'daily_20261122_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');before=book.stat();digest=hashlib.sha256(book.read_bytes()).hexdigest();after=book.stat();assert digest=='0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1' and (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns)
stat=dict(UTC=receipt['UTC'],path=str(book),SHA256=digest,bytes=after.st_size,mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns,verified_count=184,unchanged_during_read=True,no_write=True)
(V/'local_verified184_Day22_fresh_SHA_stat_20261122.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(receipt=receipt,local_workbook=stat),ensure_ascii=False))

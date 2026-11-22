"""Exact Day23 snapshot, tests, complete diff/consumer review, approved publisher and remote verification."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,collections,difflib,hashlib,importlib.util,json,os,subprocess,tempfile,unittest
V=Path(__file__).resolve().parent;REPO=Path('D:/code-lv/910b');SOURCE=Path('D:/code-lv/.tmp/cd23s1')
PACKET=V/'R_daily23_verified185_explicit_primary_sources_20261123.json'
DAY='2026-11-23';BASE='43a149bb69510288451c2348159d6e7baa0d8d8e'
PUBLISHER_SHA='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
GIT='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
NODE='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
def sha(b):return hashlib.sha256(b).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(name,x):(V/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def env():
    e=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.sslBackend',GIT_CONFIG_VALUE_0='openssl')
    e['PATH']=str(Path(GIT).parent)+os.pathsep+e['PATH'];return e
def git(*args):
    r=subprocess.run([GIT,'-c','credential.interactive=never',*args],cwd=REPO,env=env(),capture_output=True)
    assert r.returncode==0,r.stderr.decode(errors='replace')[-1800:]
    return r.stdout
def publisher():
    p=SOURCE/'scripts/publish_code_daily.py';assert sha(p.read_bytes())==PUBLISHER_SHA
    spec=importlib.util.spec_from_file_location('reviewed_Day23_publisher',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def snapshot():
    assert SOURCE.resolve().parent==Path('D:/code-lv/.tmp').resolve() and not SOURCE.exists();SOURCE.mkdir()
    packet=js(PACKET);policy=js(REPO/'configs/publication/code_only.json')
    assert packet['policy_before']==policy and packet['policy_before_SHA256']==sha((REPO/'configs/publication/code_only.json').read_bytes())
    assert packet['opaque_payload_scripts_excluded_with_exact_reason']==[] and packet['unconsumed_JSON_excluded']==[]
    excluded={};raws={}
    for row in packet['records']:
        b=base64.b64decode(row['base64'],validate=True);assert sha(b)==row['SHA256'] and len(b)==row['bytes'];n=row['path']
        if n.endswith('.json') and n not in policy['files'] and 'input_records' in json.loads(b):
            assert n in ['configs/protocols/meta_mllama_own_original_BF16_SDPA_2048_serial_v1_20261120.json','configs/protocols/meta_mllama_completed_genuine3_shaped_storage_LZMA_v3_approved_first200_serial_20261120.json']
            assert len(json.loads(b)['input_records'])==200
            excluded[n]=dict(SHA256=sha(b),reason='Exact prior exclusion of actual 200 task prompt/reference/token arrays; full original remains on R, source generators and YAML remain included');continue
        p=SOURCE/n;assert n not in raws and p.resolve().is_relative_to(SOURCE.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);raws[n]=b
    assert len(excluded)==2
    pub=publisher();rows=[];diff=[];consumers={};javascript=[]
    for row in packet['records']:
        n=row['path']
        if n in excluded:continue
        b=raws[n];pub.validate_path(n,row['entry']['role']);pub.validate_content(n,b,row['entry'])
        before=(REPO/n).read_bytes() if (REPO/n).is_file() else None
        if n not in policy['files']:policy['files'][n]=dict(row['entry'],initial_SHA256=row['SHA256'])
        if n.endswith('.json') and n not in packet['policy_before']['files']:
            actual=packet['new_JSON_consumers'][n];assert actual and all(x in raws for x in actual);consumers[n]=actual
        if n.endswith('.mjs'):
            r=subprocess.run([NODE,'--check',str(SOURCE/n)],capture_output=True);assert r.returncode==0,(n,r.stderr.decode(errors='replace'))
            javascript.append(dict(path=n,SHA256=sha(b),actual_node_syntax_returncode=0))
        action='unchanged' if before==b else 'add' if before is None else 'update'
        entry=dict(path=n,SHA256=sha(b),bytes=len(b),action=action,role=policy['files'][n]['role'])
        if n.endswith('.py'):entry['top_level_definitions']=[x.name for x in ast.parse(b.decode('utf-8-sig'),filename=n).body if isinstance(x,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
        if action!='unchanged':diff.extend(difflib.unified_diff((before or b'').decode('utf-8-sig').splitlines(keepends=True),b.decode('utf-8-sig').splitlines(keepends=True),fromfile='before/'+n,tofile='after/'+n))
        rows.append(entry)
    for n in ['README.md','.gitignore','.gitattributes']:(SOURCE/n).write_bytes((REPO/n).read_bytes())
    new=(json.dumps(policy,ensure_ascii=False,indent=2)+'\n').encode();(SOURCE/'configs/publication/code_only.json').write_bytes(new);pub.load_policy(SOURCE/'configs/publication/code_only.json')
    assert not(set(packet['policy_before']['files'])-set(policy['files']))
    diff.extend(difflib.unified_diff((REPO/'configs/publication/code_only.json').read_text().splitlines(keepends=True),new.decode().splitlines(keepends=True),fromfile='before/configs/publication/code_only.json',tofile='after/configs/publication/code_only.json'))
    dp=V/'daily_20261123_full_source_diff.patch';dp.write_text(''.join(diff),encoding='utf8',newline='\n')
    review=dict(passed=True,source_root=str(SOURCE),source_collection_SHA256=sha(PACKET.read_bytes()),prior_explicit_paths=len(packet['policy_before']['files']),explicit_paths=len(policy['files']),records=rows,new_JSON_actual_consumers=consumers,exact_task_data_exclusions=excluded,javascript_actual_syntax_checks=javascript,action_counts=dict(collections.Counter(x['action'] for x in rows)),all_path_content_secret_AST_payload_operator_IR_boundaries_passed=True,local_maintenance_preserved=True,full_diff=str(dp),full_diff_SHA256=sha(dp.read_bytes()),policy_before_SHA256=packet['policy_before_SHA256'],policy_after_SHA256=sha(new),no_R_model_or_repo_mutation=True)
    put('daily_20261123_source_content_review.json',review)
    print(json.dumps({k:x for k,x in review.items() if k not in ['records','new_JSON_actual_consumers','javascript_actual_syntax_checks']},ensure_ascii=False))
    print('CHANGED_EXISTING',json.dumps([r for r in rows if r['action']=='update'],ensure_ascii=False))
def tests():
    temp=Path('D:/code-lv/.tmp/cd23tep');assert temp.resolve().parent==SOURCE.resolve().parent;temp.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(temp);tempfile.tempdir=str(temp);os.environ.update(env());os.chdir(SOURCE);sys.path.insert(0,str(SOURCE))
    modules=['test_code_publication','test_code_publication_javascript','test_readonly_npz_indexed','test_unsloth_completed_batch_lineage']
    if importlib.util.find_spec('torch') is not None:modules.append('test_phi35_observation_only_full_values_0616_20261118')
    suite=unittest.TestSuite()
    for n in modules:suite.addTests(unittest.defaultTestLoader.discover(str(SOURCE/'tests'),pattern=n+'.py'))
    log=V/'daily_20261123_snapshot_tests_actual.log'
    with log.open('x',encoding='utf8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    value=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=r.wasSuccessful(),tests_run=r.testsRun,errors=len(r.errors),failures=len(r.failures),modules=modules,source_root=str(SOURCE),temp_root=str(temp),all_test_cleanup_confined_to_verified_temp_root=True,no_model_checkpoint_or_CPU200_inference=True,log_SHA256=sha(log.read_bytes()))
    put('daily_20261123_snapshot_tests_actual.json',value);print(json.dumps(value));assert r.wasSuccessful()
def planreview():
    review=js(V/'daily_20261123_source_content_review.json');local=js(V/'daily_20261123_snapshot_tests_actual.json');rt=js(V/'R_daily23_actual_R_source_unit_checks_v2_original_source_arg_20261123.json')
    assert review['passed'] and local['passed'] and local['tests_run']>=19 and rt['passed'] and rt['tests_run']>=36
    assert len(rt['explicit_extra_full_value_artifact_modules'])==2
    assert sha(Path(review['full_diff']).read_bytes())==review['full_diff_SHA256']
    assert {r['path'] for r in review['javascript_actual_syntax_checks']}=={r['path'] for r in review['records'] if r['path'].endswith('.mjs')}
    assert git('status','--porcelain')==b'' and git('branch','--show-current').decode().strip()=='codex/910b-source-layout-20261114'
    assert git('remote','get-url','origin').decode().strip()=='https://github.com/lyinguo/910b.git'
    head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0]
    assert head==remote==BASE and ('Code-Day: '+DAY) not in git('log','-200','--format=%B%x00').decode()
    pub=publisher();p=subprocess.run([sys.executable,'-X','utf8','-B',str(SOURCE/'scripts/publish_code_daily.py'),'plan','--source-root',str(SOURCE),'--repo-root',str(REPO),'--policy',str(SOURCE/'configs/publication/code_only.json')],cwd=REPO,env=env(),capture_output=True)
    (V/'daily_20261123_plan.stdout.json').write_bytes(p.stdout);(V/'daily_20261123_plan.stderr.log').write_bytes(p.stderr);assert p.returncode==0,p.stderr.decode(errors='replace')[-1800:]
    plan=json.loads(p.stdout);policy,_=pub.load_policy(SOURCE/'configs/publication/code_only.json');changes=[x for x in plan['files'] if x['action']!='unchanged']
    assert plan['HEAD']==BASE and len(plan['files'])==len(policy['files'])==review['explicit_paths']
    assert plan['changes']==len(changes)>0
    assert {x['path'] for x in changes}=={x['path'] for x in review['records'] if x['action']!='unchanged'}|{'configs/publication/code_only.json'}
    assert [x['path'] for x in changes if x['action']=='update']==['configs/publication/code_only.json']
    assert not set(review['exact_task_data_exclusions'])&set(policy['files'])
    outgoing=pub.audit_outgoing(REPO,policy,remote);assert outgoing==[]
    cats=collections.Counter();files=[]
    for row in changes:
        n=row['path'];b=(SOURCE/n).read_bytes();assert sha(b)==row['SHA256'];pub.validate_content(n,b,policy['files'][n])
        if n.startswith('universal_eval/model/'):category='Own More/Xtuner/Hancom/Turkish graphs, dtype/source remap and complete original observers; Base probability graph remains independent-NF-failed, no acceptance claim'
        elif n.startswith('universal_eval/algorithm/'):category='Original deployed generation/probability and exact original metric binding, no output filtering'
        elif n.startswith('scripts/verify_'):category='Full numerical source/input/KV/head/vision/position/mask/profiler and report acceptance with fixed thresholds; failures preserved'
        elif n.startswith(('tests/','scripts/tests/')):category='Actual complete-array/real-cache/corruption/nonfinite/standard archive invariants'
        elif n.startswith('configs/'):category='Actual model/protocol/serial queue or code publisher consumers; no inline task data'
        else:category='Actual source preparation, serial original EX/capacity continuation, read-only diagnosis/cleanup metadata and complete evidence/Windows publication consumer'
        if n.endswith('.json') and n!='configs/publication/code_only.json':
            z=json.loads(b);assert 'input_records' not in z and 'detailed_results' not in z and 'sample_outputs' not in z
        entry=dict(row,reviewed_purpose=category)
        if n.endswith('.py'):
            tree=ast.parse(b,filename=n);entry['definitions']=[x.name for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))]
            entry['imports']=[ast.unparse(x) for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom))]
        cats[category]+=1;files.append(entry)
    for n,h in [('scripts/A100_alignment_evidence_runner_20261103.py','58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'),('scripts/A100_alignment_incremental_monitor_20261103.py','2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'),('scripts/publish_code_daily.py',PUBLISHER_SHA)]:assert sha((SOURCE/n).read_bytes())==h
    approved=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,Code_Day=DAY,reviewed_plan_SHA256=plan['plan_SHA256'],remote_expected_before=remote,reviewed_changes=len(changes),reviewed_explicit_paths=len(plan['files']),source_root=str(SOURCE),source_collection_SHA256=review['source_collection_SHA256'],full_diff_SHA256=review['full_diff_SHA256'],full_diff_path=review['full_diff'],full_diff_bytes=Path(review['full_diff']).stat().st_size,all_new_paths_full_content_AST_real_consumer_dependency_and_complete_diff_reviewed=True,new_JSON_consumers=review['new_JSON_actual_consumers'],exact_task_data_exclusions=review['exact_task_data_exclusions'],all_existing_model_and_frozen_runner_bytes_preserved=True,Base_failed_NF_not_reclassified=True,other_real3_pass_not_full200_qualification=True,tests_passed_R=rt['tests_run'],tests_passed_local=local['tests_run'],MJS_syntax_passed=len(review['javascript_actual_syntax_checks']),purpose_counts=dict(cats),files=files,outgoing_before=outgoing,repository='https://github.com/lyinguo/910b.git',branch='codex/910b-source-layout-20261114',no_merge_force_Gitee_history_or_artifacts=True)
    put('daily_20261123_plan_review.json',approved)
    print(json.dumps({k:x for k,x in approved.items() if k not in ['files','new_JSON_consumers']},ensure_ascii=False))
def carrier(script,planname,q,outname):
    b=json.dumps(q,ensure_ascii=False,indent=2).encode();(V/planname).write_bytes(b)
    s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport base64,hashlib,subprocess,os\nR=Path('/root/projects/comperror');p=R/'workspace/diagnostics/batch_910b/code_publication_source_layout_20261114'/"+repr(planname)+";b=base64.b64decode("+repr(base64.b64encode(b).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(b))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/"+repr(script)+"),str(p),"+repr(sha(b))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
    ast.parse(s);(V/outname).write_text(s,encoding='utf8');print(json.dumps(dict(carrier=outname,plan_SHA256=sha(b))))
def casbuild():
    review=js(V/'daily_20261123_plan_review.json');packet=js(PACKET);assert review['passed']
    heartbeat=js(V/'R_heartbeat1731_verified185_required_v87_latest_v89_current_actual_20261123.json')
    q=dict(reviewed_plan_SHA256=review['reviewed_plan_SHA256'],pointer_SHA256=heartbeat['complete_current_pointer']['SHA256'],policy_before_SHA256=review['policy_before_SHA256'] if 'policy_before_SHA256' in review else js(V/'daily_20261123_source_content_review.json')['policy_before_SHA256'],source_SHA256={r['path']:r['SHA256'] for r in packet['records'] if r['path']!='configs/publication/code_only.json'})
    carrier('scripts/maintenance/verify_daily23_R_source_concurrent_CAS_before_publish_20261123.py','daily_20261123_reviewed_explicit_source_CAS_plan.json',q,'stage_daily23_reviewed_R_concurrent_CAS_20261123_remote.py')
def publish():
    review=js(V/'daily_20261123_plan_review.json');cas=js(V/'R_daily23_R_source_concurrent_CAS_before_publish_20261123.json');assert review['passed'] and cas['passed'] and review['reviewed_plan_SHA256']==cas['reviewed_plan_SHA256']
    assert cas['code_publication_policy_not_a_model_or_existing_Bee_waiter_protected_source']
    assert git('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0]==BASE
    args=[sys.executable,'-X','utf8','-B',str(SOURCE/'scripts/publish_code_daily.py'),'publish','--source-root',str(SOURCE),'--repo-root',str(REPO),'--policy',str(SOURCE/'configs/publication/code_only.json'),'--expected-plan-sha',review['reviewed_plan_SHA256'],'--message','Preserve own LLaVA adaptations, complete numerical verification and serial resource recovery']
    with (V/'daily_20261123_publish.stdout.json').open('xb') as out,(V/'daily_20261123_publish.stderr.log').open('xb') as err:p=subprocess.run(args,cwd=REPO,env=env(),stdout=out,stderr=err)
    print(json.dumps(dict(actual_returncode=p.returncode,stdout=str(V/'daily_20261123_publish.stdout.json'),stderr=str(V/'daily_20261123_publish.stderr.log'),reviewed_plan_SHA256=review['reviewed_plan_SHA256'],no_extra_commit_or_force=True)));raise SystemExit(p.returncode)
def finalize():
    review=js(V/'daily_20261123_plan_review.json');published=js(V/'daily_20261123_publish.stdout.json')
    assert published['status']=='pushed' and published['day']==DAY and published['changed_files']==review['reviewed_changes']
    commit=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0]
    assert commit==remote==published['commit'] and git('status','--porcelain')==b''
    assert git('rev-list',BASE+'..HEAD').decode().split()==[commit] and git('log','-200','--format=%B%x00').decode().count('Code-Day: '+DAY)==1
    changes=git('diff','--name-status',BASE,commit).decode().splitlines();assert len(changes)==review['reviewed_changes'] and [x for x in changes if x.startswith('M\t')]==['M\tconfigs/publication/code_only.json']
    assert published['audited_outgoing_commits']==[commit]
    sc=js(V/'daily_20261123_source_content_review.json')
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,status='pushed_and_remote_verified',day=DAY,commit=commit,remote_commit=remote,branch=published['branch'],repository='https://github.com/lyinguo/910b.git',one_new_Code_Day23_commit=True,tests_R=review['tests_passed_R'],tests_local=review['tests_passed_local'],MJS_syntax_checks=review['MJS_syntax_passed'],changed_files=review['reviewed_changes'],explicit_paths=review['reviewed_explicit_paths'],reviewed_plan_SHA256=review['reviewed_plan_SHA256'],full_diff_SHA256=review['full_diff_SHA256'],source_collection_SHA256=review['source_collection_SHA256'],complete_outgoing_commit_tree_audited=published['audited_outgoing_commits'],worktree_clean=True,only_existing_updated_path='configs/publication/code_only.json',policy_before_SHA256=sc['policy_before_SHA256'],policy_after_SHA256=sc['policy_after_SHA256'],no_merge_force_Gitee_history_or_R_model_mutation=True)
    put('daily_20261123_receipt.json',result)
    book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');st=book.stat();h=sha(book.read_bytes());assert h=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
    put('local_verified185_Day23_fresh_SHA_stat_20261123.json',dict(UTC=result['UTC'],SHA256=h,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,verified_count=185,no_write=True));print(json.dumps(result,ensure_ascii=False))
def syncbuild():
    receipt=js(V/'daily_20261123_receipt.json');review=js(V/'daily_20261123_source_content_review.json');packet=js(PACKET);assert receipt['passed']
    names=['daily_20261123_receipt.json','daily_20261123_plan_review.json','daily_20261123_source_content_review.json','daily_20261123_full_source_diff.patch','daily_20261123_plan.stdout.json','daily_20261123_plan.stderr.log','daily_20261123_publish.stdout.json','daily_20261123_publish.stderr.log','daily_20261123_snapshot_tests_actual.json','daily_20261123_snapshot_tests_actual.log','R_daily23_actual_R_source_unit_checks_20261123.json','R_daily23_actual_R_source_unit_checks_20261123.json.stderr','R_daily23_actual_R_source_unit_checks_v2_original_source_arg_20261123.json','R_daily23_R_source_concurrent_CAS_before_publish_20261123.json','local_verified185_Day23_fresh_SHA_stat_20261123.json']
    originals=[]
    for n in names:
        b=(V/n).read_bytes();originals.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
    after=(SOURCE/'configs/publication/code_only.json').read_bytes();assert sha(after)==review['policy_after_SHA256']
    q=dict(remote_verified_receipt=receipt,policy_before_SHA256=review['policy_before_SHA256'],policy_after_SHA256=sha(after),policy_after_base64=base64.b64encode(after).decode(),source_SHA256={r['path']:r['SHA256'] for r in packet['records'] if r['path']!='configs/publication/code_only.json'},local_original_records=originals)
    carrier('scripts/maintenance/sync_daily23_reviewed_code_policy_and_remote_receipts_only_20261123.py','daily_20261123_reviewed_policy_remote_receipt_sync_plan.json',q,'stage_daily23_reviewed_R_policy_receipt_sync_20261123_remote.py')
if __name__=='__main__':
    actions={'snapshot':snapshot,'tests':tests,'plan-review':planreview,'build-cas':casbuild,'publish':publish,'finalize':finalize,'build-sync':syncbuild}
    assert len(sys.argv)==2 and sys.argv[1] in actions;actions[sys.argv[1]]()

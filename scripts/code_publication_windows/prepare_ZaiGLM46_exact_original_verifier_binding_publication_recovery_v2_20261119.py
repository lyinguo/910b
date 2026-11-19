"""Preserve the failed publisher and derive a private exact-verifier recovery."""
from pathlib import Path
import ast, base64, hashlib, json
P = Path(__file__).resolve().parent
old = 'publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py'
new = 'publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_actual_verifier_20261119.py'
oldhash = 'f245f5bd16476b341d9330877aa750cf98fb6a8d0da7f808c35f9163a551b273'
wrong = '22d2919d3bca51c68a6f3de45141ed55999d781aefccb9b22c18affc79e61a1d'
actual = '148afbeee4860b7a980608ae53b933b81afea6173cfa40aad635ce80f21cb91b'
sha = lambda b: hashlib.sha256(b).hexdigest()
body = (P/old).read_bytes(); assert sha(body) == oldhash
source = body.decode('utf8'); assert source.count(wrong) == 1
fixed = source.replace(wrong, actual)
assert ast.dump(ast.parse(fixed.replace(actual, wrong))) == ast.dump(ast.parse(source))
for code in (source, fixed):
    ast.parse(code)
proof = json.loads((P/'R_ZaiGLM46_completed200_exact_publication_inputs_20261119.json').read_bytes())['formal_proof']
assert proof['independent_verifier_SHA256'] == actual and proof['passed'] and proof['sample_count'] == 200
tree = ast.parse(fixed)
guard = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert) and "proof['independent_verifier_SHA256']" in ast.unparse(n.test))
head = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'full_numeric_head_verified')
scope = {}; exec(compile(ast.Module(body=[head], type_ignores=[]), '<full-head-guard>', 'exec'), scope)
def accepts(p):
    try: exec(compile(ast.Module(body=[guard], type_ignores=[]), '<exact-verifier-guard>', 'exec'), dict(scope, proof=p))
    except AssertionError: return False
    return True
assert accepts(proof)
for k, v in [('independent_verifier_SHA256',wrong),('independent_verifier_SHA256','0'*64),('source_tensor_count',703),('original_unused_MTP_source_tensor_count',1),('own_original_BF16_preserved',False),('numeric_sample_checks',[])]:
    bad=dict(proof);bad[k]=v;assert not accepts(bad),(k,v)
assets = {'scripts/'+new: fixed.encode()}
oldwrap = 'run_ZaiGLM46_verified181_reviewed_dual_publication_actual_20261119.py'
newwrap = 'run_ZaiGLM46_verified181_reviewed_dual_publication_actual_v2_exact_verifier_20261119.py'
oldrun = 'ZaiGLM46Flash_verified181_publication_review_inputs_v2_20261119'
newrun = 'ZaiGLM46Flash_verified181_publication_review_inputs_v3_exact_verifier_20261119'
wrap = (P/oldwrap).read_text(encoding='utf8').replace(old,new).replace(oldhash,sha(fixed.encode())).replace(oldrun,newrun)
assets['scripts/'+newwrap] = wrap.encode()
reader = (P/'read_ZaiGLM46_verified181_server_book_for_local_CAS_20261119_remote.py').read_text(encoding='utf8').replace(old,new).replace(oldhash,sha(fixed.encode()))
readername = 'read_ZaiGLM46_verified181_server_book_for_local_CAS_v2_actual_verifier_20261119_remote.py'
assets['scripts/code_publication_windows/'+readername] = reader.encode()
closename = 'close_alignment_handoff_exact_ZaiGLM46_qualified181_dual_publication_only_v10_actual_verifier_20261119.py'
close = (P/'close_alignment_handoff_exact_ZaiGLM46_qualified181_dual_publication_only_v9_20261119.py').read_text(encoding='utf8').replace(old,new).replace(oldhash,sha(fixed.encode())).replace(oldrun,newrun).replace('closure_actual_receipt_v9_exact_qualified_publication','closure_actual_receipt_v10_exact_qualified_publication')
assets['scripts/maintenance/'+closename] = close.encode()
for n, raw in assets.items():
    ast.parse(raw); p=P/Path(n).name;assert not p.exists();p.write_bytes(raw)
review = (P/'ZaiGLM46Flash_local_result_sync_verified181_20261119/reviewed_inputs.json').read_bytes()
payload = dict(assets={n:dict(base64=base64.b64encode(v).decode(),SHA256=sha(v)) for n,v in assets.items()}, review=base64.b64encode(review).decode(), review_SHA256=sha(review), publisher_SHA256=sha(fixed.encode()), wrapper_SHA256=sha(wrap.encode()), newrun=newrun, wrapper='scripts/'+newwrap)
script = 'DATA='+repr(payload)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=B/'ZaiGLM46Flash_verified181_publication_review_inputs_v2_20261119';failure=json.loads((old/'publication_actual_completion.json').read_bytes());assert failure['actual_returncode']==1 and failure['source_SHA256']=='f245f5bd16476b341d9330877aa750cf98fb6a8d0da7f808c35f9163a551b273'
assert 'line 257' in (old/'publication_stderr.log').read_text() and 'AssertionError' in (old/'publication_stderr.log').read_text()
assert not (B/'ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119').exists()
assert sha(R/'scripts/publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py')==failure['source_SHA256']
assert sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119';job=json.loads((J/'job.json').read_bytes());assert sha(J/'job.json')=='18c9d24f9992311ea3261e82cc56c3121db03ccd0655e21aaafc4d2b5ee4f323'
nf=job['spec']['verifier'];assert sha(R/nf)==job['spec']['source_sha256'][nf]=='148afbeee4860b7a980608ae53b933b81afea6173cfa40aad635ce80f21cb91b'
for n,v in DATA['assets'].items():
 p=R/n;raw=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
I=B/DATA['newrun'];I.mkdir(exist_ok=False);raw=base64.b64decode(DATA['review'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['review_SHA256'];(I/'reviewed_inputs.json').write_bytes(raw)
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(R/DATA['wrapper']),DATA['review_SHA256']],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,publisher_SHA256=DATA['publisher_SHA256'],wrapper_SHA256=DATA['wrapper_SHA256'],prior_failed_publisher_and_logs_preserved=True,only_exact_original_verifier_SHA_literal_corrected=True,all_qualification_source_numerics_and_frozen_writer_guards_preserved=True,no_NN_replay=True,started_not_claimed_complete=True);(I/'background_launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(v,ensure_ascii=False))
'''
ast.parse(script)
launcher=P/'launch_ZaiGLM46_exact_verifier_binding_publication_recovery_v2_20261119_remote.py';assert not launcher.exists();launcher.write_text(script,encoding='utf8',newline='\n')
result=dict(passed=True,only_publisher_semantic_change_is_exact_original_verifier_SHA_literal=True,actual_original_verifier_SHA256=actual,original_publisher_SHA256=oldhash,corrected_publisher_SHA256=sha(fixed.encode()),positive_checks=1,negative_checks=6,all_other_publisher_AST_and_original_full_numeric_head_guards_identical=True,assets=[dict(relative=n,SHA256=sha(v)) for n,v in assets.items()],no_NN_or_G_in_local_preparation=True)
(P/'ZaiGLM46_exact_original_verifier_publication_recovery_scope_tests_20261119.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))

from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;H=P/'AutoGLMPhone9B_local_result_sync_verified182_20261120';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=json.loads((H/'plan.json').read_bytes());assert q['complete_numeric_verifier_passed'] and q['changes'][0]['new_real_NPU200'];assert sha(q['target'])==q['expected_before_sha256']=='4e740da3981c17fd7b599875ab95e60c5f709804f08c22c9fb64c3c8a5454ed8'
visual=dict(UTC=datetime.now(timezone.utc).isoformat(),before_and_author_after_actually_viewed=True,before_SHA256=sha(H/'before_693.png'),after_SHA256=sha(H/'after_693.png'),observation='Actually viewed both PNGs: G693 empty before and 0.82 after; existing neighboring values and narrow General style preserved.')
(H/'actual_PNG_visual_review.json').write_text(json.dumps(visual,indent=2)+'\n',encoding='utf8')
values=json.loads((H/'artifact_authored_G_values.json').read_bytes());assert values[691][0]==q['changes'][0]['new_value']==.82
names=['before_693.png','after_693.png','artifact_authoring_candidate.xlsx','artifact_authored_G_values.json','formula_scan.ndjson','actual_PNG_visual_review.json'];q['artifact_tool_actual_authoring_evidence']={n:dict(path=str(H/n),SHA256=sha(H/n)) for n in names};raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode();(H/'reviewed_inputs.json').write_bytes(raw)
scope=json.loads((P/'AutoGLMPhone9B_verified182_actual_publication_source_scope_20261120.json').read_bytes());assets={}
for n,rel in [(scope['publisher'],'scripts/'+scope['publisher']),(scope['wrapper'],'scripts/'+scope['wrapper'])]:
 b=(P/n).read_bytes();ast.parse(b);assets[rel]=dict(base64=base64.b64encode(b).decode(),SHA256=hashlib.sha256(b).hexdigest())
data=dict(assets=assets,review_base64=base64.b64encode(raw).decode(),review_SHA256=hashlib.sha256(raw).hexdigest(),home=scope['supervisor_home'],wrapper='scripts/'+scope['wrapper'],publisher_SHA256=scope['publisher_SHA256'])
s='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]
assert hashlib.sha256(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx').read_bytes()).hexdigest()=='4e740da3981c17fd7b599875ab95e60c5f709804f08c22c9fb64c3c8a5454ed8'
for n,v in DATA['assets'].items():
 raw=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/DATA['home'];I.mkdir(exist_ok=False);raw=base64.b64decode(DATA['review_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['review_SHA256'];(I/'reviewed_inputs.json').write_bytes(raw)
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(R/DATA['wrapper']),DATA['review_SHA256']],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,publisher_SHA256=DATA['publisher_SHA256'],reviewed_inputs_SHA256=DATA['review_SHA256'],started_not_claimed_complete=True);(I/'background_launch.json').write_text(json.dumps(v,indent=2)+'\\n',encoding='utf8');print(json.dumps(v))
'''
ast.parse(s);(P/'launch_AutoGLMPhone_verified182_actual_reviewed_single_lock_dual_20261120_remote.py').write_text(s,encoding='utf8',newline='\n');print(json.dumps(dict(review_SHA256=data['review_SHA256'],publisher_SHA256=scope['publisher_SHA256'],no_production_write_in_builder=True)))

"""Bind actual authoring and reviewed PNGs, then start the exact one-lock transaction."""
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).resolve().parent;H=P/'HuihuiGLM46Flash_local_result_sync_verified181_20261119';q=json.loads((H/'plan.json').read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert q['complete_numeric_verifier_passed'] and q['changes'][0]['new_real_NPU200']
visual=json.loads((H/'actual_PNG_visual_review.json').read_bytes());assert visual['before_and_author_after_actually_viewed'] and visual['before_SHA256']==sha(H/'before_355.png') and visual['after_SHA256']==sha(H/'after_355.png')
values=json.loads((H/'artifact_authored_G_values.json').read_bytes());assert values[353][0]==q['changes'][0]['new_value']
names=['before_355.png','after_355.png','artifact_authoring_candidate.xlsx','artifact_authored_G_values.json','formula_scan.ndjson','actual_PNG_visual_review.json']
q['artifact_tool_actual_authoring_evidence']={n:dict(path=str(H/n),SHA256=sha(H/n)) for n in names}
raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode();(H/'reviewed_inputs.json').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
wrapper=(P/'run_Huihui_verified181_reviewed_dual_publication_actual_20261119.py').read_bytes();ast.parse(wrapper)
DATA=dict(review_base64=base64.b64encode(raw).decode(),review_SHA256=digest,wrapper_base64=base64.b64encode(wrapper).decode(),wrapper_SHA256=hashlib.sha256(wrapper).hexdigest())
s='DATA='+repr(DATA)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]
I=B/'HuihuiGLM46Flash_verified181_publication_review_inputs_v2_20261119';I.mkdir(exist_ok=False)
raw=base64.b64decode(DATA['review_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['review_SHA256'];(I/'reviewed_inputs.json').write_bytes(raw)
source=R/'scripts/run_Huihui_verified181_reviewed_dual_publication_actual_20261119.py';body=base64.b64decode(DATA['wrapper_base64'],validate=True);assert hashlib.sha256(body).hexdigest()==DATA['wrapper_SHA256'];ast.parse(body)
if source.exists():assert source.read_bytes()==body
else:source.write_bytes(body)
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(source),DATA['review_SHA256']],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,source=str(source),source_SHA256=DATA['wrapper_SHA256'],reviewed_inputs_SHA256=DATA['review_SHA256'],publication_started_not_claimed_complete=True);(I/'background_launch.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf8');print(json.dumps(v,ensure_ascii=False))
'''
ast.parse(s);(P/'launch_Huihui_verified181_actual_reviewed_publication_20261119_remote.py').write_text(s,encoding='utf8',newline='\n');print(json.dumps(dict(review_SHA256=digest,wrapper_SHA256=DATA['wrapper_SHA256'],no_production_write_yet=True)))

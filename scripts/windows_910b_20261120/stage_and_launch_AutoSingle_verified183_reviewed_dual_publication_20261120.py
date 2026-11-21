from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
scope=json.loads((V/'AutoGLMPhone9BSingle_verified183_publication_source_scope_20261120.json').read_bytes());H=Path(scope['local_home'])
q=json.loads((H/'plan.json').read_bytes());author=json.loads((H/'artifact_authored_G_values.json').read_bytes());assert author[690][0]==.795
review=dict(UTC=datetime.now(timezone.utc).isoformat(),before_and_author_after_actually_viewed=True,before_SHA256=sha((H/'before_692.png').read_bytes()),after_SHA256=sha((H/'after_692.png').read_bytes()),target='G692',actual_authored_value=.795,adjacent_G693_stays_point82=True,original_narrow_column_styles_and_all_non_G_content_preserved=True)
(H/'actual_PNG_visual_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
assert sha(Path(q['target']).read_bytes())==q['expected_before_sha256']=='0b2fcf21fe5a56e5b33111c4e65c2ec0ddf53dae43d451dec756a342df810340'
q.update(artifact_authored_G_values_SHA256=sha((H/'artifact_authored_G_values.json').read_bytes()),actual_PNG_visual_review_SHA256=sha((H/'actual_PNG_visual_review.json').read_bytes()))
raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode();(H/'reviewed_inputs.json').write_bytes(raw)
assets=[]
for name in [scope['publisher'],scope['wrapper']]:
 b=(V/name).read_bytes();ast.parse(b);assets.append(dict(relative='scripts/'+name,SHA256=sha(b),base64=base64.b64encode(b).decode()))
assert assets[0]['SHA256']==scope['publisher_SHA256'] and assets[1]['SHA256']==scope['wrapper_SHA256']
data=dict(assets=assets,reviewed_inputs_base64=base64.b64encode(raw).decode(),reviewed_inputs_SHA256=sha(raw),supervisor_home=scope['supervisor_home'],wrapper=scope['wrapper'])
code='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
for v in DATA['assets']:
 p=R/v['relative'];b=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==v['SHA256'] and p.resolve().is_relative_to(R);ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
H=B/DATA['supervisor_home'];H.mkdir(exist_ok=False);raw=base64.b64decode(DATA['reviewed_inputs_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['reviewed_inputs_SHA256'];(H/'reviewed_inputs.json').write_bytes(raw)
with (H/'launcher_stdout.log').open('xb') as out,(H/'launcher_stderr.log').open('xb') as err:
 c=subprocess.Popen(['python3','-B',str(R/'scripts'/DATA['wrapper']),DATA['reviewed_inputs_SHA256']],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=c.pid,reviewed_inputs_SHA256=DATA['reviewed_inputs_SHA256'],publication_supervisor=str(H),only_qualified_single_G692_may_change_after_all_guards=True,no_model_replay=True);(H/'background_launch.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''
ast.parse(code);(V/'launch_AutoSingle_verified183_reviewed_dual_publication_20261120_remote.py').write_text(code,encoding='utf-8',newline='\n')
print(json.dumps(dict(reviewed_inputs_SHA256=sha(raw),publisher_SHA256=scope['publisher_SHA256'],only_actual_verified200_score=.795,no_NN_replay=True)))

from pathlib import Path
import ast,hashlib,base64,json
P=Path(__file__).resolve().parent
rows=[]
for name,rel in [('glm46_complete_shape_component_capacity_20261119.py','scripts/glm46_complete_shape_component_capacity_20261119.py'),('test_GLM46_complete_shape_component_capacity_20261119.py','scripts/tests/test_GLM46_complete_shape_component_capacity_20261119.py'),('review_glm46_completed3_LZMA_complete200_component_shapes_no_NN_20261119.py','scripts/review_glm46_completed3_LZMA_complete200_component_shapes_no_NN_20261119.py'),('publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py','scripts/publish_HuihuiGLM46Flash_verified181_single_original_lock_dual_transaction_v2_next182_metadata_20261119.py')]:
 b=(P/name).read_bytes();ast.parse(b);rows.append(dict(relative=rel,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
remote='FILES='+repr(rows)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'GLM46Flash_complete_component_capacity_review_launch_20261119';I.mkdir(exist_ok=False)
for row in FILES:
 p=R/row['relative'];b=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R);ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
test=subprocess.run(['python3','-B',str(R/'scripts/tests/test_GLM46_complete_shape_component_capacity_20261119.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));(I/'scope_test_stdout.json').write_text(test.stdout,encoding='utf8');(I/'scope_test_stderr.log').write_text(test.stderr,encoding='utf8');assert test.returncode==0,test.stderr
with (I/'review_stdout.json').open('xb') as out,(I/'review_stderr.log').open('xb') as err:
 result=subprocess.run(['python3','-B',str(R/'scripts/review_glm46_completed3_LZMA_complete200_component_shapes_no_NN_20261119.py')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps(dict(actual_returncode=result.returncode,sources={x['relative']:x['SHA256'] for x in FILES},review_stderr=(I/'review_stderr.log').read_text()[-7000:],review_stdout=(I/'review_stdout.json').read_text()[-15000:]),ensure_ascii=False));raise SystemExit(result.returncode)
'''
ast.parse(remote);(P/'deploy_run_GLM46_complete_component_capacity_review_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n');print(json.dumps({x['relative']:x['SHA256'] for x in rows}))

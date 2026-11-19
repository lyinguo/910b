"""Use the recorded f1..fN capture convention; the first is actual prefill."""
from pathlib import Path
import ast,hashlib,base64,json
P=Path(__file__).resolve().parent
u=(P/'glm46_complete_shape_component_capacity_v2_actual_operands_20261119.py').read_text(encoding='utf8')
u=u.replace("assert 1<=int(match[1])<forwards","assert 1<=int(match[1])<=forwards\n  if int(match[1])==1 and shape.count(input_tokens)==1:return 'actual_first_forward_prefill_position_axes',ratios['prefill']",1)
uname='glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py';(P/uname).write_text(u,encoding='utf8',newline='\n')
t=(P/'test_GLM46_complete_component_capacity_v2_actual_operands_20261119.py').read_text(encoding='utf8').replace('glm46_complete_shape_component_capacity_v2_actual_operands_20261119.py',uname)
t=t.replace("assert m.component('f1_attn_position_embeddings_0_0',[3,1,300,128],300,195,r)[0]=='sequence_dependent_decode_operand_original_maximum_retained'","assert m.component('f1_attn_position_embeddings_0_0',[3,1,300,128],300,195,r)[1]==r['prefill']\nassert m.component('f195_attn_position_embeddings_0_0',[3,1,1,128],300,195,r)[1]==r['samples']\nassert m.component('f2_attn_position_embeddings_0_0',[3,1,999,128],300,195,r)[0]=='sequence_dependent_decode_operand_original_maximum_retained'")
t=t.replace('checks=10','checks=12');tname='test_GLM46_component_capacity_v3_actual_one_based_20261119.py';(P/tname).write_text(t,encoding='utf8',newline='\n')
s=(P/'review_glm46_completed3_LZMA_complete200_component_shapes_v2_actual_operands_no_NN_20261119.py').read_text(encoding='utf8').replace('from glm46_complete_shape_component_capacity_v2_actual_operands_20261119 import','from glm46_complete_shape_component_capacity_v3_actual_one_based_20261119 import').replace('capacity_review_v2_actual_operands_20261119','capacity_review_v3_actual_one_based_20261119')
sname='review_glm46_completed3_LZMA_complete200_component_shapes_v3_actual_one_based_no_NN_20261119.py';(P/sname).write_text(s,encoding='utf8',newline='\n')
rows=[]
for n,relative in [(uname,'scripts/'+uname),(tname,'scripts/tests/'+tname),(sname,'scripts/'+sname)]:
 b=(P/n).read_bytes();ast.parse(b);rows.append(dict(relative=relative,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
remote='FILES='+repr(rows)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'GLM46Flash_complete_component_capacity_review_v3_actual_one_based_launch_20261119';I.mkdir(exist_ok=False)
for row in FILES:
 p=R/row['relative'];b=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R);ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
test=subprocess.run(['python3','-B',str(R/FILES[1]['relative'])],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));(I/'scope_test_stdout.json').write_text(test.stdout,encoding='utf8');(I/'scope_test_stderr.log').write_text(test.stderr,encoding='utf8');assert test.returncode==0,test.stderr
with (I/'review_stdout.json').open('xb') as out,(I/'review_stderr.log').open('xb') as err:
 result=subprocess.run(['python3','-B',str(R/FILES[2]['relative'])],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps(dict(actual_returncode=result.returncode,sources={x['relative']:x['SHA256'] for x in FILES},review_stderr=(I/'review_stderr.log').read_text()[-7000:],review_stdout=(I/'review_stdout.json').read_text()[-15000:]),ensure_ascii=False));raise SystemExit(result.returncode)
'''
ast.parse(remote);(P/'deploy_run_GLM46_component_capacity_review_v3_actual_one_based_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n');print(json.dumps({x['relative']:x['SHA256'] for x in rows}))

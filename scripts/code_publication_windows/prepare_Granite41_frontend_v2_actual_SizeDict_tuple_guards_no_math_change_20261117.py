import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
s=(P/'Granite41_primary_prepare_granite41_own_original_HF_BF16_4096_frontend200_META_v1_20261117.py').read_text(encoding='utf8')
s=s.replace('Granite41_own_original_HF_BF16_4096_frontend200_source_META_v1_20261117','Granite41_own_original_HF_BF16_4096_frontend200_source_META_v2_actual_SizeDict_tuple_20261117')
old="assert ip.size==dict(height=384,width=384) and ip.crop_size==dict(height=384,width=384) and ip.image_mean==ip.image_std==[.5,.5,.5]"
new="assert (ip.size.height,ip.size.width)==(384,384) and (ip.crop_size.height,ip.crop_size.width)==(384,384) and tuple(ip.image_mean)==tuple(ip.image_std)==(.5,.5,.5)"
assert old in s;s=s.replace(old,new)
ast.parse(s);name='scripts/prepare_granite41_own_original_HF_BF16_4096_frontend200_META_v2_actual_SizeDict_tuple_20261117.py';(P/Path(name).name).write_bytes(s.encode());h=hashlib.sha256(s.encode()).hexdigest()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';name=__NAME__;raw=base64.b64decode(__DATA__);assert hashlib.sha256(raw).hexdigest()==__SHA__;ast.parse(raw);p=R/name
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
I=B/'Granite41_frontend_META_v2_actual_SizeDict_tuple_guard_no_math_change_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:child=subprocess.Popen(['python3','-B',name],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=name,source_SHA256=__SHA__,only_actual_SizeDict_tuple_schema_guards_changed=True,no_pixel_token_model_math_or_original_parameter_changed=True,no_NPU_or_learned_CPU_model_forward=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__NAME__',repr(name)).replace('__DATA__',repr(base64.b64encode(s.encode()).decode())).replace('__SHA__',repr(h));ast.parse(remote);(P/'deploy_launch_Granite41_frontend_v2_actual_SizeDict_tuple_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(source=name,SHA256=h)))

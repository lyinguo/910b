from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;name='prepare_autoglm_phone9b_own_AFS_R_identical_fixed200_no_filter_input_alias_20261119.py';raw=(P/name).read_bytes();ast.parse(raw);data=dict(name=name,body=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest())
s='DATA='+repr(data)+'\n'+r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,os,subprocess
R=Path('/root/projects/comperror');p=R/'scripts'/DATA['name'];b=base64.b64decode(DATA['body'],validate=True);assert hashlib.sha256(b).hexdigest()==DATA['SHA256'];ast.parse(b);assert not p.exists();p.write_bytes(b);subprocess.run(['python3','-B',str(p)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
'''
ast.parse(s);(P/'execute_AutoGLM_own_fixed_input_alias_no_NN_20261119_remote.py').write_text(s,encoding='utf8');print(data['SHA256'])

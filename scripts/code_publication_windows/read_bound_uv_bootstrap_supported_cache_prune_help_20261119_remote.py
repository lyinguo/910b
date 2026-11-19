import sys
sys.dont_write_bytecode=True
import subprocess,json,hashlib
from pathlib import Path
uv=Path('/root/.venvs/uv-bootstrap-910b/bin/uv');out={'uv':str(uv),'uv_SHA256':hashlib.file_digest(uv.open('rb'),'sha256').hexdigest()}
for key,args in [('version',['--version']),('prune_help',['cache','prune','--help'])]:
 p=subprocess.run([str(uv),*args],capture_output=True,text=True,timeout=20);assert p.returncode==0;out[key]=p.stdout
print(json.dumps(out,ensure_ascii=False))

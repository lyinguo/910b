import sys
sys.dont_write_bytecode=True
from pathlib import Path
import shutil,subprocess,json
uv=shutil.which('uv');assert uv
out={}
for key,argv in [('version',[uv,'--version']),('cache_prune_help',[uv,'cache','prune','--help'])]:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=20);assert p.returncode==0;out[key]=dict(argv=argv,stdout=p.stdout,stderr=p.stderr)
out['uv_path']=uv;out['cache_root_children']=[p.name for p in Path('/root/.cache/uv').iterdir()];print(json.dumps(out,ensure_ascii=False))

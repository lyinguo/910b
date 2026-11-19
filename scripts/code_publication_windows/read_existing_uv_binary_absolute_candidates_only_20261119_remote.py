import sys
sys.dont_write_bytecode=True
from pathlib import Path
import shutil,subprocess,json,importlib.util
names=['/root/.local/bin/uv','/usr/local/bin/uv','/usr/bin/uv','/root/.cargo/bin/uv','/opt/uv/bin/uv','/bin/uv','/root/projects/LYG/open-webui/.venv/bin/uv']
found=[p for p in names if Path(p).is_file()];spec=importlib.util.find_spec('uv')
if spec and spec.origin:
 base=Path(spec.origin).parent
 found.extend(str(p) for p in base.rglob('uv') if p.is_file())
out=dict(candidates=names,existing=found,installed_python_uv_module_origin=spec.origin if spec else None,version_and_help=[])
for p in found:
 a=subprocess.run([p,'--version'],capture_output=True,text=True,timeout=20)
 if a.returncode==0:
  b=subprocess.run([p,'cache','prune','--help'],capture_output=True,text=True,timeout=20);out['version_and_help'].append(dict(path=p,version=a.stdout,help=b.stdout,help_rc=b.returncode));break
print(json.dumps(out,ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,contextlib,io,subprocess
R=Path('/root/projects/comperror');p=R/'scripts/select_existing_original_resource_queue_1245_20261117.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494';sp=importlib.util.spec_from_file_location('original_selector_readonly_gate',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);assert hashlib.sha256(m.ORIGINAL_RESOURCE_GATE).hexdigest()==m.ORIGINAL_RESOURCE_GATE_SHA256
s=io.StringIO()
with contextlib.redirect_stdout(s):exec(compile(m.ORIGINAL_RESOURCE_GATE,'original_frozen_resource_metadata_readonly','exec'),{})
r=subprocess.run(['du','-x','--max-depth=1',str(R/'workspace')],capture_output=True,text=True,timeout=60)
print(json.dumps({'original_frozen_read_only_resource_gate':json.loads(s.getvalue()),'workspace_directory_usage_kib':r.stdout,'du_returncode':r.returncode,'du_stderr':r.stderr,'no_dispatch_no_cleanup':True}))
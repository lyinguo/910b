import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'after_GLM46_actual200_qualified_dual_commit_or_preserved_gap_AutoGLMPhone_serial_20261119';L=B/'after_GLM46_actual200_AutoGLM_serial_queue_launch_evidence_20261119';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
js=lambda p:json.loads(Path(p).read_bytes())
out=dict(UTC=datetime.now(timezone.utc).isoformat(),launch=js(L/'launch.json'),state=js(H/'state.json') if (H/'state.json').exists() else None,completion=js(H/'actual_completion.json') if (H/'actual_completion.json').exists() else None,kernel_FLOCK=runner_lock_owners(),full_queue_stderr=(L/'queue_stderr.log').read_text(),full_queue_stdout=(L/'queue_stdout.log').read_text(),new_NN_or_G_claimed=False)
pid=out['launch']['PID'];p=Path('/proc')/str(pid)
if p.exists():
 args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x];out['actual_process']=dict(PID=pid,argv=args,cwd=os.readlink(p/'cwd'),status=(p/'status').read_text(),stat=(p/'stat').read_text());source=R/out['launch']['source'];out['fresh_source_SHA256']=hashlib.sha256(source.read_bytes()).hexdigest();assert out['fresh_source_SHA256']==out['launch']['source_SHA256'] and str(source) in args
else:out['actual_process']=None
print(json.dumps(out,ensure_ascii=False))

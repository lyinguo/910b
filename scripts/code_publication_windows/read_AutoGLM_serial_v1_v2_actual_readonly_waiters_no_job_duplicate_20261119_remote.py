import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def maybe(p):return json.loads(p.read_bytes()) if p.is_file() else None
out={}
for version,h,l,filename in [('v1','after_GLM46_actual200_qualified_dual_commit_or_preserved_gap_AutoGLMPhone_serial_20261119','after_GLM46_actual200_AutoGLM_serial_queue_launch_evidence_20261119','queue_'),('v2','after_GLM46_AutoGLM_v1_preNN_config_refusal_own_fixed_input_set_serial_v2_20261119','AutoGLM_known_preNN_fixed_input_set_recovery_serial_v2_launch_20261119','')]:
 H=B/h;L=B/l;launch=maybe(L/'launch.json');pid=launch['PID'];p=Path('/proc')/str(pid);actual=None
 if p.exists():
  args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x];actual=dict(PID=pid,argv=args,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text());assert str(R/launch['source']) in args
 src=R/launch['source'];assert hashlib.sha256(src.read_bytes()).hexdigest()==launch['source_SHA256']
 out[version]=dict(actual=actual,launch=launch,state=maybe(H/'state.json'),completion=maybe(H/'actual_completion.json'),full_stderr=(L/(filename+'stderr.log')).read_text(),full_stdout=(L/(filename+'stdout.log')).read_text())
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),read_only_queues=out,own_v1_NPU_job_exists=(B/'AutoGLMPhone9B_own_original_BF16_2048_true3_v1_real_FLOCK_20261119').exists(),own_v2_NPU_job_exists=(B/'AutoGLMPhone9B_own_original_BF16_2048_true3_v2_fixed_input_set_real_FLOCK_20261119').exists(),no_NEW_model_or_G_or_signal_operation=True),ensure_ascii=False))

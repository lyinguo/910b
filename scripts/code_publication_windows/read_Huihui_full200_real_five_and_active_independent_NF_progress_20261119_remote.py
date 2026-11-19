import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
state=json.loads((J/'state.json').read_bytes());folder=R/'workspace/reports/huihui-ai_Huihui-GLM-4.6V-Flash-abliterated/image-text-to-text/20261119_152641';report=json.loads((folder/'eval_report.json').read_bytes());line=json.loads((folder/'operator_lineage.json').read_bytes())['actual_execution'];rows=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()]
pid=state.get('child_PID');p=Path('/proc')/str(pid);actual=dict(PID=pid,exists=p.exists())
if p.exists():
 actual.update(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text())
 fds=[]
 for fd in (p/'fd').iterdir():
  try:
   name=os.readlink(fd)
   if any(s in name for s in ['sample_','safetensors','Huihui','glm46']):fds.append(name)
  except OSError:pass
 actual['relevant_actual_open_FDs']=fds
err=J/'formal200/verification_stderr.log';out=J/'formal200/verification_stdout.json'
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),job_state=state,full200_reports=list(x.name for x in folder.iterdir()),actual_samples=len(rows),committed_actual_forwards=sum(x['actual_native_forwards'] for x in rows),raw_model_metrics_not_yet_final_accepted=report['metrics'],statistics=report['statistics'],active_verifier=actual,NF_stdout_bytes=out.stat().st_size if out.exists() else None,NF_stderr=err.read_text(errors='replace')[-8000:] if err.exists() else None,full_NF_completion_exists=(J/'completion.json').exists(),no_G_publication_claim=True),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119';st=json.loads((J/'state.json').read_bytes());pid=st.get('child_PID');p=Path('/proc')/str(pid);actual=None
if pid and p.exists():
 args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x];fds={}
 for f in (p/'fd').iterdir():
  try:v=os.readlink(f)
  except FileNotFoundError:continue
  if '/workspace/' in v or '/mnt/afs/' in v:fds[f.name]=v
 actual=dict(PID=pid,argv=args,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),status=(p/'status').read_text(),io=(p/'io').read_text(),actual_open_evidence_FDs=fds)
folder=Path(st['report_dir']);report=json.loads((folder/'eval_report.json').read_bytes());summary=dict(primary_metric=report['summary']['primary_metric'],metrics=report['metrics'],statistics=report['statistics'],sample_range=report['pipeline_config']['sample_range'],actual_input_output_details_count=len(report['detailed_results']))
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),original_runner_state=st,actual_child_process=actual,original_FLOCK=runner_lock_owners(),model_report_alone_not_yet_independent_accepted=summary,final_job_completion=json.loads((J/'completion.json').read_bytes()) if (J/'completion.json').is_file() else None,no_NN_replay_or_verifier_source_or_deadline_change=True),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,subprocess,os,shutil
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119';F=B/'after_Huihui_real_dual181_then_GLM46_own_completed3_component_capacity_serial_v1_20261119';HC=B/'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119'
def maybe(p):return json.loads(p.read_bytes()) if p.exists() else None
t=subprocess.run(['python3','-B',str(R/'scripts/tests/test_Huihui_completed200_baselinegap_selective_retention_wrong_controls_20261119.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=R)
print(json.dumps(dict(test_returncode=t.returncode,test_stdout=t.stdout.decode(),test_stderr=t.stderr.decode(),Huihui_completion=maybe(J/'completion.json'),Huihui_controller_completion=maybe(HC/'actual_completion.json'),serial_next_completion=maybe(F/'actual_completion.json'),actual_procs=project_processes(),actual_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))

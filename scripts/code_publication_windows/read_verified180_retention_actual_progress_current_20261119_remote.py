import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
out={n:json.loads((H/n).read_bytes()) if (H/n).is_file() else None for n in ['progress.json','actual_completion.json']}
out.update(UTC=datetime.now(timezone.utc).isoformat(),plan_exists=(H/'plan.json').exists(),free_bytes=shutil.disk_usage(R).free,processes=project_processes(),original_FLOCK=runner_lock_owners());print(json.dumps(out,ensure_ascii=False))

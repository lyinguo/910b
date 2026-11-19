import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess
R = Path('/root/projects/comperror')
B = R / 'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
I = B / 'ZaiGLM46Flash_verified181_publication_review_inputs_v2_20261119'
source = R / 'scripts/publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py'
review = I / 'reviewed_inputs.json'
assert len(sys.argv) == 2 and hashlib.sha256(source.read_bytes()).hexdigest() == 'f245f5bd16476b341d9330877aa750cf98fb6a8d0da7f808c35f9163a551b273' and (hashlib.sha256(review.read_bytes()).hexdigest() == sys.argv[1])
assert not (I / 'publication_actual_completion.json').exists()
start = datetime.now(timezone.utc).isoformat()
with (I / 'publication_stdout.json').open('xb') as out, (I / 'publication_stderr.log').open('xb') as err:
    child = subprocess.run(['python3', '-B', str(source), str(review)], cwd=R, stdout=out, stderr=err, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
value = dict(started_UTC=start, completed_UTC=datetime.now(timezone.utc).isoformat(), PID=os.getpid(), actual_returncode=child.returncode, source=str(source), source_SHA256='f245f5bd16476b341d9330877aa750cf98fb6a8d0da7f808c35f9163a551b273', reviewed_inputs_SHA256=sys.argv[1], original_five_reports_model_sources_and_numeric_arrays_unmodified_by_publication=True)
(I / 'publication_actual_completion.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
raise SystemExit(child.returncode)

import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, json
R = Path('/root/projects/comperror')
D = R/'workspace/diagnostics/batch_910b'
B = D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J = B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
H = B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
paths = [J/'job.json', J/'state.json', J/'completion.json', H/'actual_completion.json', H/'own_passed3_complete_shape_capacity_review.json', B/'verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120/actual_completion.json']
job = json.loads((J/'job.json').read_bytes())
a = Path(job['spec']['alignment_approval_review'])
paths.append(a)
approval = json.loads(a.read_bytes())
paths += [Path(approval['true3_proof']), Path(approval['capacity_review']), J/'fresh3/verification_stdout.json', J/'formal200/verification_stdout.json']
out = {}
for p in dict.fromkeys(paths):
    if p.exists():
        b=p.read_bytes()
        out[str(p)]={'SHA256':hashlib.sha256(b).hexdigest(),'complete_text':b.decode('utf-8')}
    else:
        out[str(p)]={'exists':False}
print(json.dumps({'files':out},ensure_ascii=False))

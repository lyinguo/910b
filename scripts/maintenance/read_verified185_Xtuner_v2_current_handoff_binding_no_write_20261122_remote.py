import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';ptr=json.loads(p.read_bytes());q=Path(ptr['new_primary_pending']);h=Path(ptr['handoff_directory'])
print(json.dumps(dict(pointer=ptr,pointer_SHA256=sha(p),pending_SHA256=sha(q),closing_SHA256=sha(h/'closing_addendum.json'),canonical_fresh={n:sha(n) for n in ptr['canonical_SHA256']},runtime_jobs={str(j):json.loads(j.read_bytes()) for j in [B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122/job.json',B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122/job.json'] if j.exists()}),ensure_ascii=False))

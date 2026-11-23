"""Read complete exact frontend failures and input contracts, never replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
out={}
for h in sorted(B.glob('Kanana_own_original_BF16_dedicated_frontend200*')):
    rows={}
    for n in ['state.json','review.json','actual_completion.json','native_processor_and_original_generation_contract.json','prompt_reference_first_difference.json','installed_original_HF_image_transform_external_sources.json']:
        p=h/n
        if p.is_file():
            b=p.read_bytes();rows[n]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_value=json.loads(b))
    out[str(h)]=rows
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),all_exact_frontend_stage_records=out,no_NN_NF_or_mutation=True),ensure_ascii=False))

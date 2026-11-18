import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
F=B/'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117'
J=B/'Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117'
files={}
for name in ['review.json','readiness.json','actual_completion.json','own_source_headers.json','source_headers.json']:
 p=F/name
 if p.exists():files[name]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),value=json.loads(p.read_bytes()))
job=json.loads((J/'job.json').read_bytes())
binding=json.loads((D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json').read_bytes());frozen=Path(binding['local_A100_reference_snapshot']);assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba'
print(json.dumps(dict(own_frontend_files=[p.name for p in F.iterdir()],files=files,parent_job_structural_template_only=job,own_preserved_original_D=json.loads(frozen.read_bytes())['models']['microsoft/Phi-3.5-vision-instruct'],original_full_D_file=str(frozen)),ensure_ascii=False))

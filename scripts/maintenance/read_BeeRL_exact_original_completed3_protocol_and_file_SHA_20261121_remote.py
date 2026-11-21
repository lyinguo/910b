import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'Bee8BRL_own_original_BF16_4096_true3_v1_real_FLOCK_20261116'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=json.loads((J/'job.json').read_bytes());done=json.loads((J/'completion.json').read_bytes());cfg=json.loads((R/job['spec']['config']).read_bytes());p=Path(cfg['model']['kwargs']['protocol_file']);proto=json.loads(p.read_bytes())
a,=done['accepted'];pf=json.loads(Path(a['proof']).read_bytes());front=json.loads(Path(proto['own_frontend200']).read_bytes())
out={'job_SHA256':sha(J/'job.json'),'completion_SHA256':sha(J/'completion.json'),'proof_SHA256':sha(a['proof']),'proof':pf,'protocol':proto,'protocol_SHA256':sha(p),'original_config':cfg,'own_frontend_first':front[0],'all200_patches':[x['patches'] for x in front],'all200_tensor_metadata_empty':all(x['tensor_metadata']=={} for x in front)}
print(json.dumps(out,ensure_ascii=False))

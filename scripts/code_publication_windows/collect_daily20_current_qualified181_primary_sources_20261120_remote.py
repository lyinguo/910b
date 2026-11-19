import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
p=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';pointer=json.loads(p.read_bytes());pending=Path(pointer['new_primary_pending']);digest=hashlib.sha256(pending.read_bytes()).hexdigest();assert digest==pointer['new_primary_pending_SHA256']=='43ccaedff3fcb0e5b0134db98dc34616290b251329f2c7008553d8d34d3a12ac'
assert pointer['verified_delivered_count']==181
home=C/'daily_20261120_explicit_primary_source_review_v1'
args=['python3','-B',str(R/'scripts/collect_code_publication_primary_sources.py'),'--pending',str(pending),'--pending-sha',digest,'--review-home',str(home)]
r=subprocess.run(args,cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)

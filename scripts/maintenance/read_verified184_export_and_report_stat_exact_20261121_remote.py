import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121';out={}
for n in ['verified184_export_complete_metadata.json','active184_full_report_stat_baseline.json','full_target_dual_publication_audit.json']:
 b=(H/n).read_bytes();out[n]={'SHA256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps(out))
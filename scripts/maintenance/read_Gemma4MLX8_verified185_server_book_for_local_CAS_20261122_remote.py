import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
I=B/'Gemma4E4BMLX8_verified185_publication_review_inputs_v1_20261122';o={'UTC':datetime.now(timezone.utc).isoformat()}
for f in [I/'publication_actual_completion.json',I/'publication_stderr.log',H/'transaction.json',H/'server_phase_waiting_for_local_receipt.json']:
 if f.exists():
  raw=f.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode(errors='replace')
  o[str(f)]=v
f=H/'publication_receipt.json'
if f.exists():
 r=json.loads(f.read_bytes());assert r['completed'] and r['canonical_counts']['done']==185 and [x['model_id'] for x in r['targets']]==['unsloth/gemma-4-E4B-it-MLX-8bit'];book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert sha(book)==r['canonical_after_sha256'][str(book)];o.update(server_receipt=r,server_receipt_SHA256=sha(f),book_SHA256=sha(book),book_base64=base64.b64encode(book.read_bytes()).decode())
print(json.dumps(o,ensure_ascii=False))

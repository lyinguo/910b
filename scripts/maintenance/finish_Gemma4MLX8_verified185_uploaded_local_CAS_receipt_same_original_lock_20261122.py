import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
raw=Path(sys.argv[1]).read_bytes();q=json.loads(raw);p=json.loads((H/'publication_receipt.json').read_bytes());assert p['completed'] and p['canonical_counts']['done']==185
assert hashlib.sha256(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx').read_bytes()).hexdigest()==q['after_book_SHA256']=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
assert p['targets'][0]['model_id']=='unsloth/gemma-4-E4B-it-MLX-8bit' and p['targets'][0]['formal_proof_sha256']=='3cf18242f6f6276cfe552ccbb5d2bfd54ea56ad0863785b21097e26404bbad8d'
dest=H/'reviewed_local_receipt_ready.json';assert not dest.exists();dest.write_bytes(raw)
print(json.dumps(dict(receipt_ready=True,receipt_request_SHA256=hashlib.sha256(raw).hexdigest(),original_publisher_to_finish_under_same_EX=True)))

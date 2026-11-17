import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent;f=P/'CoderGemma4E4B_local_result_sync_verified176_20261117/local_publication_receipt.json';b=f.read_bytes();r=json.loads(b);V=dict(local_receipt_base64=base64.b64encode(b).decode(),local_receipt_SHA256=hashlib.sha256(b).hexdigest(),after_book_SHA256=r['after_sha256']);assert r['completed'] and r['changed_G_cells']==['G212']
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117';V=__V__
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');assert hashlib.sha256(book.read_bytes()).hexdigest()==V['after_book_SHA256'];raw=base64.b64decode(V['local_receipt_base64']);assert hashlib.sha256(raw).hexdigest()==V['local_receipt_SHA256'];r=json.loads(raw);assert r['source_server_receipt_SHA256']==hashlib.sha256((H/'publication_receipt.json').read_bytes()).hexdigest()
p=H/'reviewed_local_receipt_ready.json';assert not p.exists();p.write_bytes(json.dumps(V,ensure_ascii=False,indent=2).encode());print(json.dumps(dict(uploaded=True,UTC=datetime.now(timezone.utc).isoformat(),local_receipt_SHA256=V['local_receipt_SHA256'],server_and_local_book_SHA256=V['after_book_SHA256'])))
'''.replace('__V__',repr(V));ast.parse(s);(P/'upload_Coder176_exact_local_CAS_receipt_20261117_remote.py').write_bytes(s.encode());print(json.dumps(V|{'local_receipt_base64':'[not printed]'}))

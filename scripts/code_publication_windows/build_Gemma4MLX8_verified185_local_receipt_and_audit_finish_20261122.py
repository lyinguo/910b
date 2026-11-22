import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;L=V/'Gemma4E4BMLX8_local_result_sync_verified185_20261122'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(L/'local_publication_receipt.json').read_bytes();r=json.loads(raw);q=json.loads((L/'reviewed_server_publication_inputs.json').read_bytes());assert r['changed_G_cells']==['G664'] and r['after_sha256']=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
assert sha(Path(q['target']).read_bytes())==r['after_sha256'];q.update(local_receipt_base64=base64.b64encode(raw).decode(),local_receipt_SHA256=sha(raw),after_book_SHA256=r['after_sha256']);qb=json.dumps(q,ensure_ascii=False,indent=2).encode();(L/'reviewed_local_receipt_ready.json').write_bytes(qb)
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
raw=Path(sys.argv[1]).read_bytes();q=json.loads(raw);p=json.loads((H/'publication_receipt.json').read_bytes());assert p['completed'] and p['canonical_counts']['done']==185
assert hashlib.sha256(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx').read_bytes()).hexdigest()==q['after_book_SHA256']=='2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
assert p['targets'][0]['model_id']=='unsloth/gemma-4-E4B-it-MLX-8bit' and p['targets'][0]['formal_proof_sha256']=='3cf18242f6f6276cfe552ccbb5d2bfd54ea56ad0863785b21097e26404bbad8d'
dest=H/'reviewed_local_receipt_ready.json';assert not dest.exists();dest.write_bytes(raw)
print(json.dumps(dict(receipt_ready=True,receipt_request_SHA256=hashlib.sha256(raw).hexdigest(),original_publisher_to_finish_under_same_EX=True)))
''';ast.parse(s);n='finish_Gemma4MLX8_verified185_uploaded_local_CAS_receipt_same_original_lock_20261122.py';(V/n).write_bytes(s.encode());rel='scripts/maintenance/'+n
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,subprocess,os\nR=Path('/root/projects/comperror');b=base64.b64decode("+repr(base64.b64encode(s.encode()).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(s.encode()))+"\nast.parse(b);p=R/"+repr(rel)+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nI=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Gemma4E4BMLX8_verified185_publication_review_inputs_v1_20261122';q=I/'uploaded_actual_local_receipt_request.json';assert not q.exists();raw=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(raw).hexdigest()=="+repr(sha(qb))+";q.write_bytes(raw)\nr=subprocess.run(['python3','-B',str(p),str(q)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'upload_Gemma4MLX8_verified185_reviewed_local_CAS_receipt_20261122_remote.py').write_bytes(carrier.encode());print(json.dumps(dict(source=rel,source_SHA256=sha(s.encode()),local_receipt_SHA256=sha(raw),after_book_SHA256=r['after_sha256'])))

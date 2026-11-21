from pathlib import Path
import hashlib, json, base64, ast
P = Path(__file__).resolve().parent
H = P / 'GLM41Thinking_local_result_sync_verified184_20261120'
raw = (H / 'local_publication_receipt.json').read_bytes()
receipt = json.loads(raw)
assert receipt['completed'] and receipt['changed_G_cells'] == ['G696']
v = dict(local_receipt_base64=base64.b64encode(raw).decode(), local_receipt_SHA256=hashlib.sha256(raw).hexdigest(), after_book_SHA256=receipt['after_sha256'])
s = 'V=' + repr(v) + '\n' + "import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport hashlib,json,base64\nR=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'GLM41Thinking_original_VLLM_BF16_8192_LZMA_verified184_blank_G_publication_20261120';book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')\nassert hashlib.sha256(book.read_bytes()).hexdigest()==V['after_book_SHA256'];raw=base64.b64decode(V['local_receipt_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==V['local_receipt_SHA256'];r=json.loads(raw);assert r['source_server_receipt_SHA256']==hashlib.sha256((H/'publication_receipt.json').read_bytes()).hexdigest()\np=H/'reviewed_local_receipt_ready.json';data=json.dumps(V,ensure_ascii=False,indent=2).encode()\nif p.exists():assert p.read_bytes()==data\nelse:\n with p.open('xb') as f:f.write(data)\nprint(json.dumps(dict(uploaded=True,local_receipt_SHA256=V['local_receipt_SHA256'],server_and_local_book_SHA256=V['after_book_SHA256'])))\n"
ast.parse(s)
(P / 'upload_GLM41Thinking_verified184_exact_local_CAS_receipt_20261120_remote.py').write_text(s, encoding='utf8', newline='\n')
print(json.dumps(dict(receipt_SHA256=v['local_receipt_SHA256'], after_book_SHA256=v['after_book_SHA256'])))

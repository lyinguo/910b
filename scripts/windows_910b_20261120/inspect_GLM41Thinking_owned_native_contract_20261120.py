import ast,json
from pathlib import Path
V=Path(__file__).resolve().parent
d=json.loads((V/'R_GLM41Thinking_own_saved_assets_native_contract_20261120.json').read_bytes())
print('TOP',list(d))
for name,r in d['files'].items():
 print('OWN_SOURCE',name,r['SHA256'])
 if name.endswith('.yaml'):print(r['complete_text'])
 else:
  tree=ast.parse(r['complete_text'])
  for n in tree.body:
   if isinstance(n,(ast.Assign,ast.AnnAssign)) and '_IMAGE_PLACEHOLDER_RE' in ast.unparse(n):print(ast.unparse(n))
   if isinstance(n,ast.ClassDef) and n.name=='VLLMModel':
    for f in n.body:
     if isinstance(f,ast.FunctionDef) and f.name in ['_build_message','_extra_chat_kwargs','_generate']:
      print(ast.unparse(f))
print('ROW',d['actual_workbook_row'])
print('ASSETS', {k:r['SHA256'] for k,r in d['own_assets'].items()})

from pathlib import Path
import ast
import base64
import hashlib
import json

P = Path('D:/code-lv/.tmp/batch_910b_20261030')
old = P / 'prepare_phi35_own_original_chat_native_tokens200_existing_pixels_meta_map_v5_20261118.py'
name = 'prepare_phi35_own_original_chat_native_tokens200_existing_pixels_meta_map_v6_exact_ChatParams_20261118.py'
raw = old.read_text(encoding='utf-8').replace(
    'Phi35_own_original_chat_generation_prefix_native_tokens200_existing_pixel_proof_meta_map_v5_no_NN_20261118',
    'Phi35_own_original_chat_generation_prefix_native_tokens200_existing_pixel_proof_meta_map_v6_exact_ChatParams_no_NN_20261118',
).replace(
    'ChatParams(add_generation_prompt=True)',
    "ChatParams(chat_template_kwargs={'add_generation_prompt': True, 'continue_final_message': False})",
)
assert raw != old.read_text(encoding='utf-8')
ast.parse(raw)
target = P / name
assert not target.exists()
target.write_bytes(raw.encode('utf-8'))
rows = []
for relative, local in [
    ('universal_eval/model/reference_sources/phi35_owned_checkpoint_layout_20261118.py', P / 'phi35_owned_checkpoint_layout_20261118.py'),
    ('scripts/' + name, target),
]:
    data = local.read_bytes()
    rows.append(dict(relative=relative, base64=base64.b64encode(data).decode(), SHA256=hashlib.sha256(data).hexdigest()))
deploy = 'ROWS=' + repr(rows) + '''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,runpy
R=Path('/root/projects/comperror')
for row in ROWS:
 p=R/row['relative'];raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
runpy.run_path(str(R/'scripts/NAME'),run_name='__main__')
'''.replace('NAME', name)
(P / 'deploy_run_Phi35_original_chat_prefix_native200_existing_pixels_meta_map_v6_20261118_remote.py').write_bytes(deploy.encode('utf-8'))
print(json.dumps(dict(primary=name, SHA256=hashlib.sha256(target.read_bytes()).hexdigest(), changes='ChatParams field binding only; original add_generation_prompt True and continue_final_message False; no NN or original model mutation')))

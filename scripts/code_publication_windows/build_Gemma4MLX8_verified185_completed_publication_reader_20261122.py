import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
BASE=Path('D:/code-lv/.tmp/cd22s1/scripts/code_publication_windows')
s=(BASE/'read_Prithiv_verified180_dual_publication_exact_inputs_1916_20261117_remote.py').read_text(encoding='utf-8')
s=s.replace('Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117','Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122').replace('Prithiv_completed_card2_true3_user_verified179_retention_capacity_recovered_continuation_20261117','Gemma4E4BMLX8_completed_genuine3_formal_NF2_actual_profiler_schema_then_capacity_serial_v2_20261122').replace('MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117','OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121').replace('prior179','prior184').replace('verified179','verified184').replace('active179','active184')
ast.parse(s)
name='read_Gemma4MLX8_verified185_dual_publication_exact_inputs_20261122_remote.py'
(V/name).write_bytes(s.encode())
rel='scripts/maintenance/'+name
raw=s.encode();h=hashlib.sha256(raw).hexdigest()
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,os,subprocess\nR=Path('/root/projects/comperror');p=R/"+repr(rel)+"\nb=base64.b64decode("+repr(base64.b64encode(raw).decode())+")\nassert hashlib.sha256(b).hexdigest()=="+repr(h)+"\nast.parse(b)\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(p)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))\nraise SystemExit(r.returncode)\n"
ast.parse(carrier)
(V/'stage_read_Gemma4MLX8_verified185_publication_exact_inputs_20261122_remote.py').write_text(carrier,encoding='utf-8')
print(json.dumps(dict(source=rel,SHA256=h,no_NN_or_canonical_write=True)))

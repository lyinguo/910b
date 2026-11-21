"""Private standard-NPY storage layout; old raw bytes and decoder unchanged."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
old='complete_native_raw_bytes_Fortran_LZMA_20261120'
new='complete_native_raw_bytes_natural_order_byteplanes_LZMA_v2_20261120'
s=(V/(old+'.py')).read_text(encoding='utf-8')
needle='reshape([*t.shape,t.element_size()])'
assert s.count(needle)==1
s=s.replace(needle,'reshape(t.numel(),t.element_size())').replace('standard shaped uint8','standard natural-C-order byteplane uint8')
(V/(new+'.py')).write_text(s,encoding='utf-8',newline='\n')
dn='diagnose_MetaMllama_completed_true3_complete_natural_order_byteplanes_LZMA_v2_20261120'
d=(V/'diagnose_MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_capacity_20261120.py').read_text(encoding='utf-8')
d=d.replace(old,new).replace('MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_storage_capacity_diagnostic_only_20261120','MetaMllama_completed_true3_complete_natural_order_byteplanes_LZMA_storage_v2_diagnostic_only_20261120')
d=d.replace("phase='full_values_decode_write_fortran_LZMA'","phase='complete_natural_order_byteplanes_LZMA'")
(V/(dn+'.py')).write_text(d,encoding='utf-8',newline='\n')
names={new+'.py':'universal_eval/utils/'+new+'.py',dn+'.py':'scripts/'+dn+'.py'}
items={}
for n,rel in names.items():
 raw=(V/n).read_bytes();ast.parse(raw);items[rel]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
template=(V/'build_stage_MetaMllama_complete_shaped_raw_Fortran_LZMA_diagnostic_20261120.py').read_text(encoding='utf-8')
tree=ast.parse(template);assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
remote=ast.literal_eval(assignment.value.func.value)
remote=remote.replace('PAYLOAD',base64.b64encode(json.dumps(items).encode()).decode())
remote=remote.replace('MetaMllama_completed3_full_shaped_raw_Fortran_LZMA_diagnostic_source_launch_20261120','MetaMllama_completed3_natural_order_byteplanes_LZMA_v2_diagnostic_source_launch_20261120').replace('diagnose_MetaMllama_completed_true3_complete_shaped_raw_Fortran_LZMA_capacity_20261120',dn)
ast.parse(remote)
(V/'stage_MetaMllama_natural_order_byteplanes_storage_v2_20261120_remote.py').write_text(remote,encoding='utf-8',newline='\n')
(V/'MetaMllama_natural_order_byteplanes_storage_v2_source_SHA256_20261120.json').write_text(json.dumps({n:v['SHA256'] for n,v in items.items()},indent=2)+'\n',encoding='utf-8')
print(json.dumps({n:v['SHA256'] for n,v in items.items()},indent=2))

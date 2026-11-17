import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,copy,msgspec
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.model.adapters.pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117 import EXTENSION
old=ast.parse((R/'universal_eval/model/adapters/paligemma2_10b_pt448_A100_native_vllm_20261110_adapter.py').read_text());new=ast.parse((R/'universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117.py').read_text())
a=next(x for c in old.body if isinstance(c,ast.ClassDef) for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='load');b=next(x for c in new.body if isinstance(c,ast.ClassDef) and c.name=='Pali10Pt448NativeNamedWorkerEngine' for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='load')
b=copy.deepcopy(b);call,=[x for x in ast.walk(b) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='LLM'];delta=[x for x in call.keywords if x.arg=='worker_extension_cls'];assert len(delta)==1 and ast.literal_eval(delta[0].value)==EXTENSION;call.keywords=[x for x in call.keywords if x.arg!='worker_extension_cls'];assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
for method in ['own_pali_install_full_value_capture','own_pali_begin_full_value_capture','own_pali_finish_full_value_capture']:
 value=('collective_rpc',method,('/own/path',{'own_tensor':{'dtype':'BF16','shape':[1,2],'shard':'own.safetensors'}},'/own/evidence'),{});assert msgspec.msgpack.decode(msgspec.msgpack.encode(value))[1]==method
for call in ast.walk(new):
 if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='collective_rpc':assert isinstance(call.args[0],ast.Constant) and isinstance(call.args[0].value,str)
assert 'VLLM_ALLOW_INSECURE_SERIALIZATION=1' not in (R/'universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117.py').read_text()
print('Original native loader AST identical except observation extension; all RPC methods are plain named strings; strict message encoding passed; no model forward')

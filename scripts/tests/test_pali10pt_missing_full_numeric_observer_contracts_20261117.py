"""CPU-only validation of rounding/corruption guards and observer structure."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,importlib.util,json,hashlib,tempfile
import numpy as np
import torch
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
nf=R/'scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_20261117.py';spec=importlib.util.spec_from_file_location('own_pali_value_contract',nf);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[]
torch.manual_seed(1701);reference=torch.randn(128,256)*torch.pow(2.,torch.arange(256).float()%20-10)
module.numeric(reference.bfloat16(),reference,'independent declared BF16 rounding range');checks.append('BF16 quantization across exponents accepted')
def rejected(a,b,label):
 try:module.numeric(a,b,label)
 except AssertionError:checks.append(label+' rejected');return
 raise AssertionError(label+' was silently accepted')
rejected(reference[:2],reference,'shape corruption')
bad=reference.clone();bad[0,0]=float('nan');rejected(bad,reference,'NaN corruption')
bad=reference.clone();bad[0,0]=float('inf');rejected(bad,reference,'infinite corruption')
rejected(-reference,reference,'weight/output sign corruption')
module.numeric(torch.zeros(4),torch.zeros(4),'all zero source');checks.append('zero-source denominator defined')
# Independent raw bit-pattern oracle checks signed zero, subnormal endpoints,
# infinities and NaN payload preservation. They are serialization tests only;
# nonfinite neural values remain rejected by the numerical guard above.
bits=np.array([0,32768,1,127,128,16256,32640,65408,32705],dtype=np.uint16);expected=hashlib.sha256(bits.tobytes()).hexdigest();tensor=torch.from_numpy(bits.copy()).view(torch.bfloat16)
with tempfile.TemporaryDirectory(prefix='pali_numeric_binary_contract_') as td:
 f=Path(td)/'roundtrip.npz';np.savez_compressed(f,data=tensor.view(torch.uint16).numpy())
 with np.load(f,allow_pickle=False) as z:assert hashlib.sha256(z['data'].tobytes()).hexdigest()==expected
checks.append('lossless BF16 binary oracle including signed zero/subnormals/payloads')
adapter=R/'universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_observer_v2_ABI_20261117.py';tree=ast.parse(adapter.read_bytes());cls,=[n for n in tree.body if isinstance(n,ast.ClassDef)]
for name in ['load','batch_predict']:
 fn,=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name];super_calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==name and isinstance(n.func.value,ast.Call) and isinstance(n.func.value.func,ast.Name) and n.func.value.func.id=='super'];assert len(super_calls)==1
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['LLM','SamplingParams'] for n in ast.walk(tree));checks.append('single original load/chat invocation, no new engine/sampler arguments')
assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('torch' in ast.unparse(n) or '.worker' in ast.unparse(n)) for n in tree.body);checks.append('observer import delayed until original physical-device isolation')
worker=R/'universal_eval/model/pali10pt448_full_native_value_20261117/worker_v2_ABI.py';s=worker.read_text();ast.parse(s);assert "self.wrap(attention.rotary_emb,'forward'" not in s and "self.wrap(attention,'forward'" in s;checks.append('shared RoPE module not wrapped repeatedly per layer')
print(json.dumps(dict(passed=True,checks=checks,CPU_only=True,no_NPU_allocation_or_model_forward=True,no_model_or_metric_result_claimed=True),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,ast,copy,hashlib,base64,textwrap
P=Path(__file__).parent
packet=json.loads((P/'R_Pali10pt_existing_own_adapter_protocol_stages_20261117.json').read_text(encoding='utf8'))
original=packet['source_files']['universal_eval/model/adapters/paligemma2_10b_pt448_A100_native_vllm_20261110_adapter.py']['text']
cls=next(x for x in ast.parse(original).body if isinstance(x,ast.ClassDef));load=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='load');newload=copy.deepcopy(load)
extension='universal_eval.model.pali10pt448_full_native_value_20261117.worker_named_RPC_v3.OwnPaliFullValueWorkerExtension'
llm,=[n for n in ast.walk(newload) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='LLM'];assert not any(k.arg=='worker_extension_cls' for k in llm.keywords);llm.keywords.append(ast.keyword(arg='worker_extension_cls',value=ast.Constant(extension)))
probe=copy.deepcopy(newload);call,=[n for n in ast.walk(probe) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='LLM'];call.keywords=[k for k in call.keywords if k.arg!='worker_extension_cls'];assert ast.dump(probe,include_attributes=False)==ast.dump(load,include_attributes=False)
observer=(P/'pali10pt448_own_preserved_vllm_full_value_observer_20261117.py').read_text(encoding='utf8');oc=next(x for x in ast.parse(observer).body if isinstance(x,ast.ClassDef));oc.name='Pali10Pt448OwnFullValuesNamedRPCV4';oc.bases=[ast.Name(id='Pali10Pt448NativeNamedWorkerEngine',ctx=ast.Load())]
class RPC(ast.NodeTransformer):
 def visit_ImportFrom(self,node):
  if node.module and 'pali10pt448_full_native_value_20261117.worker' in node.module:return None
  return node
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and node.func.attr=='collective_rpc' and node.args and isinstance(node.args[0],ast.Name):
   names={'install_capture':'own_pali_install_full_value_capture','begin_capture':'own_pali_begin_full_value_capture','finish_capture':'own_pali_finish_full_value_capture'};assert node.args[0].id in names;node.args[0]=ast.Constant(names[node.args[0].id])
  return node
 def visit_keyword(self,node):
  self.generic_visit(node)
  if node.arg=='all_original_engine_arguments_dtype_sampling_prompt_and_batching_inherited':node.arg='all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only'
  return node
oc=ast.fix_missing_locations(RPC().visit(oc));ownload=next(x for x in oc.body if isinstance(x,ast.FunctionDef) and x.name=='load')
extra=ast.parse("assert self._engine.llm_engine.vllm_config.parallel_config.worker_extension_cls == EXTENSION\nassert os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION') in [None, '0']\nself._lineage.update(observation_only_worker_extension_cls=EXTENSION, named_RPC_only_no_arbitrary_callable_deserialization=True, actual_engine_kwargs_delta={'worker_extension_cls': EXTENSION})").body
ownload.body.extend(extra)
native='''"""Own original native vLLM math, plus standard named worker RPC observers.

The single additional engine keyword selects an observation-only worker mixin.
No arbitrary-function serialization flag, alternate model, sampler or backend.
"""
import sys,os,tempfile,json,hashlib
from pathlib import Path
from universal_eval.model.adapters.paligemma2_10b_pt448_A100_native_vllm_20261110_adapter import PaliGemma10BPt448Native910B,MODEL_ID,MAX_TOKENS,EVIDENCE_ROOT,LOG,digest
EXTENSION=__EXTENSION__
class Pali10Pt448NativeNamedWorkerEngine(PaliGemma10BPt448Native910B):
__LOAD__
__OBSERVER__
'''.replace('__EXTENSION__',repr(extension)).replace('__LOAD__',textwrap.indent(ast.unparse(newload),'    ')).replace('__OBSERVER__',ast.unparse(oc))
worker='''"""Only named observation methods are added to the existing native worker."""
from pathlib import Path
import shutil
class OwnPaliFullValueWorkerExtension:
    def own_pali_install_full_value_capture(self,root,headers,home):
        from universal_eval.model.pali10pt448_full_native_value_20261117.worker_v2_ABI import install_capture
        assert shutil.disk_usage(Path(home)).free > 3*1024**3+1024**3+512*1024**2
        return install_capture(self,root,headers,home)
    def own_pali_begin_full_value_capture(self,group):
        from universal_eval.model.pali10pt448_full_native_value_20261117.worker_v2_ABI import begin_capture
        observer=self._own_pali10pt_full_capture
        assert shutil.disk_usage(observer.home).free > 3*1024**3+1024**3+512*1024**2+3*2*1024**3
        return begin_capture(self,group)
    def own_pali_finish_full_value_capture(self):
        from universal_eval.model.pali10pt448_full_native_value_20261117.worker_v2_ABI import finish_capture
        observer=self._own_pali10pt_full_capture
        exact_uncompressed_bytes=sum(a.nbytes for a in observer.arrays.values())
        assert shutil.disk_usage(observer.home).free > 3*1024**3+1024**3+512*1024**2+exact_uncompressed_bytes+1024*1024
        return finish_capture(self)
'''
nf=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_20261117.py').read_text(encoding='utf8');oldassert="assert lin['all_original_engine_arguments_dtype_sampling_prompt_and_batching_inherited']"
assert oldassert in nf
nf=nf.replace(oldassert,"assert lin['all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only'] and lin['named_RPC_only_no_arbitrary_callable_deserialization'];assert lin['observation_only_worker_extension_cls']==p['observation_only_worker_extension_cls'] and lin['actual_engine_kwargs_delta']=={'worker_extension_cls':p['observation_only_worker_extension_cls']}")
adapter_rel='universal_eval/model/adapters/pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117.py';worker_rel='universal_eval/model/pali10pt448_full_native_value_20261117/worker_named_RPC_v3.py';nf_rel='scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v4_named_worker_RPC_20261117.py'
prep=(P/'prepare_Pali10pt_missing_full_native_value_true3_v3_source_bound_20261117.py').read_text(encoding='utf8')
prep=prep.replace('Pali10Pt448_missing_full_native_value_true3_v3_source_bound_readiness_20261117','Pali10Pt448_missing_full_native_value_true3_v4_named_worker_RPC_readiness_20261117').replace('Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v3_real_FLOCK_20261117','Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v4_named_worker_RPC_real_FLOCK_20261117')
prep=prep.replace('pali10pt448_own_preserved_vllm_full_value_observer_v2_ABI_20261117','pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117').replace('Pali10Pt448OwnPreservedVllmFullValues','Pali10Pt448OwnFullValuesNamedRPCV4').replace('scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_20261117.py',nf_rel)
prep=prep.replace("[nfs,adapter,worker,me,'universal_eval/model/pali10pt448_full_native_value_20261117/__init__.py']","[nfs,adapter,worker,"+repr(worker_rel)+",me,'universal_eval/model/pali10pt448_full_native_value_20261117/__init__.py']")
prep=prep.replace('p.update(full_numeric_observer_not_model_math_change=True,','p.update(observation_only_worker_extension_cls='+repr(extension)+',full_numeric_observer_not_model_math_change=True,')
prep=prep.replace('missing_full_native_true3_v3_20261117.json','missing_full_native_true3_v4_named_worker_RPC_20261117.json').replace('missing-full-native-value-true3-v3-20261117.yaml','missing-full-native-value-true3-v4-named-worker-RPC-20261117.yaml')
prep=prep.replace('Own original native vLLM BF16/1024 and batch10 unchanged. New missing full-value numerical3 phase;','Own original native vLLM BF16/1024, sampling and batch10 unchanged; only the standard observation worker_extension_cls is added and named RPC is used. New missing full-value numerical3 phase;').replace('original_native_vllm_dtype1024_batch10_prompt_extraction_scoring_identical=True','original_native_model_math_dtype1024_batch10_prompt_extraction_scoring_identical=True,only_observation_worker_extension_keyword_added=True,no_insecure_callable_serialization_enabled=True')
prep_rel='scripts/prepare_Pali10pt_missing_full_native_value_true3_v4_named_worker_RPC_source_bound_20261117.py'
test='''import sys
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
'''
test_rel='scripts/tests/test_pali10pt_named_worker_RPC_original_loader_and_strict_encoding_20261117.py'
files={adapter_rel:native,worker_rel:worker,nf_rel:nf,prep_rel:prep,test_rel:test}
for n,s in files.items():ast.parse(s);(P/Path(n).name).write_bytes(s.encode())
payload={n:dict(SHA256=hashlib.sha256(s.encode()).hexdigest(),data=base64.b64encode(s.encode()).decode()) for n,s in files.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);f=R/n
 if f.exists():assert f.read_bytes()==raw
 else:f.write_bytes(raw)
I=B/'Pali10Pt448_named_worker_RPC_v4_source_unit_and_readiness_no_NN_actual_launch_20261117';I.mkdir(exist_ok=False);results=[]
for name,source in [('unit','scripts/tests/test_pali10pt_named_worker_RPC_original_loader_and_strict_encoding_20261117.py'),('readiness','scripts/prepare_Pali10pt_missing_full_native_value_true3_v4_named_worker_RPC_source_bound_20261117.py')]:
 with (I/(name+'_stdout.json')).open('xb') as out,(I/(name+'_stderr.log')).open('xb') as err:p=subprocess.run(['python3','-B',source],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
 results.append(dict(stage=name,actual_returncode=p.returncode,stdout=(I/(name+'_stdout.json')).read_text(),stderr=(I/(name+'_stderr.log')).read_text()))
 if p.returncode:break
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=results[-1]['actual_returncode'],sources={n:x['SHA256'] for n,x in FILES.items()},results=results,no_NPU_dispatch_or_model_forward=True,old_completed_compatibility3_and_B10_and_old_failed_v3_preserved=True);(I/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False));raise SystemExit(v['actual_returncode'])
'''.replace('__FILES__',repr(payload));ast.parse(remote);(P/'deploy_Pali10pt_named_worker_RPC_v4_unit_readiness_no_NN_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))

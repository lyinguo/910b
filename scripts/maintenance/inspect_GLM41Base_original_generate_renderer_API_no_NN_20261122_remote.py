"""Inspect installed original completion renderer API; no engine or learned model."""
import sys
sys.dont_write_bytecode=True
import inspect,json,hashlib
from pathlib import Path
from vllm.entrypoints.llm import LLM
from vllm.renderers import base
items={}
for cls in [LLM,base.BaseRenderer] if hasattr(base,'BaseRenderer') else [LLM]:
 for name in dir(cls):
  if ('render' in name or 'preprocess' in name or name=='generate') and callable(getattr(cls,name)):
   try:s=inspect.getsource(getattr(cls,name));items[cls.__name__+'.'+name]=dict(signature=str(inspect.signature(getattr(cls,name))),complete_source=s,source_file=inspect.getfile(getattr(cls,name)))
   except (TypeError,OSError,ValueError):pass
print(json.dumps(dict(installed_original_source_interfaces=items,renderer_module_file=base.__file__,renderer_module_complete_source=Path(base.__file__).read_text(),no_learned_CPU_or_NPU_execution=True),ensure_ascii=False))

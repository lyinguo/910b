"""Isolated compatibility for the checkpoint's own MIT/Apache HoVLE code.

No checkpoint/shared package is edited. Inactive LoRA and nonzero drop_path_rate are rejected.
Inactive imports are omitted without replacement symbols or fake packages.
"""
import pathlib,sys,types,hashlib,importlib,ast,copy,logging,json
import torch


def classes(directory):
 directory=pathlib.Path(directory).resolve();raw=json.loads((directory/'config.json').read_text());assert raw.get('use_backbone_lora',0)==raw.get('use_llm_lora',0)==0 and raw['embedding_config'].get('drop_path_rate',0)==0.;name='hovle_own_original_v2_'+hashlib.sha256(str(directory).encode()).hexdigest()[:16]
 if name+'.modeling_internvl_chat' in sys.modules:return sys.modules[name+'.modeling_internvl_chat']
 pkg=types.ModuleType(name);pkg.__path__=[str(directory)];sys.modules[name]=pkg
 old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
 try:
  module=types.ModuleType(name+'.modeling_holistic_embedding');module.__package__=name;module.__file__=str(directory/'modeling_holistic_embedding.py')
  source=pathlib.Path(module.__file__).read_text()
  for a,b in [('from timm.models.layers import DropPath','# Original drop_path_rate is zero: inactive import omitted, no replacement symbol or stand-in'),('torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)','torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device="cpu")')]:
   assert source.count(a)==1;source=source.replace(a,b)
  sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
  chat=types.ModuleType(name+'.modeling_internvl_chat');chat.__package__=name;chat.__file__=str(directory/'modeling_internvl_chat.py');source=pathlib.Path(chat.__file__).read_text();a='from peft import LoraConfig, get_peft_model';assert source.count(a)==1;source=source.replace(a,'# Inactive LoRA imports omitted; configuration must have both LoRA ranks zero')
  sys.modules[chat.__name__]=chat;exec(compile(source,chat.__file__,'exec'),chat.__dict__)
  return chat
 finally:sys.dont_write_bytecode=old

def tokenizer(directory,package):
 old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
 try:return importlib.import_module(package+'.tokenization_internlm2').InternLM2Tokenizer.from_pretrained(directory,local_files_only=True)
 finally:sys.dont_write_bytecode=old

def compatible_model_class(module):
 from transformers.generation import GenerationMixin
 class HoVLEOriginalGenerationCompatibility(module.InternVLChatModel,GenerationMixin):
  @classmethod
  def _supports_default_dynamic_cache(cls):
   # Own forward returns 8 holistic +24 LM native tuple slots. Original empty
   # DynamicCache patch normalized the unused prefill cache to None. Request
   # that same None directly; retain the actual native32 cache after prefill.
   return False
 return HoVLEOriginalGenerationCompatibility

def apply_original_runtime_patches(model,source,source_sha256,device):
 """Replay exact original frozen causal-mask/cache/chat compatibility locally."""
 source=pathlib.Path(source);assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha256
 text=source.read_text();tree=ast.parse(text);names=['_patch_hovle_prepare_inputs','_patch_hovle_causal_mask','_patch_hovle'];nodes=[]
 for name in names:
  node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name));node.decorator_list=[];nodes.append(node)
 env={'torch':torch,'logger':logging.getLogger('HoVLE_original_exact_runtime')}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(source),'exec'),env)
 owner=types.SimpleNamespace(_model=model,_device=str(device),is_hovle=lambda:True)
 for name in names:env[name](owner)
 assert model.config.num_hidden_layers==24
 return {name:hashlib.sha256(ast.get_source_segment(text,next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)).encode()).hexdigest() for name in names}

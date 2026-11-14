"""Own Ovis2-2B classes and checkpoint, without the historical mock initializer."""
import ast,hashlib,importlib,json,logging,sys
from pathlib import Path
MODEL='AIDC-AI/Ovis2-2B'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def own_classes(root,source_home,expected):
 root=Path(root);source_home=Path(source_home);package=source_home/'own_ovis22b_1512'
 package.mkdir(parents=True,exist_ok=False);(package/'__init__.py').write_text('',encoding='utf-8')
 changes=[]
 for name in ('configuration_aimv2.py','modeling_aimv2.py','configuration_ovis.py','modeling_ovis.py'):
  original=(root/name).read_bytes();assert hashlib.sha256(original).hexdigest()==expected[name]
  tree=ast.parse(original)
  class Compatibility(ast.NodeTransformer):
   def visit_Call(self,node):
    self.generic_visit(node)
    if isinstance(node.func,ast.Attribute) and node.func.attr=='register' and isinstance(node.func.value,ast.Name) and node.func.value.id in ('AutoConfig','AutoModel'):
     if not any(k.arg=='exist_ok' for k in node.keywords):node.keywords.append(ast.keyword(arg='exist_ok',value=ast.Constant(True)));changes.append(name+': own configuration registration accepts existing mapping')
    return node
   def visit_Attribute(self,node):
    self.generic_visit(node)
    if name=='modeling_ovis.py' and isinstance(node.ctx,ast.Load) and node.attr in ('_no_split_modules','_skip_keys_device_placement','_keep_in_fp32_modules','is_parallelizable') and isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id=='self' and node.value.attr in ('llm','visual_tokenizer'):
     changes.append(name+': optional transformer metadata '+node.attr)
     return ast.copy_location(ast.Call(func=ast.Name(id='getattr',ctx=ast.Load()),args=[node.value,ast.Constant(node.attr),ast.Constant(False if node.attr=='is_parallelizable' else None)],keywords=[]),node)
    return node
  tree=Compatibility().visit(tree);ast.fix_missing_locations(tree)
  text=ast.unparse(tree)+'\n';compile(text,name,'exec');(package/name).write_text(text,encoding='utf-8',newline='\n')
 sys.path.insert(0,str(source_home));importlib.invalidate_caches()
 C=importlib.import_module(package.name+'.configuration_ovis');M=importlib.import_module(package.name+'.modeling_ovis')
 CA=importlib.import_module(package.name+'.configuration_aimv2');MA=importlib.import_module(package.name+'.modeling_aimv2')
 from transformers import AutoModel,CLIPImageProcessor
 image_settings=json.loads((root/'preprocessor_config.json').read_bytes())
 assert image_settings['image_processor_type']=='CLIPImageProcessor'
 class OwnDeclaredImageProcessor:
  @staticmethod
  def from_pretrained(path,*args,**kwargs):
   assert Path(path).resolve()==root.resolve() and not args and not kwargs
   return CLIPImageProcessor.from_dict(image_settings)
 M.AutoImageProcessor=OwnDeclaredImageProcessor
 changes.append('own declared native CLIPImageProcessor.from_dict avoids unrelated remote model-config prompt; complete200 values must match previous native-auto CPU proof')
 AutoModel.register(CA.AIMv2Config,MA.AIMv2Model,exist_ok=True)
 AutoModel.register(C.Aimv2VisualTokenizerConfig,M.Aimv2VisualTokenizer,exist_ok=True)
 return C,M,changes,package
def create(root,source_home,expected,device=None,load_weights=True):
 import torch
 from transformers import Qwen2Tokenizer as AutoTokenizer,GenerationConfig
 from safetensors import safe_open
 root=Path(root);C,M,changes,package=own_classes(root,source_home,expected)
 M.AutoTokenizer=AutoTokenizer
 changes.append('explicit same original Qwen2Tokenizer class bypasses AutoTokenizer custom-model-config prompt; full200 token equality required')
 tok=AutoTokenizer.from_pretrained(str(root),local_files_only=True)
 # Only the own formatter's accepted runtime tokenizer type changes; tokenization is untouched.
 if type(tok).__name__ not in C.QwenConversationFormatter.support_tokenizer_types:
  C.QwenConversationFormatter.support_tokenizer_types=list(C.QwenConversationFormatter.support_tokenizer_types)+[type(tok).__name__]
  changes.append('own Qwen formatter accepts actual tokenizer runtime class '+type(tok).__name__)
 config=C.OvisConfig.from_dict(json.loads((root/'config.json').read_bytes()))
 config.name_or_path=str(root);config.multimodal_max_length=8192
 assert config.llm_attn_implementation=='flash_attention_2'
 # Selected original run.log explicitly resolves this exact own field to eager.
 config.llm_attn_implementation='eager'
 before=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
 try:
  if load_weights:model=M.Ovis(config)
  else:
   with torch.device('meta'):model=M.Ovis(config)
 finally:torch.set_default_dtype(before)
 model.generation_config=GenerationConfig.from_pretrained(str(root),local_files_only=True)
 manifest=None
 if load_weights:
  state={}
  for file in sorted(root.glob('*.safetensors')):
   with safe_open(file,framework='pt',device='cpu') as f:
    for key in f.keys():assert key not in state;state[key]=f.get_tensor(key)
  model.tie_weights()
  loaded=model.load_state_dict(state,strict=False)
  assert not loaded.unexpected_keys,loaded.unexpected_keys
  target=model.state_dict()
  for key in loaded.missing_keys:
   t=target[key]
   assert any(t.untyped_storage().data_ptr()==target[k].untyped_storage().data_ptr() and t.storage_offset()==target[k].storage_offset() and t.shape==target[k].shape for k in state),('missing non-alias',key)
  assert all(p.dtype==torch.bfloat16 for p in model.parameters())
  if device is not None:model=model.to(device)
  model.eval();target=model.state_dict();parameters={}
  def digest(t):
   t=t.detach().cpu().contiguous();return {'shape':list(t.shape),'dtype':str(t.dtype),'SHA256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}
  for key,source in state.items():
   actual=target[key];cast=source.to(actual.dtype);assert digest(cast)==digest(actual),key
   parameters[key]={'source':digest(source),'cast':digest(cast),'actual_device_type':actual.device.type}
  alias={}
  for n,p in model.named_parameters(remove_duplicate=False):alias.setdefault((p.untyped_storage().data_ptr(),p.storage_offset()),[]).append(n)
  manifest={'source_tensor_count':len(state),'source_to_actual':parameters,'missing_only_physical_aliases':loaded.missing_keys,
   'physical_aliases':[v for v in alias.values() if len(v)>1],'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()})}
  del state
 return model,{'model_id':MODEL,'independent_own_source_package':str(package),'compatibility_changes':changes,
  'original_log_resolved_llm_attention':'eager','vision_attention':'own AIMv2 SDPA',
  'original_dtype':'torch.bfloat16','original_multimodal_max_length':8192,
  'generation_config':model.generation_config.to_dict(),'source_manifest':manifest,
  'original_source_SHA256':expected,'standalone_factory_source_SHA256':sha(__file__),
  'no_flash_PEFT_or_CPU_model_stub':True,'no_library_or_original_checkpoint_file_modified':True}
def original_predict_replay(source,kwargs,model):
 import os,logging
 from typing import Any,Dict,List,Optional,Tuple,Union
 tree=ast.parse(source);old=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OvisAdapter')
 methods=[n for n in old.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','predict','batch_predict')]
 cls=ast.ClassDef(name='OwnOriginalOvisPrediction',bases=[],keywords=[],body=methods,decorator_list=[])
 module=ast.Module(body=[cls],type_ignores=[]);ast.fix_missing_locations(module)
 namespace=locals().copy();namespace['logger']=logging.getLogger('own.Ovis.original.predict')
 exec(compile(module,'own_original_predict_AST_no_mock_loader','exec'),namespace)
 obj=namespace[cls.name](**kwargs);obj._model=model;obj._text_tokenizer=model.get_text_tokenizer();obj._visual_tokenizer=model.get_visual_tokenizer()
 return obj

"""Isolated loading of the checkpoint's own InternVL/InternLM2 implementation.

The original OpenGVLab MIT and InternLM Apache notices remain in the read-only
checkpoint sources. Only the nonlearned evaluation DropPath and meta schedule
construction are supplied here. No shared package or checkpoint is modified.
"""
import pathlib,sys,types,hashlib,importlib
import torch

class EvaluationDropPath(torch.nn.Identity):
 def __init__(self,drop_prob=0.):super().__init__();self.drop_prob=drop_prob
 def forward(self,x):
  if self.training and self.drop_prob:raise RuntimeError('Evaluation-only DropPath')
  return x

def classes(directory):
 directory=pathlib.Path(directory).resolve();name='figcodifier_original_'+hashlib.sha256(str(directory).encode()).hexdigest()[:16]
 if name+'.modeling_internvl_chat' in sys.modules:return sys.modules[name+'.modeling_internvl_chat']
 pkg=types.ModuleType(name);pkg.__path__=[str(directory)];sys.modules[name]=pkg
 vision=types.ModuleType(name+'.modeling_intern_vit');vision.__package__=name;vision.__file__=str(directory/'modeling_intern_vit.py');vision.DropPath=EvaluationDropPath
 source=pathlib.Path(vision.__file__).read_text()
 for old,new in [('from timm.models.layers import DropPath','# Original evaluation DropPath is exact identity; supplied by this isolated adapter'),('torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)','torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device="cpu")')]:
  assert source.count(old)==1;source=source.replace(old,new)
 sys.modules[vision.__name__]=vision
 old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
 try:
  exec(compile(source,vision.__file__,'exec'),vision.__dict__)
  return importlib.import_module(name+'.modeling_internvl_chat')
 finally:sys.dont_write_bytecode=old

def tokenizer(directory,package):
 old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
 try:return importlib.import_module(package+'.tokenization_internlm2').InternLM2Tokenizer.from_pretrained(directory,local_files_only=True)
 finally:sys.dont_write_bytecode=old

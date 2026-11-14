"""Independent CPU reference to the checkpoint-owned legacy Florence vision equations; never official answers."""
import pathlib,ast,types,math,collections,torch,numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from transformers.models.florence2.modeling_florence2 import Florence2DropPath
def independent_pixels(image,config):
 image=image.convert('RGB').resize((768,768),Image.Resampling.LANCZOS);a=np.asarray(image).astype(np.float64);a=(a*config['rescale_factor']).astype(np.float32);a=(a-np.array(config['image_mean'],dtype=np.float32))/np.array(config['image_std'],dtype=np.float32);return torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0)
def semantic_target(k):
 if k=='image_projection':return 'model.multi_modal_projector.image_projection.weight','transpose_2d'
 if k=='language_model.final_logits_bias':return 'original_final_logits_bias','identity'
 for old,new in [('vision_tower.','model.vision_tower.'),('image_proj_norm.','model.multi_modal_projector.image_proj_norm.'),('image_pos_embed.','model.multi_modal_projector.image_position_embed.'),('visual_temporal_embed.','model.multi_modal_projector.visual_temporal_embed.'),('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.')]:
  if k.startswith(old):
   name=new+k[len(old):]
   if k.startswith('vision_tower.'):
    if '.convs.' in name:name=name.replace('.proj.','.conv.')
    for a,b in [('.conv1.fn.dw.','.conv1.'),('.conv2.fn.dw.','.conv2.'),('.window_attn.norm.','.norm1.'),('.channel_attn.norm.','.norm1.'),('.window_attn.fn.','.window_attn.'),('.channel_attn.fn.','.channel_attn.'),('.ffn.norm.','.norm2.'),('.ffn.fn.net.','.ffn.')]:name=name.replace(a,b)
   return name,'identity'
 raise AssertionError(k)
def build_legacy_vision(weight_path,raw_vision_config,dtype=torch.float32,original_FP16_rounded=True):
 source=(pathlib.Path(weight_path)/'modeling_florence2.py').read_text();tree=ast.parse(source);nodes=[]
 for node in tree.body:
  if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.lineno<695:nodes.append(node)
 namespace={'torch':torch,'nn':nn,'F':F,'math':math,'OrderedDict':collections.OrderedDict,'rearrange':rearrange,'DropPath':Florence2DropPath,'trunc_normal_':nn.init.trunc_normal_,'checkpoint':torch.utils.checkpoint}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<own original Florence vision CPU>','exec'),namespace)
 vision=namespace['DaViT'].from_config(types.SimpleNamespace(**raw_vision_config));del vision.head;del vision.norms;vision=vision.to(dtype=dtype)
 from safetensors import safe_open
 tensors={}
 with safe_open(str(pathlib.Path(weight_path)/'model.safetensors'),framework='pt',device='cpu') as f:
  for key in f.keys():
   if key.startswith('vision_tower.'):
    v=f.get_tensor(key);v=v.half() if original_FP16_rounded else v;tensors[key[len('vision_tower.'):]]=v.to(dtype=dtype)
 result=vision.load_state_dict(tensors,strict=True);assert not result.missing_keys and not result.unexpected_keys;vision.eval();return vision

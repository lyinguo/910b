"""Own original Oct4 adapter projector and prescribed token registration, instance-only."""
import ast,pathlib,types,torch
from universal_eval.model.adapters.florence_original_vision_ops_v2_910b import install_original_vision_equations

def original_Oct4_projector(module,image_features):
 b,d,h,w=image_features.shape;n=h*w;x=image_features.permute(0,2,3,1).reshape(b,n,d);side=int(n**.5)
 if side*side==n:
  position_input=x.view(b,side,side,d).permute(0,3,1,2);position=module.image_position_embed(position_input).permute(0,2,3,1).reshape(b,n,d);x=x+position
 x=x.view(b,1,n,d);spatial=x.mean(dim=2);time=x.mean(dim=1);x=torch.cat([spatial,time],dim=1);x=x@module.image_projection.weight.T;return module.image_proj_norm(x)

def install_original_Oct4_equations(model):
 counts=install_original_vision_equations(model);model.model.multi_modal_projector.forward=types.MethodType(original_Oct4_projector,model.model.multi_modal_projector);return dict(counts,original_adapter_temporal_addition=False)

def register_own_original_special_tokens(weight_path,tokenizer):
 assert len(tokenizer)==50265
 tree=ast.parse((pathlib.Path(weight_path)/'processing_florence2.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Florence2Processor');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');node=next(n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tokens_to_add' for t in n.targets));view=types.SimpleNamespace(additional_special_tokens=list(tokenizer.extra_special_tokens));tokens=eval(compile(ast.Expression(node.value),'<own original Florence processor token declaration>','eval'),{'tokenizer':view})['additional_special_tokens'];before=tokenizer.get_vocab().copy();added=tokenizer.add_special_tokens({'additional_special_tokens':tokens});assert added==1024 and len(tokenizer)==51289 and all(tokenizer.convert_tokens_to_ids(k)==v for k,v in before.items());out={t:tokenizer.convert_tokens_to_ids(t) for t in tokens};assert sorted(out.values())==list(range(50265,51289)) and '<image>' not in out and all(tokenizer.convert_ids_to_tokens(i) is not None for i in range(51289));return out

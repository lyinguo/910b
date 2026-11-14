"""Instance-only restoration of the checkpoint's actual DaViT attention and projector equations."""
import types,torch
from torch.nn import functional as F
def original_channel(module,x):
 b,n,c=x.shape;qkv=module.qkv(x).reshape(b,n,3,module.groups,c//module.groups).permute(2,0,3,1,4);q,k,v=qkv.unbind(0);q=q*(float(n)**-0.5);attention=q.transpose(-1,-2)@k;attention=attention.softmax(dim=-1);x=(attention@v.transpose(-1,-2)).transpose(-1,-2);x=x.transpose(1,2).reshape(b,n,c);return module.proj(x)
def original_window(module,x):
 b,h,w,c=x.shape;s=module.window_size;right=(s-w%s)%s;bottom=(s-h%s)%s;x=F.pad(x,(0,0,0,right,0,bottom));hp,wp=x.shape[1:3];x=x.view(b,hp//s,s,wp//s,s,c).permute(0,1,3,2,4,5).contiguous().view(-1,s*s,c);bw,n,c=x.shape;qkv=module.qkv(x).reshape(bw,n,3,module.num_heads,c//module.num_heads).permute(2,0,3,1,4);q,k,v=qkv.unbind(0);q=q*module.scale;attention=(q@k.transpose(-2,-1)).softmax(dim=-1);x=(attention@v).transpose(1,2).reshape(bw,n,c);x=module.proj(x);x=x.view(b,hp//s,wp//s,s,s,c).permute(0,1,3,2,4,5).contiguous().view(b,hp,wp,c)
 if right or bottom:x=x[:,:h,:w,:].contiguous()
 return x.view(b,h*w,c)
def original_projector(module,image_features):
 b,d,h,w=image_features.shape;x=image_features.permute(0,2,3,1).reshape(b,h*w,d);position=module.image_position_embed(image_features).permute(0,2,3,1).reshape(b,h*w,d);x=x+position;temporal=module.visual_temporal_embed(x.view(b,1,-1,d)[:,:,0]);x=x.view(b,1,-1,d)+temporal.view(1,1,1,d);spatial=x.mean(dim=2);time=x.mean(dim=1);x=torch.cat([spatial,time],dim=1);x=x@module.image_projection.weight.T;return module.image_proj_norm(x)
def install_original_vision_equations(model):
 counts={'channel_modules':0,'window_modules':0}
 for name,module in model.model.vision_tower.named_modules():
  if type(module).__name__=='Florence2VisionChannelAttention':module.forward=types.MethodType(original_channel,module);counts['channel_modules']+=1
  if type(module).__name__=='Florence2VisionWindowAttention':module.forward=types.MethodType(original_window,module);counts['window_modules']+=1
 total=sum(model.config.vision_config.depths);assert total==12 and counts=={'channel_modules':total,'window_modules':total},counts
 model.model.multi_modal_projector.forward=types.MethodType(original_projector,model.model.multi_modal_projector)
 return counts

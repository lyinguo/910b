"""Independent, own Donut uint8 resize/thumbnail/pad and fused normalization."""
import numpy as np
import torch
import torch.nn.functional as F

def pixels(image,p):
 assert p['size']=={'height':1280,'width':960} and p['resample']==2 and not p['do_align_long_axis']
 assert all(p[k] for k in ['do_resize','do_thumbnail','do_pad','do_rescale','do_normalize']) and not p.get('do_center_crop',False)
 x=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).contiguous()
 def resize(x,h,w):
  if tuple(x.shape[-2:])==(h,w):return x
  return F.interpolate(x[None],size=(h,w),mode='bilinear',align_corners=False,antialias=True)[0]
 h,w=x.shape[-2:];edge=960
 nh,nw=(edge,int(edge*w/h)) if h<=w else (int(edge*h/w),edge)
 x=resize(x,nh,nw);h,w=x.shape[-2:];nh,nw=min(h,1280),min(w,960)
 if nh!=h or nw!=w:
  if h>w:nw=int(w*nh/h)
  elif w>h:nh=int(h*nw/w)
  x=resize(x,nh,nw)
 h,w=x.shape[-2:];top=(1280-h)//2;left=(960-w)//2;canvas=torch.zeros((3,1280,960),dtype=torch.uint8);canvas[:,top:top+h,left:left+w]=x
 mean=torch.tensor(p['image_mean'],dtype=torch.float32)*(1./p['rescale_factor']);std=torch.tensor(p['image_std'],dtype=torch.float32)*(1./p['rescale_factor'])
 return ((canvas.float()-mean[:,None,None])/std[:,None,None])[None]

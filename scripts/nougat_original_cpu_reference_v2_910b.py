"""Independent explicit CPU replay of the model's original TorchVision crop/resize/pad/fused normalization."""
import numpy as np
import torch
import torch.nn.functional as F
def _resize_uint8(x,height,width,mode):
 if tuple(x.shape[-2:])==(height,width):return x
 return F.interpolate(x[None],size=(height,width),mode=mode,align_corners=False,antialias=True)[0]
def independent_nougat_pixels(image,p):
 assert p['size']=={'height':896,'width':672} and p['resample']==2 and not p['do_align_long_axis']
 assert all(p[k] for k in ['do_crop_margin','do_resize','do_thumbnail','do_pad','do_rescale','do_normalize'])
 x=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).contiguous();red,green,blue=x.unbind(0);gray=(0.2989*red+0.587*green+0.114*blue).to(torch.uint8)
 low=gray.min();high=gray.max()
 if high!=low:
  normalized=(gray-low)/(high-low)*255;ys,xs=torch.where(normalized<200);x=x[:,int(ys.min()):int(ys.max())+1,int(xs.min()):int(xs.max())+1]
 height,width=x.shape[-2:];short=672
 if height<=width:new_height,new_width=short,int(short*width/height)
 else:new_height,new_width=int(short*height/width),short
 x=_resize_uint8(x,new_height,new_width,'bilinear');height,width=x.shape[-2:];new_height=min(height,896);new_width=min(width,672)
 if new_height!=height or new_width!=width:
  if height>width:new_width=int(width*new_height/height)
  elif width>height:new_height=int(height*new_width/width)
  x=_resize_uint8(x,new_height,new_width,'bicubic')
 height,width=x.shape[-2:];top=(896-height)//2;left=(672-width)//2;padded=torch.zeros((3,896,672),dtype=torch.uint8);padded[:,top:top+height,left:left+width]=x
 mean=torch.tensor(p['image_mean'],dtype=torch.float32)*(1./p['rescale_factor']);std=torch.tensor(p['image_std'],dtype=torch.float32)*(1./p['rescale_factor'])
 return ((padded.float()-mean[:,None,None])/std[:,None,None])[None]

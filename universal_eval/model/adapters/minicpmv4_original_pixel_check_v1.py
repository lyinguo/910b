"""Independent uint8 normalization and patch layout for the own original slicer."""
import sys
sys.dont_write_bytecode=True
from PIL import Image
import numpy as np
import torch
def check_pixels(image_processor,media_path,actual):
 with Image.open(media_path) as im:images=image_processor.get_sliced_images(im.convert('RGB'))
 assert len(images)==len(actual)
 for im,t in zip(images,actual):
  a=np.array(im,dtype=np.uint8).astype(np.float32)/np.float32(255)
  a=((a-np.float32(.5))/np.float32(.5)).transpose(2,0,1)
  h,w=a.shape[1:];assert h%14==w%14==0
  want=a.reshape(3,h//14,14,w//14,14).transpose(0,2,1,3,4).reshape(3,14,-1)
  assert t.dtype==torch.float32 and np.array_equal(want,t.cpu().numpy())
 return len(images)

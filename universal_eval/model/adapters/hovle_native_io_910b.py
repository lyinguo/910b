"""Own HoVLE native byte hashing and original PIL/ToTensor pixels; no model imports."""
import pathlib,hashlib
import torch
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':fp(x)}
def pixels(im):
 from torchvision import transforms
 ratios=sorted({(i,j) for n in range(1,13) for i in range(1,n+1) for j in range(1,n+1) if 1<=i*j<=12},key=lambda x:x[0]*x[1]);best=(1,1);dist=float('inf');w,h=im.size
 for r in ratios:
  d=abs(w/h-r[0]/r[1])
  if d<dist:best,dist=r,d
  elif d==dist and w*h>.5*448*448*r[0]*r[1]:best=r
 full=im.resize((best[0]*448,best[1]*448));tiles=[full.crop((x*448,y*448,(x+1)*448,(y+1)*448)) for y in range(best[1]) for x in range(best[0])]
 if len(tiles)>1:tiles.append(im.resize((448,448)))
 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);return torch.stack([transform(t) for t in tiles]),list(best)

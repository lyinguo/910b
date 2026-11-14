"""Independent PIL/NumPy, own SentencePiece input/output and tensor evidence replay."""
import pathlib,json,hashlib,re
import torch,numpy as np,sentencepiece as spm
from PIL import Image
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':fp(x)}
def arrays(path,records):
 result={}
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(records)
  for k,r in records.items():
   x=torch.from_numpy(z[k].copy()).view(getattr(torch,r['dtype'].split('.')[-1])).reshape(r['shape']);assert tr(x)==r;result[k]=x
 return result
def independent_pixels(im):
 ratios=sorted(set((i,j) for n in range(1,13) for i in range(1,n+1) for j in range(1,n+1) if 1<=i*j<=12),key=lambda r:r[0]*r[1]);w,h=im.size;ds=[abs(w/h-r[0]/r[1]) for r in ratios];best=min(ds);tied=[i for i,d in enumerate(ds) if d==best];large=[i for i in tied if w*h>.5*448*448*ratios[i][0]*ratios[i][1]];grid=ratios[large[-1] if large else tied[0]]
 full=im.resize((grid[0]*448,grid[1]*448),resample=Image.Resampling.BICUBIC);tiles=[full.crop((x*448,y*448,(x+1)*448,(y+1)*448)) for y in range(grid[1]) for x in range(grid[0])]
 if len(tiles)>1:tiles.append(im.resize((448,448),resample=Image.Resampling.BICUBIC))
 x=torch.from_numpy(np.stack([np.asarray(t).copy().transpose(2,0,1) for t in tiles])).float()/255;return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None],list(grid)
class OwnSentencePieceReference:
 def __init__(self,wp):
  self.sp=spm.SentencePieceProcessor(model_file=str(pathlib.Path(wp)/'tokenizer.model'));self.cfg=json.loads((pathlib.Path(wp)/'tokenizer_config.json').read_text());self.special={v['content']:int(k) for k,v in self.cfg['added_tokens_decoder'].items()};self.special_ids={int(k) for k,v in self.cfg['added_tokens_decoder'].items() if v['special']};assert len(self.special_ids)==len(self.special)
  self.splitter=re.compile('('+'|'.join(re.escape(x) for x in sorted(self.special,key=len,reverse=True))+')');assert self.sp.get_piece_size()==92544
 def encode(self,text):
  ids=[1]
  for part in self.splitter.split(text):
   if part:ids.extend([self.special[part]] if part in self.special else self.sp.encode(part,out_type=int))
  return ids
 def decode(self,ids):
  assert all(0<=i<92557 for i in ids);pieces=[self.sp.id_to_piece(i) for i in ids if i not in self.special_ids];s=self.sp.decode(pieces)
  # Own convert_tokens_to_string unconditionally performs this cleanup.
  for old,new in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(" ' ","'"),(" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")]:s=s.replace(old,new)
  # Its _maybe_add_prefix_space compares token strings to ID set; for nonempty
  # normal tokens it adds one space then slices [1:], leaving this exact string.
  return s

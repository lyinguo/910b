"""Independent own92553 SentencePiece decoder and PIL/tensor evidence helpers."""
import pathlib,json,re
import sentencepiece as spm
from figcodifier_independent_cpu_910b import sha,fp,tr,arrays,independent_pixels

class HoVLEOwnSentencePieceReference:
 def __init__(self,wp):
  self.sp=spm.SentencePieceProcessor(model_file=str(pathlib.Path(wp)/'tokenizer.model'));cfg=json.loads((pathlib.Path(wp)/'tokenizer_config.json').read_text());self.special={v['content']:int(k) for k,v in cfg['added_tokens_decoder'].items()};self.special_ids={int(k) for k,v in cfg['added_tokens_decoder'].items() if v['special']};assert self.sp.get_piece_size()==92544 and max(self.special.values())==92552
  self.splitter=re.compile('('+'|'.join(re.escape(x) for x in sorted(self.special,key=len,reverse=True))+')')
 def encode(self,text):
  ids=[1]
  for part in self.splitter.split(text):
   if part:ids.extend([self.special[part]] if part in self.special else self.sp.encode(part,out_type=int))
  return ids
 def decode(self,ids):
  assert all(0<=v<92553 for v in ids);parts=[]
  for v in ids:
   if v in self.special_ids:continue
   assert v<92544;parts.append(self.sp.id_to_piece(v))
  out=self.sp.decode(parts)
  for old,new in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(" ' ","'"),(" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")]:out=out.replace(old,new)
  return out

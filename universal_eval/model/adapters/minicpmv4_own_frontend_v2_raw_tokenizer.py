"""Restore the own raw tokenizer backend after HF constructor loses image specials."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
from tokenizers import Tokenizer
from universal_eval.model.adapters.minicpmv4_own_frontend_v1 import OwnMiniCPMV4Inputs,tensor_sha,flatten_tensors

class OwnMiniCPMV4InputsV2(OwnMiniCPMV4Inputs):
 def __init__(self,weight_path):
  super().__init__(weight_path);W=Path(weight_path)
  original_vocab=self.tokenizer.get_vocab()
  raw=Tokenizer.from_file(str(W/'tokenizer.json'))
  assert raw.get_vocab()==original_vocab and raw.get_vocab_size()==73448
  # Preserve all original token ids, normalizer, pretokenizer, decoder and special
  # metadata. No add_tokens, vocabulary substitution, model resizing or new ids.
  self.tokenizer._tokenizer=raw
  own=json.loads((W/'tokenizer.json').read_bytes())
  markers={x['content']:x['id'] for x in own['added_tokens']}
  for text in ['<image>','</image>','<slice>','</slice>','<image_id>','</image_id>']:
   assert self.tokenizer.encode(text,add_special_tokens=False)==[markers[text]]
  assert len(self.tokenizer)==73448 and self.tokenizer.convert_tokens_to_ids('<|im_end|>')==73440
 def prepare(self,prompt,media_path):
  text,v=super().prepare(prompt,media_path)
  bounds=v['image_bound'][0];assert len(bounds)==len(v['pixel_values'][0])>0
  for start,end in bounds:assert int(end)-int(start)==64
  return text,v

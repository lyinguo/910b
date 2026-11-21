"""Apply only this checkpoint's own saved added_tokens_decoder to Rust tokens."""
from pathlib import Path
import hashlib,json
from tokenizers import Tokenizer,AddedToken
def load_owned_rust(W):
 W=Path(W);p=W/'tokenizer_config.json';raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()=='468e400c12d54be1b1b0e008fa5ee9382c8c9ef29a5dd0b5093421ef261935ed'
 raw_vocab=(W/'tokenizer.json').read_bytes();assert hashlib.sha256(raw_vocab).hexdigest()=='76ebeac0d8bd7879ead7b43c16b44981f277e47225de2bd7de9ae1a6cc664a8c'
 rust=Tokenizer.from_str(raw_vocab.decode());before=rust.get_vocab();cfg=json.loads(raw)
 for token_id,record in sorted(cfg['added_tokens_decoder'].items(),key=lambda p:int(p[0])):
  token=AddedToken(**record)
  if record['special']:rust.add_special_tokens([token])
  else:rust.add_tokens([token])
  assert rust.token_to_id(record['content'])==int(token_id)
 assert len(before)==151343 and rust.get_vocab_size()==151352 and all(rust.token_to_id(k)==v for k,v in before.items())
 return rust

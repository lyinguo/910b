import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,inspect,json,os
R=Path('/root/projects/comperror');W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration/zai-org/GLM-4.1V-9B-Thinking')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from tokenizers import Tokenizer
from transformers import AutoProcessor
p=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);rust=Tokenizer.from_file(str(W/'tokenizer.json'));a=rust.get_vocab();b=p.tokenizer.get_vocab();c=json.loads((W/'tokenizer_config.json').read_bytes())
added={k:v for k,v in b.items() if k not in a};changed={k:[v,b.get(k)] for k,v in a.items() if b.get(k)!=v}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),own_processor_class=type(p).__name__,tokenizer_class=type(p.tokenizer).__name__,own_raw_vocab_size=len(a),actual_processor_vocab_size=len(b),new_processor_tokens=added,raw_tokens_not_same_in_processor=changed,own_tokenizer_config=c,processor_init_kwargs=p.to_dict(),actual_added_tokens={str(k):dict(content=str(v),special=v.special,normalized=v.normalized,lstrip=v.lstrip,rstrip=v.rstrip,single_word=v.single_word) for k,v in p.tokenizer.added_tokens_decoder.items()},own_assets_SHA256={n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in ['tokenizer.json','tokenizer_config.json','config.json','preprocessor_config.json','chat_template.jinja']},actual_tokenizer_source=inspect.getfile(type(p.tokenizer)),actual_NEW_model_forwards=0),ensure_ascii=False))

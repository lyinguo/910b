import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,inspect
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from transformers import AutoProcessor
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration/unsloth/gemma-4-E4B-it-MLX-8bit')
p=AutoProcessor.from_pretrained(str(W),local_files_only=True)
cls=type(p)
for name,value in cls.__dict__.items():
 if callable(value) and any(x in name for x in ['image','token','call']):
  try: print('METHOD',name,inspect.getsource(value))
  except (OSError,TypeError):pass
print('ATTR',json.dumps({k:v for k,v in p.__dict__.items() if isinstance(v,(str,int,float,bool,type(None)))},ensure_ascii=False))
print('GEN', (W/'generation_config.json').read_text())

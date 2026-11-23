"""Read the installed original vLLM Gemma3n architecture and own config headers."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
base=Path('/mnt/afs/miniconda3/envs/lvzhiqiang/lib/python3.12/site-packages')
models=base/'vllm/model_executor/models'
files=sorted(models.glob('*gemma3n*.py'))
files += [base/'vllm/model_executor/layers/layernorm.py']
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma3nForConditionalGeneration/nicoboss/MedraN-E4B-Uncensored-EP7')
files += [W/'config.json',W/'model.safetensors.index.json']
records={}
for p in files:
    if not p.is_file():records[str(p)]=dict(exists=False);continue
    b=p.read_bytes();st=p.stat()
    records[str(p)]=dict(exists=True,SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),mtime_ns=st.st_mtime_ns,complete_source_text=b.decode('utf8'))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),complete_exact_sources=records,
    original_MedraN_weight_directory_available=W.is_dir(),no_weight_array_load=True,
    no_model_forward_or_runtime_change=True,readonly_no_install_download_or_write=True),ensure_ascii=False))

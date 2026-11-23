"""Read exact original unweighted architecture sources, without model execution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
base=Path('/mnt/afs/miniconda3/envs/lvzhiqiang/lib/python3.12/site-packages')
files=['timm/layers/fast_norm.py','timm/layers/norm.py','timm/layers/norm_act.py',
       'timm/models/mobilenetv5.py','timm/models/_efficientnet_blocks.py',
       'transformers/models/gemma3n/modeling_gemma3n.py','transformers/models/timm_wrapper/modeling_timm_wrapper.py']
records={}
for n in files:
    p=base/n
    if not p.is_file():records[str(p)]=dict(exists=False);continue
    st=p.stat();b=p.read_bytes()
    records[str(p)]=dict(exists=True,SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),
        mtime_ns=st.st_mtime_ns,complete_source_text=b.decode('utf8'))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exact_original_source_records=records,
    read_only=True,no_model_or_checkpoint_or_tensor_load=True,no_learned_CPU_or_GPU_or_NPU_forward=True,
    no_package_install_or_download=True,no_A100_current_job_state_claim=True),ensure_ascii=False))

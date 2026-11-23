"""Exact original dependency source and header metadata only, without weight data."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,struct
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma3nForConditionalGeneration/nicoboss/MedraN-E4B-Uncensored-EP7')
ix=json.loads((W/'model.safetensors.index.json').read_bytes());headers={}
for name in sorted(set(ix['weight_map'].values())):
    p=W/name;st=p.stat()
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];assert 0<n<64*1024*1024
        b=f.read(n);assert len(b)==n
    h=json.loads(b)
    headers[name]=dict(file_bytes=st.st_size,mtime_ns=st.st_mtime_ns,header_bytes=n,header_SHA256=hashlib.sha256(b).hexdigest(),
                      tensor_metadata={k:v for k,v in h.items() if k!='__metadata__'})
p=Path('/mnt/afs/miniconda3/envs/lvzhiqiang/lib/python3.12/site-packages/timm/models/regnet.py')
b=p.read_bytes()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),MedraN_exact_current_shard_headers=headers,
    RegStage_original_dependency_source=dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_source_text=b.decode('utf8')),
    no_checkpoint_tensor_data_read=True,no_model_or_CPU_GPU_NPU_forward=True,
    no_package_install_download_or_write=True,current_metadata_does_not_prove_historical_A100_execution=True),ensure_ascii=False))

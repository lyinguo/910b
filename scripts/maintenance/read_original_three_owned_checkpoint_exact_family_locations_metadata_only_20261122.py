"""Check only the same three model IDs in their two existing registered class namespaces; no substitution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels')
rows=[]
for mid in ['nvidia/Nemotron-3-Content-Safety','AiAsistent/gemma-3-4b-it-Cognitive-Liberty','CalmState/gemma-3-4b-polyglot-r1']:
    for cls in ['Gemma3ForConditionalGeneration','Unknown']:
        p=root/cls/mid
        try:
            exists=p.is_dir();assets={};weights=[]
            if exists:
                for name in ['config.json','model.safetensors.index.json','tokenizer_config.json']:
                    q=p/name
                    if q.is_file():
                        b=q.read_bytes();assets[name]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b))
                for q in p.iterdir():
                    if q.name.endswith('.safetensors') and q.is_file():
                        st=q.stat();weights.append(dict(name=q.name,bytes=st.st_size))
            rows.append(dict(model_id=mid,path=str(p),directory_available=exists,own_small_asset_metadata=assets,own_weight_file_stat_only=weights))
        except OSError as e:rows.append(dict(model_id=mid,path=str(p),directory_available=False,error=repr(e)))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exact_same_owned_model_ID_metadata_only=rows,no_weight_tensor_read_or_A100_claim_or_dispatch=True,no_download_substitution_or_source_change=True,any_recovered_path_requires_original_frozen_full_file_and_tensor_SHA_before_private_retry=True),ensure_ascii=False))

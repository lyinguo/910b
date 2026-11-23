"""Read this model's own saved deployment and source/header metadata only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,struct
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='kakaocorp/kanana-1.5-v-3b-instruct'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/KananaVForConditionalGeneration')/MID
def rec(p):
    b=p.read_bytes()
    try:v=json.loads(b)
    except ValueError:v=b.decode('utf8')
    return dict(path=str(p),bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),complete_value=v)
paths=[R/'configs/KananaVForConditionalGeneration/kakaocorp--kanana-1.5-v-3b-instruct.yaml']
paths += sorted(W.glob('*.py'))
paths += [W/n for n in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json'] if (W/n).is_file()]
records={str(p):rec(p) for p in paths if p.is_file()}
ix=json.loads((W/'model.safetensors.index.json').read_bytes());headers={}
for name in sorted(set(ix['weight_map'].values())):
    p=W/name;st=p.stat()
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];assert 0<n<64*1024*1024;b=f.read(n);assert len(b)==n
    headers[name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,header_SHA256=hashlib.sha256(b).hexdigest(),complete_header=json.loads(b))
previous={}
for h in sorted(B.glob('*[Kk]anana*')):
    if not h.is_dir():continue
    previous[str(h)]=dict(files=[p.name for p in h.iterdir() if p.is_file()])
    for n in ['actual_completion.json','review.json','readiness.json','state.json','completion.json']:
        p=h/n
        if p.is_file() and p.stat().st_size<=1024*1024:previous[str(h)][n]=rec(p)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_source_records=records,
    own_shard_header_records=headers,previous_exact_model_stages=previous,
    no_weight_array_or_frontend_or_model_replay=True,no_mutation=True),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
def read(p,expected=None):
 b=Path(p).read_bytes();h=hashlib.sha256(b).hexdigest();assert expected is None or h==expected
 t=b.decode('utf8');out=dict(SHA256=h,bytes=len(b),complete_text=t)
 try:out['complete_JSON']=json.loads(t)
 except ValueError:pass
 return out
files={}
p=H/'handoff.txt';files[str(p)]=read(p,'19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1');assert '\r' not in files[str(p)]['complete_text']
p=H/'facts.json';files[str(p)]=read(p,'1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7')
for p in [H/'closing_addendum.json',D/'A100_ALIGNMENT_HANDOFF_CURRENT.json']:files[str(p)]=read(p)
p=H/'late_actual_notes/verified184_BeeRL_actual_fullNF_recovered_official_readback_20261121/full_actual_verified184_BeeRL_formal_fullNF_official_readback_20261121.txt';files[str(p)]=read(p,'7ab917a8b39cbafdd7ca0ee514692e372345a0e0b73995f339fff105f83be213')
ptr=files[str(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')]['complete_JSON']
for k in ['latest_full_actual_summary','latest_actual_snapshot']:
 if ptr.get(k):p=Path(ptr[k]);files[str(p)]=read(p,ptr.get(k+'_SHA256'))
actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ptr['canonical_SHA256']}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),mandatory_read_order=list(files),files=files,all_canonical_unchanged=actual==ptr['canonical_SHA256'],canonical_SHA256=actual,canonical_differences={p:dict(before=h,actual=actual[p]) for p,h in ptr['canonical_SHA256'].items() if actual[p]!=h},late_actual_notes_inventory=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted((H/'late_actual_notes').iterdir())]),ensure_ascii=False))

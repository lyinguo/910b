import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
def whole(p,expected=None):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert expected is None or h==expected
 t=b.decode('utf8');v=dict(SHA256=h,bytes=len(b),complete_text=t)
 if p.suffix=='.json':v['complete_JSON']=json.loads(t)
 return v
files={}
for p,h in [(H/'handoff.txt','19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'),(H/'facts.json','1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7'),(H/'late_actual_notes/verified185_Xtuner_same_v2_More_original_probability_ready_serial_complete_v79_20261122/full_actual_verified185_Xtuner_same_v2_More_original_probability_ready_serial_v79_20261122.txt','5e1dacec7b6921278fcefb761e38df7f15416d54bdcf18bb65dc2aaa00890e01')]:files[str(p)]=whole(p,h)
p=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';pointer=whole(p);files[str(p)]=pointer;ptr=pointer['complete_JSON']
files[str(H/'closing_addendum.json')]=whole(H/'closing_addendum.json',ptr['closing_addendum_SHA256'])
for key in ['latest_full_actual_summary','latest_late_actual_note']:
 p=Path(ptr[key]);files[str(p)]=whole(p,ptr[key+'_SHA256'])
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),mandatory_full_read_order=list(files),files=files,all_current_canonical_fresh_SHA256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ptr['canonical_SHA256']}),ensure_ascii=False))

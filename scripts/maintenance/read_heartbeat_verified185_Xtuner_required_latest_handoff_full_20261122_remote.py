import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
def whole(p,expected=None):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert expected is None or h==expected
 text=b.decode('utf8');value=dict(SHA256=h,bytes=len(b),complete_text=text)
 if p.suffix=='.json':value['complete_JSON']=json.loads(text)
 return value
files={};files[str(H/'handoff.txt')]=whole(H/'handoff.txt','19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1');files[str(H/'facts.json')]=whole(H/'facts.json','1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7')
p=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';ptr=whole(p);files[str(p)]=ptr;files[str(H/'closing_addendum.json')]=whole(H/'closing_addendum.json',ptr['complete_JSON']['closing_addendum_SHA256'])
f=Path(ptr['complete_JSON']['latest_full_actual_summary']);files[str(f)]=whole(f,ptr['complete_JSON']['latest_full_actual_summary_SHA256']);assert files[str(f)]['SHA256']=='2ff99c0093965c86e582f23e1ca1afdbb7873ea21edb9843bdb4ac465da51f81'
late=Path(ptr['complete_JSON']['latest_late_actual_note']);files[str(late)]=whole(late,ptr['complete_JSON']['latest_late_actual_note_SHA256']);print(json.dumps(dict(mandatory_full_read_order=list(files),files=files,all_current_canonical_fresh_SHA256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ptr['complete_JSON']['canonical_SHA256']}),ensure_ascii=False))

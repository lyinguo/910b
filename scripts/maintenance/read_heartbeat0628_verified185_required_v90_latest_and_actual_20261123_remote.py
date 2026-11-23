"""Read complete specified handoffs, then actual serial state, without model replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
def rec(p,expected=None):
    p=Path(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
    if expected:assert h==expected,(str(p),h)
    try:v=json.loads(b)
    except ValueError:v=b.decode('utf8')
    return dict(path=str(p),SHA256=h,bytes=len(b),complete_value=v)
out={}
for name,h in [('handoff.txt','19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'),
               ('facts.json','1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7'),
               ('late_actual_notes/verified185_Day23_actual_commit_resource_blocked_full_actual_v90_20261123/full_actual_verified185_Day23_commit_resource_blocked_v90_20261123.txt','62226a97a8e3a86e9e9b17c5f0916c5aa23d14ef3c3919d9727a92488ca25e21'),
               ('late_actual_notes/verified185_Day23_actual_commit_resource_blocked_full_actual_v90_20261123/facts.json','b2b5a43e08e59a855f0ad23a05ee4c56e616becac1ff61c2c372099d055d6990')]:
    out[str(H/name)]=rec(H/name,h)
out[str(H/'closing_addendum.json')]=rec(H/'closing_addendum.json')
pr=rec(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');out[pr['path']]=pr;ptr=pr['complete_value']
for key,hkey in [('current_full_actual_note','current_full_actual_note_SHA256'),('current_full_facts','current_full_facts_SHA256'),('new_primary_pending','new_primary_pending_SHA256')]:
    p=Path(ptr[key]);out[str(p)]=rec(p,ptr[hkey])
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
assert r.returncode==0,r.stderr
state=json.loads((D/'state.json').read_bytes())
pending={k:v for k,v in state['models'].items() if v['status']=='needs_adaptation'}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),complete_required_and_later_handoff_records=out,
                     actual=json.loads(r.stdout),remaining_complete_weights_pending=pending,
                     no_NN_or_NF_replay_no_mutation=True),ensure_ascii=False))

"""Read the required complete handoffs first, then inspect the actual existing serial queue."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
N=H/'late_actual_notes/verified185_Base_failed_Turkish_same_serial_full_actual_v87_20261122'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p,expected=None):
    p=Path(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
    assert not expected or h==expected,(str(p),h,expected)
    s=b.decode('utf8')
    try:value=json.loads(s)
    except ValueError:value=s
    return dict(path=str(p),SHA256=h,bytes=len(b),mtime_ns=p.stat().st_mtime_ns,complete_value=value)

required={}
for p,h in [(H/'handoff.txt','19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'),
            (H/'facts.json','1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7'),
            (N/'full_actual_verified185_Base_failed_Turkish_same_serial_v87_20261122.txt','327a420602dc4a91090489b44827dbf4323e707dacc4923e6a16242233dc9455'),
            (N/'facts.json','473cc4719104e22179085557bd6d696b0f3889e0a27cc8f7d8dffda4a5ad23e6')]:
    required[str(p)]=rec(p,h)
closing=rec(H/'closing_addendum.json')
pointer=rec(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');ptr=pointer['complete_value']
assert closing['SHA256']==ptr['closing_addendum_SHA256']
later={}
for key in ['latest_late_actual_note','current_completed_stage_facts','current_full_actual_note','current_full_facts']:
    p=ptr.get(key)
    if p and Path(p).is_file() and p not in required:
        later[p]=rec(p,ptr.get(key+'_SHA256'))
pending=Path(ptr['new_primary_pending']);assert sha(pending)==ptr['new_primary_pending_SHA256']
rows={x['path']:x for x in json.loads(pending.read_bytes())['all_current_primary_source_and_true_consumer_configs']}
reader=R/'scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py'
assert sha(reader)==rows[str(reader)]['SHA256']
monitor=R/'scripts/A100_alignment_incremental_monitor_20261103.py'
assert sha(monitor)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
run=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
assert run.returncode==0,run.stderr
actual=json.loads(run.stdout)
waiter_source=R/'scripts/continue_BeeSFT_completed_storage_NF_wait_actual_capacity_v5_verified185_serial_20261122.py'
assert sha(waiter_source)==rows[str(waiter_source)]['SHA256']
waiter_text=waiter_source.read_text(encoding='utf8')
out=dict(UTC=datetime.now(timezone.utc).isoformat(),required_complete_handoffs=required,complete_current_closing=closing,
         complete_current_pointer=pointer,complete_later_actual_notes_and_facts=later,
         current_primary_pending=dict(path=str(pending),SHA256=sha(pending),source_count=len(rows)),
         actual_same_serial_and_completed_stages=actual,
         existing_BeeSFT_v5_source=dict(path=str(waiter_source),SHA256=sha(waiter_source),complete_source=waiter_text),
         all_required_bytes_read_before_actual_process_inspection=True,no_NN_replay_no_job_or_canonical_write=True)
print(json.dumps(out,ensure_ascii=False))

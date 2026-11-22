"""Read all required handoff bytes before the frozen monitor/current real process check."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677';N=H/'late_actual_notes/verified185_More_completed_Hancom_private_v3_original_registration_actual_complete_v83_20261122'
def record(p,expected=None):
 p=Path(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if expected:assert h==expected,(str(p),h,expected)
 try:v=json.loads(b)
 except ValueError:v=b.decode('utf8')
 return dict(path=str(p),SHA256=h,bytes=len(b),mtime_ns=p.stat().st_mtime_ns,complete_value=v)
required={}
for p,h in [(H/'handoff.txt','19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'),(H/'facts.json','1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7'),(N/'full_actual_verified185_More_fullNF0_capacity_Hancom_private_registration_v3_serial_v83_20261122.txt','50090ff47bb2ca61052a91b4003fd6f5138e0eb3d800977513e60740069f1104'),(N/'facts.json','467638c9cbbc825b53c395d9d4cb48aefd8312d9c37720ad66dc50dc35315bc0')]:required[str(p)]=record(p,h)
pointer=record(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');ptr=pointer['complete_value'];later={str(H/'closing_addendum.json'):record(H/'closing_addendum.json')}
for key in ['latest_full_actual_summary','latest_actual_note','latest_late_actual_note','latest_actual_snapshot','latest_round_facts']:
 p=ptr.get(key)
 if p and Path(p).is_file():later[p]=record(p,ptr.get(key+'_SHA256'))
reader=R/'scripts/maintenance/read_Hancom_same_genuine3_fullNF_and_GLM41Base_own_frontend_no_NN_v5_20261122_remote.py'
pending=record(ptr['new_primary_pending'],ptr['new_primary_pending_SHA256']);rows={r['path']:r for r in pending['complete_value']['all_current_primary_source_and_true_consumer_configs']};assert str(reader) in rows and hashlib.sha256(reader.read_bytes()).hexdigest()==rows[str(reader)]['SHA256']
run=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr;actual=json.loads(run.stdout);caps={}
for p,r in actual['Hancom_same_serial_full_actual']['Hancom_private_v3_current_complete_records'].items():
 if p.endswith('/actual_completion.json') and isinstance(r['complete_value'],dict):
  c=r['complete_value'].get('capacity_review')
  if c and Path(c).is_file():caps[c]=record(c)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),required_full_handoffs=required,complete_current_pointer=pointer,complete_later_closing_notes_and_snapshots=later,current_primary_pending=pending,Hancom_capacity_reviews_first_complete_read=caps,current_real_serial_and_next_original_frontend=actual,all_required_bytes_read_before_current_process_inspection=True),ensure_ascii=False))

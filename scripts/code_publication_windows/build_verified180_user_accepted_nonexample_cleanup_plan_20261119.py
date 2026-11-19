from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();a=json.loads((P/'R_user_verified180_accepted_nonexample_fresh_discovery_20261119.json').read_bytes())
assert a['no_deletion'] and a['candidate_count']==4972
cfg=dict(version=1,expected_done_count=180,latest_actual_user_authorization=dict(user_message_verbatim=a['latest_user_verbatim'],source='Latest direct user message in the original910b thread',scope='Only independently accepted complete200 nonexample matrices and truly unnecessary temporary files; all full reports/source/config/logs/needed numerical examples remain',serial_continue_immediately_after_actual_release=True),discovery_SHA256=a['discovery_SHA256'],prior_helper_SHA256=a['prior_pure_helper_source_SHA256'],allowed_model_ids=[s['model_id'] for s in a['scopes']],keep_original_logical_indices=a['keep_original_logical_indices'],max_remove_files=a['candidate_count'],max_remove_bytes=a['candidate_bytes'],known_no_NN_waiters=[dict(PID=1557351,start_ticks=446651691,cwd='/root/projects/comperror',relative='scripts/continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py',source_SHA256='2059c83e466d0feb78af89d7cc7a7ed041c75370a417aefd10a4cb61b8790252',state='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119/state.json',allowed_idle_phases=['wait_for_single_model_complete_original_capacity','wait_for_actual_NN_NF_FLOCK_release']),dict(PID=1530341,start_ticks=440002754,cwd='/root',relative='scripts/continue_glm46flash_completed_true3_wait_actual_capacity_resume_fresh3_fixed200_v3_1723_20261118.py',source_SHA256='96dfdd09b7b14a4845e142cbcc6dcc4384130a362ff1bea777f9fcf2a059637b',state='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118/state.json',allowed_idle_phases=['waiting_for_actual_original_disk_capacity_and_all_NN_NF_FLOCK_release'])],preserve_all_full_reports_sources_configs_logs_snapshots_and_needed_examples=True,no_failed_small_incomplete_or_unverified_values=True,no_hot_edit_signal_or_model_guard_relaxation=True)
raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();(P/'verified180_accepted_nonexample_user_serial_continue_20261119.json').write_bytes(raw)
sources=[]
for name,relative in [('retain_verified180_current_accepted_nonexample_user_serial_20261119.py','scripts/maintenance/retain_verified180_current_accepted_nonexample_user_serial_20261119.py'),('verified180_accepted_nonexample_user_serial_continue_20261119.json','configs/retention/verified180_accepted_nonexample_user_serial_continue_20261119.json')]:
 b=(P/name).read_bytes();ast.parse(b) if name.endswith('.py') else json.loads(b);sources.append(dict(relative=relative,SHA256=sha(b),base64=base64.b64encode(b).decode()))
data=dict(sources=sources,discovery_SHA256=a['discovery_SHA256'])
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib,traceback
R=Path('/root/projects/comperror')
for x in DATA['sources']:
 p=R/x['relative'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];assert p.resolve().is_relative_to(R)
 ast.parse(b) if p.suffix=='.py' else json.loads(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.retain_verified180_current_accepted_nonexample_user_serial_20261119')
try:result=m.plan(DATA['discovery_SHA256'])
except Exception:
 (m.H/'plan_failure_preserved.txt').write_text(traceback.format_exc(),encoding='utf8');raise
print(json.dumps(result,ensure_ascii=False))
'''
ast.parse(remote);(P/'deploy_plan_verified180_user_accepted_nonexample_cleanup_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],model_scopes=len(cfg['allowed_model_ids']),candidate_count=cfg['max_remove_files'],candidate_bytes=cfg['max_remove_bytes']),ensure_ascii=False))

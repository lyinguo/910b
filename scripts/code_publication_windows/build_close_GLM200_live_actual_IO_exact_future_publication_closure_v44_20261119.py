from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda n:json.loads((P/n).read_bytes())
closed=read('R_close_GLM200_qualified181_AutoGLM_serial_all_sources_v43_20261119.json');parent=read('GLM200_qualified181_AutoGLM_serial_complete_late_plan_v43_20261119.json');assert sha(Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes())==parent['official_readback']['after_TOML_SHA256']
files=[('close_alignment_handoff_exact_ZaiGLM46_qualified181_dual_publication_only_v9_20261119.py','scripts/maintenance/close_alignment_handoff_exact_ZaiGLM46_qualified181_dual_publication_only_v9_20261119.py'),('read_GLM46_actual_model_IO_FDs_CPU_and_card4_stage_without_signal_20261119_remote.py','scripts/code_publication_windows/read_GLM46_actual_model_IO_FDs_CPU_and_card4_stage_without_signal_20261119_remote.py'),(Path(__file__).name,'scripts/code_publication_windows/'+Path(__file__).name)]
sources=[]
for n,rel in files:
 raw=(P/n).read_bytes();ast.parse(raw);sources.append(dict(relative=rel,SHA256=sha(raw),role='Actual one-second IO/CPU/FD phase audit or guarded future actual qualified dual-publication pointer transition; no active model/code/deadline change',base64=base64.b64encode(raw).decode()))
names=[n for n,_ in files]+['R_GLM46_current_actual_CPU_IO_FD_and_card4_stage_20261119.json','R_close_GLM200_qualified181_AutoGLM_serial_all_sources_v43_20261119.json','GLM46_live_actual200_to_qualified181_artifact_review_driver_20261119/all_actual_probe_lines.jsonl','GLM46_live_actual200_to_qualified181_artifact_review_driver_20261119/actual_probe_latest.json']
packets=[]
for n in names:
 raw=(P/n).read_bytes();assert len(raw)<8*1024**2;packets.append(dict(name=n,bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
plan={k:v for k,v in parent.items() if k not in ['local_original_packets','additional_existing_primary_sources','expected_pointer_SHA256','note_name','new_pending']};plan.update(expected_pointer_SHA256=closed['pointer_SHA256'],note_name='GLM200_actual_IO_FD_full_matrix_capture_and_exact_future_qualified181_transition_not_executed_20261119',local_original_packets=packets,additional_existing_primary_sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],new_pending='/root/projects/comperror/workspace/diagnostics/batch_910b/code_publication_source_layout_20261114/pending_after_daily_20261119_verified180_GLM200_actual_IO_phase_and_guarded_qualified181_closure_all_primary_v44.json')
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'GLM200_live_actual_IO_exact_future_publication_closure_late_plan_v44_20261119.json').write_bytes(raw);data=dict(sources=sources,plan_BASE64=base64.b64encode(raw).decode(),plan_SHA256=sha(raw))
payload='DATA='+repr(data)+'\n'+r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for v in DATA['sources']:
 p=R/v['relative'];b=base64.b64decode(v['base64'],validate=True);assert p.resolve().is_relative_to(R) and hashlib.sha256(b).hexdigest()==v['SHA256'];ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
raw=base64.b64decode(DATA['plan_BASE64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw);path=Path(plan['handoff_directory'])/'GLM200_live_actual_IO_exact_future_publication_closure_late_plan_v44_20261119.json';assert not path.exists();path.write_bytes(raw)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118');print(json.dumps(m.execute(path,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(payload);(P/'execute_close_GLM200_live_actual_IO_future_qualified181_transition_v44_20261119_remote.py').write_text(payload,encoding='utf8');print(json.dumps(dict(plan_SHA256=sha(raw),new_primary_sources=3,no_G_or_canonical_write=True)))

"""Transport exact source and evidence originals; payload helper is evidence only."""
import ast,base64,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
source='close_verified183_exact271_retention_Meta_frontend_and_official_readback_20261120.py'
primary={source:'scripts/maintenance/'+source,'build_stage_verified183_exact271_late_closing_20261120.py':'scripts/windows_910b_20261120/build_stage_verified183_exact271_late_closing_20261120.py','build_MetaMllama_own_frontend_meta_source_stage_20261120.py':'scripts/windows_910b_20261120/build_MetaMllama_own_frontend_meta_source_stage_20261120.py','read_MetaMllama_frontend_actual_20261120_remote.py':'scripts/maintenance/read_MetaMllama_frontend_actual_20261120_remote.py'}
existing=['scripts/maintenance/discover_verified183_current_accepted_correct_nonexamples_tmp_metadata_only_20261120.py','scripts/maintenance/review_verified183_correct_nonexample_foreign_FD_metadata_only_20261120.py','scripts/maintenance/retain_verified183_AutoGLM_two_current_accepted_correct_nonexamples_20261120.py','configs/retention/verified183_AutoGLM_two_current_accepted_correct_nonexamples_20261120.json','scripts/prepare_meta_mllama_own_original_HF_BF16_2048_frontend200_meta_sources_20261120.py']
payload={rel:dict(SHA256=sha((V/name).read_bytes()),base64=base64.b64encode((V/name).read_bytes()).decode()) for name,rel in primary.items()}
originals={}
for name in ['official_automation_910b_after_verified183_20261120.toml','official_automation_910b_verified183_requested_args_20261120.json','official_automation_910b_verified183_actual_tool_result_20261120.json','official_automation_910b_verified183_exact_readback_20261120.json']:
 raw=(V/name).read_bytes();originals[name]=dict(SHA256=sha(raw),base64=base64.b64encode(raw).decode())
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
payload=json.loads(base64.b64decode('SOURCES'));originals=json.loads(base64.b64decode('ORIGINALS'));existing=json.loads(base64.b64decode('EXISTING'))
for rel,rec in payload.items():
 raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];ast.parse(raw);p=R/rel;assert not p.exists()
for rel,rec in payload.items():
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(rec['base64']))
H=D/'heartbeat_handoffs'/ 'A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
incoming=H/'late_actual_notes/exact271_official_incoming_20261120';incoming.mkdir(exist_ok=False)
rows=[]
for name,rec in originals.items():
 p=incoming/name;raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];p.write_bytes(raw);rows.append(dict(path=str(p),name=name,SHA256=sha(p)))
src=[dict(relative=n,SHA256=sha(R/n),role='new reviewed exact retention / own frontend / late handoff or real Windows consumer') for n in [*existing,*payload]]
plan=incoming/'exact_late_closing_plan.json';plan.write_text(json.dumps(dict(new_primary_sources=src,official_original_records=rows),ensure_ascii=False,indent=2)+'\\n')
cp=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/SOURCENAME'),str(plan),sha(plan)],cwd=R,capture_output=True,text=True)
result=dict(returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr,plan=str(plan),plan_SHA256=sha(plan));(incoming/'actual_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(result,ensure_ascii=False));sys.exit(cp.returncode)
'''.replace('SOURCES',base64.b64encode(json.dumps(payload).encode()).decode()).replace('ORIGINALS',base64.b64encode(json.dumps(originals).encode()).decode()).replace('EXISTING',base64.b64encode(json.dumps(existing).encode()).decode()).replace('SOURCENAME',source)
ast.parse(remote);(V/'stage_verified183_exact271_late_closing_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n')
print('Stage ready; only diagnostic closing/pending/pointer and exact new source files will be written.')

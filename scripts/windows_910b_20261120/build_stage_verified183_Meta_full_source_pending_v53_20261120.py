"""Archive useful actual source/consumers and bounded late evidence, not transport."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
names=['append_verified183_Meta_actual_full_source_pending_20261120.py','build_stage_verified183_Meta_full_source_pending_v53_20261120.py','build_meta_mllama_own_original_SDPA_2048_full_observer_sources_20261120.py','build_meta_mllama_own_original_SDPA_2048_fullNF_20261120.py','build_stage_MetaMllama_own_SDPA2048_serial_controller_20261120.py','build_meta_mllama_fullNF_v2_exact_original_answer_instruction_metadata_20261120.py','build_stage_meta_mllama_formal_NF2_same_genuine3_no_NN_replay_20261120.py','build_MetaMllama_NF3_exact_original_power_reference_no_model_replay_20261120.py','build_stage_meta_mllama_formal_NF3_same_genuine3_no_NN_replay_20261120.py','read_MetaMllama_serial_actual_process_FLOCK_progress_20261120_remote.py','read_MetaMllama_current_installed_default_attention_meta_no_NN_20261120_remote.py','read_MetaMllama_original_SDPA_scale_and_captured_arguments_no_NN_20261120_remote.py','read_frozen_report_contract_static_AST_v2_20261120_remote.py','read_verified183_existing_retention_exclusion_reasons_only_20261120_remote.py']
payload={}
for name in names:
 raw=(V/name).read_bytes();ast.parse(raw);folder='scripts/maintenance/' if name.startswith(('append_','read_')) else 'scripts/windows_910b_20261120/';payload[folder+name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
source_inputs=json.loads((V/'MetaMllama_own_SDPA2048_serial_exact_sources_20261120.json').read_bytes())
extra=[*source_inputs,'scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v2_exact_instruction_metadata_20261120.py','scripts/continue_meta_mllama_completed_genuine3_formal_NF2_exact_instruction_no_NN_replay_20261120.py','configs/queues/MetaMllama_completed_genuine3_formal_NF2_exact_instruction_no_NN_replay_capacity_serial_20261120.json','scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py','scripts/continue_meta_mllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_20261120.py','configs/queues/MetaMllama_completed_genuine3_formal_NF3_exact_instruction_no_NN_replay_capacity_serial_20261120.json']
evidence_names=['MetaMllama_native_source_derivation_review_20261120.json','MetaMllama_fullNF_v2_exact_instruction_derivation_review_20261120.json','MetaMllama_NF3_exact_original_power_derivation_review_20261120.json','R_MetaMllama_original_SDPA_exact_scale_control_20261120.json','R_MetaMllama_installed_default_attention_meta_20261120.json','R_frozen_report_contract_actual_support_v3_20261120.json','R_MetaMllama_serial_actual9_20261120.json']
originals={name:dict(SHA256=hashlib.sha256((V/name).read_bytes()).hexdigest(),base64=base64.b64encode((V/name).read_bytes()).decode()) for name in evidence_names}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
payload=json.loads(base64.b64decode('SOURCES'));extra=json.loads(base64.b64decode('EXTRA'));originals=json.loads(base64.b64decode('ORIGINALS'))
for rel,rec in payload.items():
 raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];ast.parse(raw);assert not (R/rel).exists()
for rel,rec in payload.items():
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(rec['base64']))
ptr=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(ptr)=='02b13c07c1e6d61d1c0cdb7543ba2c05decaabfffebd2b2c3e6b5c37d7111c88';H=Path(json.loads(ptr.read_bytes())['handoff_directory']);incoming=H/'late_actual_notes/Meta_full_source_pending_v53_incoming_20261120';incoming.mkdir(exist_ok=False)
records=[]
for name,rec in originals.items():
 raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];p=incoming/name;p.write_bytes(raw);records.append(dict(path=str(p),name=name,SHA256=sha(p)))
sources=[dict(relative=n,SHA256=sha(R/n),role='actual own Meta runtime/complete independent verification/control/Windows consumer/late handoff source') for n in [*extra,*payload]]
plan=incoming/'source_pending_plan.json';plan.write_text(json.dumps(dict(expected_pointer_SHA256=sha(ptr),late_note_name='Meta_own_original_SDPA2048_genuine3_formal_NF3_all_sources_20261120',pending_name='pending_after_daily_20261120_verified183_Meta_ownSDPA2048_NF1_NF2_NF3_all_primary_v53.json',new_primary_sources=sources,original_records=records),ensure_ascii=False,indent=2)+'\\n')
cp=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/append_verified183_Meta_actual_full_source_pending_20261120.py'),str(plan),sha(plan)],cwd=R,capture_output=True,text=True);v=dict(returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr,plan=str(plan),plan_SHA256=sha(plan));(incoming/'actual_execution.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n');print(json.dumps(v,ensure_ascii=False));sys.exit(cp.returncode)
'''.replace('SOURCES',base64.b64encode(json.dumps(payload).encode()).decode()).replace('EXTRA',base64.b64encode(json.dumps(extra).encode()).decode()).replace('ORIGINALS',base64.b64encode(json.dumps(originals).encode()).decode())
ast.parse(remote);(V/'stage_verified183_Meta_full_source_pending_v53_20261120_remote.py').write_text(remote,encoding='utf8',newline='\n');print('Explicit Meta primary source archive ready; no model/G/source mutation.')

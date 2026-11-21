"""Declare exact new sources and evidence for one compact successor handoff."""
import ast,base64,hashlib,json,tomllib
from pathlib import Path
from datetime import datetime,timezone
V=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):
 p=Path(p);b=p.read_bytes();s=p.stat();return dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');export=Path('D:/code-lv/.tmp/batch_910b_20261030/AutoGLMPhone9BSingle_verified183_incremental_real_results_export_20261120.json')
assert sha(book)=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81' and sha(export)=='59400a4f9eeacd2746088fab2952957f5b1802ef38ee9b415aac6e2f6a49e5a0'
fresh=dict(UTC=datetime.now(timezone.utc).isoformat(),local_workbook=rec(book),local_current_export=rec(export),current_verified_count=183,no_local_book_or_export_write=True)
(V/'local_verified183_before_handoff_fresh_SHA_stat_20261120.json').write_text(json.dumps(fresh,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
auto=Path('C:/Users/lyg/.codex/automations/910b/automation.toml');raw=auto.read_bytes();a=tomllib.loads(raw.decode());(V/'official_automation_910b_before_verified183_20261120.toml').write_bytes(raw)
before=dict(id=a['id'],name=a['name'],status=a['status'],rrule=a['rrule'],target_thread_id=a.get('target_thread_id'),kind=a.get('kind'),notificationPolicy=a.get('notification_policy',None))
assert before['id']=='910b' and before['status']=='ACTIVE' and before['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300' and before['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
(V/'official_automation_910b_before_verified183_fields_20261120.json').write_text(json.dumps(dict(before_TOML_SHA256=hashlib.sha256(raw).hexdigest(),before_prompt_SHA256=hashlib.sha256(a['prompt'].encode()).hexdigest(),preserved_fields=before),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
tag=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=D+'/heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_'+tag
incoming=B+'/verified183_compact_handoff_declared_local_source_originals_'+tag
primary=[];payload={}
def add(local,rel,role):
 p=V/local;b=p.read_bytes()
 if p.suffix=='.py':ast.parse(b)
 target=R+'/'+rel;payload[target]=dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode(),only_new_or_exact_same=True)
 primary.append(dict(relative=rel,SHA256=hashlib.sha256(b).hexdigest(),role=role))
for name in ['apply_AutoGLMPhone9BSingle_verified183_exact_server_bytes_local_CAS_20261120.py','autoglm_phone_single_workbook_cell_verified183_20261120.mjs','build_AutoGLMPhone9BSingle_verified183_exact_local_receipt_upload_20261120.py','render_AutoGLMPhone9BSingle_verified183_actual_published_20261120.mjs','stage_and_launch_AutoSingle_verified183_reviewed_dual_publication_20261120.py','build_glm41thinking_own_native_fullNF_sources_20261120.py','build_GLM41Thinking_original_added_tokens_frontend_v2_20261120.py','build_GLM41Thinking_actual_MediaWithBytes_frontend_v3_20261120.py','build_GLM41Thinking_own_frontend_stage_launch_20261120.py','build_GLM41Thinking_owned_sources_frontend_v2_stage_20261120.py','build_GLM41Thinking_owned_media_v3_stage_20261120.py','build_GLM41Thinking_own_serial_true3_capacity_fresh3_fixed200_20261120.py','build_GLM41Thinking_serial_controller_exact_stage_launch_20261120.py','inspect_GLM41Thinking_owned_native_contract_20261120.py','build_stage_compact_verified183_current_source_archive_20261120.py']:
 add(name,'scripts/windows_910b_20261120/'+name,'review_next_Day21_actual_Windows_publication_or_own_model_source_preparation_consumer')
for name in ['read_AutoSingle_verified183_publication_and_followup_actual_20261120_remote.py','read_AutoGLMPhone9BSingle_verified183_server_book_for_local_CAS_20261120_remote.py','read_GLM41Thinking_own_saved_assets_and_native_VLLM_input_contract_no_NN_20261120_remote.py','read_GLM41Thinking_own_tokenizer_native_added_tokens_no_NN_20261120_remote.py','read_GLM41Thinking_serial_actual_progress_20261120_remote.py','read_GLM41Thinking_current_model_loading_CPU_FD_context_20261120_remote.py','read_installed_native_media_wrapper_exact_source_20261120_remote.py','run_frozen_incremental_monitor_full_latest_checkpoint_20261120_remote.py','create_compact_verified183_GLM41Thinking_current_handoff_20261120.py']:
 add(name,'scripts/maintenance/'+name,'actual_used_readonly_evidence_or_compact_handoff_maintenance_source')
native=[
 ('prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_20261120.py','scripts/'),
 ('prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v2_own_added_tokens_20261120.py','scripts/'),
 ('prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py','scripts/'),
 ('prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py','scripts/'),
 ('continue_GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.py','scripts/'),
 ('glm41thinking_owned_tokenizer_config_20261120.py','universal_eval/model/reference_sources/'),
 ('glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend.py','universal_eval/model/adapters/'),
 ('glm41thinking_own_original_VLLM_BF16_8192_20261120_frontend_v2_actual_media.py','universal_eval/model/adapters/'),
 ('glm41thinking_own_original_VLLM_BF16_8192_20261120_native.py','universal_eval/model/adapters/'),
 ('glm41thinking_own_original_VLLM_BF16_8192_20261120_native_v2_actual_media.py','universal_eval/model/adapters/'),
 ('glm41thinking_own_original_VLLM_BF16_8192_20261120_MC.py','universal_eval/algorithm/'),
 ('verify_glm41thinking_own_original_VLLM_BF16_8192_native_v1_20261120.py','scripts/'),
 ('verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py','scripts/'),
 ('GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120.json','configs/queues/'),
 ('publish_AutoGLMPhone9BSingle_verified183_single_original_lock_dual_transaction_v1_20261120.py','scripts/'),
 ('run_AutoGLMPhone9BSingle_verified183_single_lock_dual_publication_20261120.py','scripts/')]
for name,prefix in native:add(name,prefix+name,'own_runtime_configuration_producer_verifier_or_exact_publication_primary')
originals=[]
names=['official_automation_910b_before_verified183_20261120.toml','official_automation_910b_before_verified183_fields_20261120.json','local_verified183_before_handoff_fresh_SHA_stat_20261120.json','AutoGLMPhone9BSingle_verified183_publication_source_scope_20261120.json','GLM41Thinking_own_native_source_diff_scope_20261120.json','prepare_AutoSingle_verified183_exact_publication_and_local_artifact_20261120.py','stage_GLM41Thinking_own_frontend_launch_20261120_remote.failed_v1.py','R_GLM41Thinking_own_frontend_launch_20261120.json.stderr','R_GLM41Thinking_serial_actual_progress3_20261120.json.stderr','R_native_media_wrapper_exact_source_20261120.json.stderr','R_GLM41Thinking_own_frontend_current2_20261120.json','R_GLM41Thinking_own_frontend_v2_current1_20261120.json','R_GLM41Thinking_own_frontend_v3_current1_20261120.json','R_GLM41Thinking_own_tokenizer_native_added_tokens_20261120.json','R_native_media_wrapper_exact_source_v2_20261120.json','R_GLM41Thinking_serial_actual_progress5_20261120.json','R_GLM41Thinking_model_loading_CPU_FD_context_20261120.json','R_frozen_incremental_monitor_verified183_GLM41Thinking_20261120.json','R_AutoSingle_verified183_publication_actual3_20261120.json','R_GLM41Thinking_serial_controller_exact_launch_20261120.json']
names.append('R_create_compact_verified183_current_handoff_20261120.json')
localpub=V/'AutoGLMPhone9BSingle_local_result_sync_verified183_20261120'
for p in [*(V/name for name in names),*(localpub/name for name in ['local_publication_receipt.json','actual_PNG_visual_review.json','actual_published_PNG_review.json','published_after_692.png','before_692.png','after_692.png','reviewed_inputs.json'])]:
 assert p.is_file(),str(p);key=p.name if p.parent==V else 'local_delivery/'+p.name;b=p.read_bytes();target=incoming+'/'+key;payload[target]=dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode(),evidence_only=True);originals.append(dict(name=key,path=target,SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
text=V/'verified183_compact_full_handoff_template_20261120.txt';target=incoming+'/handoff.txt';b=text.read_bytes();payload[target]=dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode(),evidence_only=True)
latest=json.loads((V/'R_AutoSingle_verified183_publication_actual3_20261120.json').read_bytes())
current=json.loads((V/'R_GLM41Thinking_serial_actual_progress5_20261120.json').read_bytes());protected=[json.loads(v['complete_text'])['protected_canonical_hashes'] for k,v in current['files'].items() if v and k.endswith('/job.json')];assert len(protected)==1;canonical=protected[0]
assert all(canonical[k]==h for k,h in latest['canonical_SHA256'].items()) and len(canonical)==9
plan=dict(handoff_directory=H,incoming_originals_root=incoming,expected_pointer_SHA256='c5dacf61d6c7a9b0c62dbcd747dbdd5ec3b7f7fdafc1d851d2ab43396e36bf35',canonical_SHA256=canonical,full_handoff_text=target,full_handoff_text_SHA256=sha(text),additional_primary_sources=primary,local_original_records=originals,new_pending_name='pending_after_daily_20261120_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_all_primary_v51.json')
plan_bytes=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();plan_path=incoming+'/plan.json';payload[plan_path]=dict(SHA256=hashlib.sha256(plan_bytes).hexdigest(),base64=base64.b64encode(plan_bytes).decode(),evidence_only=True)
remote=r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
items=json.loads(base64.b64decode('PAYLOAD'))
for name,rec in items.items():
 p=Path(name);raw=base64.b64decode(rec['base64']);assert hashlib.sha256(raw).hexdigest()==rec['SHA256'];assert p.is_relative_to(R)
 if name.endswith('.py') and not rec.get('evidence_only'):ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw,name
for name,rec in items.items():
 p=Path(name)
 if p.exists():continue
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(rec['base64']))
source=R/'scripts/maintenance/create_compact_verified183_GLM41Thinking_current_handoff_20261120.py'
q=subprocess.run(['python3','-B',str(source),'PLAN_PATH','PLAN_SHA'],cwd=R,capture_output=True,text=True,timeout=55,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps(dict(actual_returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,handoff_directory='HANDOFF',no_canonical_or_live_model_changes=True),ensure_ascii=False));sys.exit(q.returncode)
'''.replace('PAYLOAD',base64.b64encode(json.dumps(payload).encode()).decode()).replace('PLAN_PATH',plan_path).replace('PLAN_SHA',hashlib.sha256(plan_bytes).hexdigest()).replace('HANDOFF',H)
ast.parse(remote);helper=V/'stage_create_compact_verified183_handoff_20261120_remote.py';helper.write_text(remote,encoding='utf8',newline='\n')
(V/'verified183_compact_handoff_declared_plan_local_20261120.json').write_bytes(plan_bytes)
print(json.dumps(dict(remote_helper=str(helper),handoff_directory=H,declared_primary_sources=len(primary),local_original_records=len(originals),plan_SHA256=hashlib.sha256(plan_bytes).hexdigest(),before_official_fields=before),ensure_ascii=False))

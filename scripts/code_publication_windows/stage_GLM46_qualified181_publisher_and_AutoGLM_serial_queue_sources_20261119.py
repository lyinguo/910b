"""Stage reviewed new sources/config, start only the non-NN serial queue observer."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent
def sha(v):return hashlib.sha256(v).hexdigest()
def item(name,relative=None):
 raw=(P/name).read_bytes()
 if name.endswith('.py'):ast.parse(raw)
 return dict(relative=relative or 'scripts/code_publication_windows/'+name,SHA256=sha(raw),base64=base64.b64encode(raw).decode())
pubscope=json.loads((P/'ZaiGLM46_verified181_publication_sources_scope_review_20261119.json').read_bytes())
pub='publish_ZaiGLM46Flash_verified181_single_original_lock_dual_transaction_v1_20261119.py'
watch='watch_GLM46_actual_completed200_then_qualified_dual_commit_or_preserved_gap_start_AutoGLMPhone_serial_20261119.py'
own=json.loads((P/'AutoGLMPhone9B_own_native_fullNF_source_scope_review_20261119.json').read_bytes())
cfg=dict(max_concurrent_models=1,qualifying_GLM_requires_actual_dual_publication_before_next_NN=True,completed_true3_or200_replay_forbidden=True,original_reports_sources_configs_logs_and_required_numeric_examples_preserved=True,current_GLM_job_SHA256='18c9d24f9992311ea3261e82cc56c3121db03ccd0655e21aaafc4d2b5ee4f323',frozen_runner_SHA256='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90',qualified_GLM_publisher_SHA256=sha((P/pub).read_bytes()),before_verified180_workbook_SHA256='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34',own_next_model_id='zai-org/AutoGLM-Phone-9B-Multilingual',own_next_readiness_preparer='scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_true3.py',own_next_source_SHA256={x['relative']:x['SHA256'] for x in own['new_sources']},own_next_input200_SHA256='21e7a17a36e7ca4a0a5689c5be4e981704a00cb91b463017f759b2f6fc6e1dd4',queue_waiting_is_not_model_NN_or_fullNF=True,own_next_first200_requires_genuine_own3_fullNF_complete_ZIP_all200_shapes_capacity_and_internal_measurement_approval=True,no_weight_download_or_package_install_or_source_math_precision_budget_extraction_change=True,no_NPU0_access_or_signal_or_live_job_edit=True)
cfg['own_next_source_SHA256'].update({'scripts/prepare_autoglm_phone9b_own_original_BF16_2048_20261119_frontend200.py':'9f0d561e43e673ec075c3454811db8777115d590dcbe4c856b064109691e8224','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py':'a858f771182c1e8240e8e94d158d6089a4c7ee421b7cae94ab4301c9fb2be485','scripts/glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py':'2a86561bc32852de4dcc04179699dc8609eba4ce3c61217cda92e85ad9909301'})
cfgname='after_GLM46_actual200_qualified_dual_commit_or_preserved_gap_AutoGLMPhone_serial_20261119.json';raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();(P/cfgname).write_bytes(raw)
items=[item(pub,'scripts/'+pub),item(watch,'scripts/'+watch),dict(relative='configs/queues/'+cfgname,SHA256=sha(raw),base64=base64.b64encode(raw).decode())]
reviewed=['prepare_ZaiGLM46_verified181_guarded_dual_publication_sources_20261119.py','prepare_AutoGLMPhone_own_original_BF16_2048_frontend_sources_20261119.py','prepare_AutoGLMPhone9B_own_native_fullNF_serial_readiness_sources_20261119.py','test_AutoGLMPhone9B_own_binding_complete_audit_storage_and_dispatch_sources_20261119.py','test_GLM46_to_AutoGLM_serial_queue_full200_three_D_gate_20261119.py','read_AutoGLMPhone_own_processor_dataset_and_three_D_input_contract_20261119_remote.py','read_AutoGLMPhone_own_original_config_source_and_SF_headers_only_for_serial_next_20261119_remote.py','read_existing_pending_original_config_architecture_meta_for_serial_next_no_model_load_20261119_remote.py',Path(__file__).name]
reviewed+=['prepare_ZaiGLM46_verified181_local_baseline_preview_no_predicted_score_20261119.py','prepare_ZaiGLM46_verified181_completed200_actual_score_local_plan_20261119.py','apply_ZaiGLM46_verified181_exact_server_bytes_local_CAS_20261119.py','build_ZaiGLM46_verified181_exact_local_receipt_upload_20261119.py','build_launch_ZaiGLM46_verified181_actual_reviewed_publication_20261119.py','read_ZaiGLM46_verified181_server_book_for_local_CAS_20261119_remote.py','run_ZaiGLM46_verified181_reviewed_dual_publication_actual_20261119.py','render_ZaiGLM46_verified181_actual_published_20261119.mjs','read_ZaiGLM46_completed200_exact_publication_inputs_no_live_assumptions_20261119_remote.py','zai_glm46_workbook_cell_verified181_20261119.mjs']
items.extend(item(x) for x in reviewed)
data=dict(items=items,queue_config='configs/queues/'+cfgname,queue_source='scripts/'+watch)
payload='DATA='+repr(data)+'\n'+r'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
for v in DATA['items']:
 p=R/v['relative'];b=base64.b64decode(v['base64'],validate=True);assert p.resolve().is_relative_to(R) and hashlib.sha256(b).hexdigest()==v['SHA256']
 if p.suffix=='.py':ast.parse(b)
 if p.exists():assert p.read_bytes()==b,(str(p),'New source unexpectedly changed; never overwrite live source')
for v in DATA['items']:
 p=R/v['relative']
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v['base64'],validate=True))
cfg=json.loads((R/DATA['queue_config']).read_bytes())
for name,h in cfg['own_next_source_SHA256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
for name in ['universal_eval.model.adapters.autoglm_phone9b_own_original_BF16_2048_20261119_native','universal_eval.model.adapters.autoglm_phone9b_own_original_BF16_2048_20261119_frontend','universal_eval.algorithm.autoglm_phone9b_own_original_BF16_2048_20261119_MC','scripts.verify_autoglm_phone9b_own_original_BF16_2048_20261119_native']:importlib.import_module(name)
H=B/'after_GLM46_actual200_qualified_dual_commit_or_preserved_gap_AutoGLMPhone_serial_20261119';L=B/'after_GLM46_actual200_AutoGLM_serial_queue_launch_evidence_20261119';assert not H.exists() and not L.exists();L.mkdir()
with (L/'queue_stdout.log').open('xb') as out,(L/'queue_stderr.log').open('xb') as err:
 child=subprocess.Popen(['python3','-B',str(R/DATA['queue_source'])],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,source=DATA['queue_source'],source_SHA256=hashlib.sha256((R/DATA['queue_source']).read_bytes()).hexdigest(),config=DATA['queue_config'],config_SHA256=hashlib.sha256((R/DATA['queue_config']).read_bytes()).hexdigest(),phase='read_only_serial_queue_observer_started_not_a_second_NN',own_module_imports_passed=True,no_original_GLM_model_source_job_YAML_or_deadline_edits=True,staged_sources=[{k:v for k,v in x.items() if k!='base64'} for x in DATA['items']]);(L/'launch.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(value,ensure_ascii=False))
'''
ast.parse(payload);(P/'run_stage_GLM46_qualified181_AutoGLM_serial_queue_actual_20261119_remote.py').write_text(payload,encoding='utf8',newline='\n');(P/'GLM46_qualified181_AutoGLM_serial_queue_new_source_scope_20261119.json').write_text(json.dumps({k:v for k,v in data.items() if k!='items'}|dict(items=[{k:v for k,v in x.items() if k!='base64'} for x in items]),indent=2)+'\n',encoding='utf8');print(json.dumps(dict(primary_sources=len(items),queue_source_SHA256=sha((P/watch).read_bytes()),queue_config_SHA256=sha(raw),no_NEW_model_or_G_writes=True)))

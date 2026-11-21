from pathlib import Path
import ast,base64,hashlib,json,tomllib
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda n:json.loads((P/n).read_bytes())
closed=read('R_close_verified182_Day20_GLM154_Single_serial_v48_20261120.json')
prior=read('verified182_Day20_GLM154_Single_serial_late_plan_v48_20261120.json')
before=(P/'official910b_before_stage_bound_reader_v5_and_Single_event_waiter_20261120.toml.original').read_bytes()
after=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes()
(P/'official910b_after_stage_bound_reader_v5_and_Single_event_waiter_20261120.toml.original').write_bytes(after)
a,b=tomllib.loads(before.decode()),tomllib.loads(after.decode())
keep=lambda v:{k:x for k,x in v.items() if k not in ('prompt','updated_at')}
args=read('official910b_verified182_Single_stage_bound_reader_v5_event_waiter_arguments_20261120.json')
assert keep(a)==keep(b) and b['prompt']==args['prompt'].rstrip('\n') and b['status']=='ACTIVE'
official=dict(before_preserved_fields=keep(a),after_preserved_fields=keep(b),before_TOML_SHA256=sha(before),after_TOML_SHA256=sha(after),before_prompt_SHA256=sha(a['prompt'].encode()),after_prompt_SHA256=sha(b['prompt'].encode()),requested_prompt_normalized_SHA256=sha(args['prompt'].rstrip('\n').encode()),official_update_succeeded=True,no_raw_active_TOML_write=True,source_bound_reader_corrected_without_model_or_job_change=True)
(P/'official910b_verified182_stage_bound_v5_event_waiter_exact_readback_20261120.json').write_text(json.dumps(official,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sources={};install=[]
for n in ['read_AutoGLMPhone9BSingle_current_bound_report_committed_progress_v4_20261120_remote.py','read_AutoGLMPhone9BSingle_bound_stage_no_stale_fresh3_report_v5_20261120_remote.py','read_verified182_next_own_weight_pending_GLM_family_metadata_no_NN_20261120_remote.py','read_GLM41VBase_own_original_probability_adapter_assets_headers_next_serial_no_NN_20261120_remote.py',Path(__file__).name]:
 raw=(P/n).read_bytes();ast.parse(raw);v=dict(relative='scripts/code_publication_windows/'+n,SHA256=sha(raw),role='Actual current-stage-bound read-only progress / own next-model protocol metadata or official exact readback handoff builder; no NN or G writer')
 sources[v['relative']]=v;install.append(dict(v,base64=base64.b64encode(raw).decode()))
for rel,n in [('scripts/continue_after_AutoGLMSingle_actual_fixed200_outcome_dual_if_qualified_then_original_resources_serial_20261120.py','continue_after_AutoGLMSingle_actual_fixed200_outcome_dual_if_qualified_then_original_resources_serial_20261120.py'),('configs/queues/after_AutoGLMSingle_actual_fixed200_conditional_dual_then_original_resources_serial_20261120.json','after_AutoGLMSingle_actual_fixed200_conditional_dual_then_original_resources_serial_20261120.json')]:
 raw=(P/n).read_bytes();sources[rel]=dict(relative=rel,SHA256=sha(raw),role='Actual sparse-state read-only completed200 / conditional real dual publication then original-resource event continuation, max1, no model replay/G writes')
originals=[*list(n for n in [Path(v['relative']).name for v in install]),'R_launch_AutoGLMSingle_actual200_conditional_dual_original_resources_waiter_20261120.json','official910b_before_verified182_Day20_GLM154_Single_serial_retry_20261120.toml.original','official910b_after_verified182_Day20_GLM154_Single_serial_retry_20261120.toml.original','official910b_verified182_Day20_GLM154_Single_serial_retry_arguments_20261120.json','official910b_verified182_Day20_Single_serial_retry_result_20261120.json','official910b_verified182_Day20_Single_serial_exact_readback_20261120.json','official910b_before_stage_bound_reader_v5_and_Single_event_waiter_20261120.toml.original','official910b_after_stage_bound_reader_v5_and_Single_event_waiter_20261120.toml.original','official910b_verified182_Single_stage_bound_reader_v5_event_waiter_arguments_20261120.json','official910b_verified182_stage_bound_reader_v5_event_waiter_result_20261120.json','official910b_verified182_stage_bound_v5_event_waiter_exact_readback_20261120.json','R_AutoGLMPhone9BSingle_bound_current_progress_v4_actual4_20261120.json','R_AutoGLMPhone9BSingle_bound_current_progress_v4_actual5_20261120.json','R_AutoGLMPhone9BSingle_bound_stage_v5_actual1_20261120.json','R_AutoGLMPhone9BSingle_bound_stage_v5_actual2_20261120.json','R_verified182_next_own_weight_pending_GLM_family_metadata_no_NN_20261120.json','R_GLM41VBase_own_original_probability_adapter_assets_headers_next_serial_no_NN_20261120.json']
packets=[]
for n in dict.fromkeys(originals):
 raw=(P/n).read_bytes();assert len(raw)<8*1024**2
 packets.append(dict(name=n,bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
plan={k:v for k,v in prior.items() if k not in ('local_original_packets','additional_existing_primary_sources')}
plan.update(expected_pointer_SHA256=closed['pointer_SHA256'],official_readback=official,note_name='official_verified182_Day20_Single_fresh3_pass_fixed200_stage_bound_current_event_waiter_all_sources_v49_20261120',local_original_packets=packets,additional_existing_primary_sources=list(sources.values()),new_pending='/root/projects/comperror/workspace/diagnostics/batch_910b/code_publication_source_layout_20261114/pending_after_daily_20261120_verified182_Single_fresh3_pass_fixed200_event_waiter_stage_bound_all_primary_v49.json')
B='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
j=B+'/AutoGLMPhone9BSingle_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261120';w=B+'/after_AutoGLMSingle_actual_fixed200_conditional_dual_then_original_resources_serial_20261120'
plan['current_stage_files'].update({'Single/new_fresh3_accepted':j+'/fresh3/accepted.json','Single/readonly_event_waiter_state':w+'/state.json','Single/readonly_event_waiter_actual':w+'/actual_completion.json','Single/readonly_event_waiter_failure':w+'/actual_failure.json'})
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();name='official_verified182_Single_stage_bound_event_waiter_late_plan_v49_20261120.json';(P/name).write_bytes(raw)
data=dict(install=install,plan_name=name,plan_base64=base64.b64encode(raw).decode(),plan_SHA256=sha(raw))
code='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for v in DATA['install']:
 p=R/v['relative'];raw=base64.b64decode(v['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['SHA256'] and p.resolve().is_relative_to(R);ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw);p=Path(plan['handoff_directory'])/DATA['plan_name'];assert not p.exists();p.write_bytes(raw)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118');print(json.dumps(m.execute(p,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(code);(P/'execute_close_official_verified182_Single_stage_bound_event_waiter_v49_20261120_remote.py').write_text(code,encoding='utf-8',newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw),official_TOML_SHA256=sha(after),prompt_SHA256=official['after_prompt_SHA256'],additional_sources=len(sources),archival_original_packets=len(packets))))

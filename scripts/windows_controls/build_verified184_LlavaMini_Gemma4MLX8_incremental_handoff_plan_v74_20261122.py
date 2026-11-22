"""Build an explicit incremental handoff plan from separate evidence/templates.
Rendered diagnostics and transports remain evidence, never publication source.
"""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
lists=['Gemma4MLX8_own_frontend_native_primary_paths_20261122.json','Gemma4MLX8_own_serial_true3_primary_paths_20261122.json','Gemma4MLX8_completed3_profiler_NF2_formal_serial_primary_paths_20261122.json','LlavaMini_single_JSON_formal_acceptance_capacity_v3_primary_paths_20261122.json']
names=set()
for name in lists:
 for row in json.loads((V/name).read_bytes()):names.add(row if isinstance(row,str) else row['path'])
names.add('scripts/maintenance/read_Gemma4MLX8_formal_NF2_serial_recovery_v4_committed_trace_20261122_remote.py')
names.update(['scripts/continue_LlavaMini_completed_formal_fullNF_only_capacity_v4_zipfile_20261122.py','scripts/maintenance/read_LlavaMini_completed3_full_numeric_proof_and_exact_stdout_framing_20261122_remote.py','scripts/maintenance/read_LlavaMini_formal_single_JSON_acceptance_capacity_actual_20261122_remote.py','scripts/maintenance/read_LlavaMini_completed_formal_fullNF_capacity_v4_actual_20261122_remote.py','scripts/maintenance/read_next_MedGemmaPT_and_Gemma4E4B_MLX8_own_saved_deployment_headers_no_NN_20261122.py','scripts/maintenance/read_Gemma4_native_processor_token_expansion_and_saved_generation_no_NN_20261122_remote.py','scripts/maintenance/read_Gemma4MLX8_current_serial_proc_FLOCK_attempt_full_small_20261122_remote.py','scripts/maintenance/read_Gemma4MLX8_current_serial_v2_console_report_FD_20261122_remote.py','scripts/maintenance/read_Gemma4MLX8_formal_NF2_serial_recovery_current_20261122_remote.py','scripts/maintenance/read_Gemma4MLX8_completed3_actual_profiler_schema_unique_tasks_20261122_remote.py','scripts/maintenance/read_next_native_Llava_owned_headers_saved_deployments_no_NN_20261122.py','scripts/maintenance/record_verified184_LlavaMini_formal_capacity_Gemma4MLX8_serial_full_handoff_v74_20261122.py','scripts/windows_controls/build_Gemma4MLX8_own_Transformers_frontend_and_native_sources_20261122.py','scripts/windows_controls/'+Path(__file__).name])
base=(V/'full_actual_verified184_Day22_pushed_LlavaMini_own_true3_fullNF_current_template_20261122.txt').read_text(encoding='utf-8')
sections=(V/'verified184_LlavaMini_completed_and_Gemma4MLX8_serial_sections_v74_20261122.txt').read_text(encoding='utf-8')
llava,gemma=sections.split('---GEMMA---\n');start=base.index('【本轮新LLaVA-Mini');end=base.index('【Ovis已完整通过',start)
full=base[:start]+llava+'\n'+gemma+'\n'+base[end:]
full=full.replace(base.splitlines()[0],'完整当前交接：verified184保持，LLaVA-Mini真3/6已正式接收但完整200容量拒；自己的Gemma4 MLX8名下实际BF16真3/24已正式NF2通过并按新批准fresh3/fixed200串行继续，真实退出优先。Day22唯一代码已正常push，不重复已完成阶段。')
full=full.replace('直接父全文ba63fbb140a722b22d08159cfbccc2b9a55279f34fc97eead89c933d6c04f188及其父7ab917a8b39cbafdd7ca0ee514692e372345a0e0b73995f339fff105f83be213','直接父全文008d90cc0b81d4563616fc8411392d95d474fcf0ae396cdd1da707a06a1100c7，其父ba63fbb140a722b22d08159cfbccc2b9a55279f34fc97eead89c933d6c04f188与7ab917a8b39cbafdd7ca0ee514692e372345a0e0b73995f339fff105f83be213')
full=full.replace('自己LLaVA-Mini实际3/6/当前全数值NF以最新snapshot和actual退出优先','LLaVA-Mini已完成3/6正式数值接收、200约3.104TiB容量拒，自己的Gemma4MLX8真实fresh3/fixed200以最新snapshot和actual退出优先')
full=full.replace('本轮仅新增上述117精确清理；未删任何临时/日志或未验/失败/小样本','继承父117精确清理；当前本轮新增删除0，未删任何临时/日志或未验/失败/小样本')
full=full.replace('自己LLaVA-Mini实际3/6/当前全数值NF','自己的Gemma4 MLX8实际NEWfresh3/fixed200串行')
footer=full.index('【本轮更晚实际优先】')
full=full[:footer]+'''【本轮更晚实际优先】下列为当前真实串行进程/原内核FLOCK/完成样本和前向/容量的摘要，完整原始JSON和完整日志单独逐SHA保存在所指actual_snapshot，避免无限拼接历史。当前模型进程存在可能包括加载或完整值保存，不代表持续提交NPU；真正NN以前向/当前PID的trace和真实kernel为准。正式CPU独验为NF，前端/meta/压缩/等候器/封装metadata不充NN或新的数值重放。旧job已失败不改0，新的独立正式验收与原genuineNN精确绑定、无NN重放。仅official工具维护原910b ACTIVE/原名称/原聊天/300分钟，活TOML只读；更新前后原字节和实际回读留证，最新closing/pointer/proc/actual退出优先。
CURRENT_ACTUAL_JSON
'''
assert '正在正式完整数值验收' not in full and 'CURRENT_SOURCE_COUNT' in full and 'CURRENT_ACTUAL_JSON' in full
plan=dict(expected_pointer_SHA256='031b2a70fa20e17f08162c23dbf3a6c4147e7fc620816b50c49c95e867d534b1',expected_pending_SHA256='feeae2d85e1d826cf9be832e8240e50cff5c561c3333bd3523e3e2c6d8a7a6eb',round_name='verified184_LlavaMini_formal_capacity_Gemma4MLX8_serial_all_preserved_v74_20261122',full_note_name='full_actual_verified184_LlavaMini_completed_capacity_Gemma4MLX8_serial_v74_20261122.txt',pending_name='pending_after_daily_20261122_verified184_LlavaMini_formal_capacity_Gemma4MLX8_all_primary_v74_20261122.json',new_primary_paths=sorted(names),current_model_job_paths=['LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122','Gemma4E4BMLX8_own_original_Transformers_BF16_true3_v1_real_FLOCK_20261122','Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122'],full_handoff_template=full,local_original_records=[])
records=['build_verified184_incremental_handoff_v74_initial_v1_failed_source.py.txt','build_verified184_incremental_handoff_v74_initial_v1_failure.json','verified184_LlavaMini_completed_and_Gemma4MLX8_serial_sections_v74_20261122.txt','official_910b_before_Llava_Gemma_serial_v74_20261122.toml','local_verified184_Day22_fresh_SHA_stat_20261122.json','R_LlavaMini_completed3_full_numeric_proof_and_exact_stdout_framing_20261122.json','R_LlavaMini_formal_single_JSON_acceptance_capacity_actual_20261122.json','R_LlavaMini_completed_formal_fullNF_capacity_v4_actual_20261122.json','R_Gemma4MLX8_own_template_contract_20261122.json','R_Gemma4_native_processor_token_expansion_original_20261122.txt','R_Gemma4MLX8_own_full_frontend200_native_source_readiness_20261122.json','R_Gemma4MLX8_completed3_actual_profiler_schema_20261122.json','R_Gemma4MLX8_formal_NF2_capacity_or200_current_20261122.json','R_Gemma4MLX8_formal200_native_progress_20261122.json','R_next_native_Llava_own_metadata_no_NN_20261122.json']
for name in records:
 p=V/name;raw=p.read_bytes();plan['local_original_records'].append(dict(name=name,SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
out=V/'verified184_LlavaMini_Gemma4MLX8_incremental_handoff_plan_v74_20261122.json';out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(dict(plan=str(out),SHA256=sha(out.read_bytes()),explicit_new_primary_paths=len(names),full_template_characters=len(full),local_original_records=len(records)),ensure_ascii=False))

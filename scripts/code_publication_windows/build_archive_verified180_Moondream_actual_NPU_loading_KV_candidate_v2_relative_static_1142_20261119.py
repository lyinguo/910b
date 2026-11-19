"""Archive completed loading diagnostics and incomplete model observation work."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,tomllib
P=Path(__file__).resolve().parent;R='/root/projects/comperror';D=R+'/workspace/diagnostics/batch_910b';B=D+'/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D+'/code_publication_source_layout_20261114'
read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
prior=read('R_heartbeat1106_required180_Moondream_full_handoff_latest_20261119.json')
fresh=read('R_Moondream3_own_actual_loading_runtime_KV_candidate_inventory_1135_20261119.json')
monitor=read('R_heartbeat1106_post_Moondream_load_KV_frozen_monitor_complete_20261119.json')
queue=read('R_heartbeat1106_post_Moondream_load_KV_selector_waiter_actual_20261119.json')
parent=fresh['pointer'];assert prior['current_pointer_SHA256']==fresh['pointer_SHA256']
assert not fresh['actual_project_processes'] and not fresh['actual_original_FLOCK']
assert not monitor['actual_project_processes'] and not monitor['actual_original_kernel_FLOCK']
assert all(not monitor['latest_complete'][k] for k in ['source_errors','inspection_errors','new_log_error_candidates'])
assert sum(v['pending_bytes'] for v in monitor['latest_complete']['log_deltas'])==0
stages=fresh['stage_metadata'];cpu=B+'/Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_v2_json_sets_1116_20261119';npu=B+'/Moondream3_own_actual_NPU_loading_all662_runtime_buffers_no_forward_1120_20261119';failed=B+'/Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_1106_20261119'
for home in [cpu,npu]:
 assert stages[home]['actual_completion.json']['value']['actual_returncode']==0
 assert stages[home]['review.json']['value']['actual_original_loaded_physical_dtypes']==['torch.bfloat16']
 assert stages[home]['review.json']['value']['actual_NEW_model_forwards']==0
assert stages[failed]['actual_completion.json']['value']['actual_returncode']==1
unit_home=P/'Moondream3_complete_KV_unlearned_storage_unit_values_v2_exact_seven_1130_20261119'
unit=json.loads((unit_home/'independent_storage_test.json').read_bytes());assert unit['passed'] and unit['occurrences']==27 and unit['complete_physical_layers']==24
unit_arrays=[]
for home in [P/'Moondream3_complete_KV_unlearned_storage_unit_values_1128_20261119',unit_home]:
 for path in sorted(home.rglob('*')):
  if path.is_file():unit_arrays.append(dict(path=str(path),bytes=path.stat().st_size,SHA256=sha(path.read_bytes()),role='Complete local unlearned storage unit-test originals; retain all values, not model results'))
failure=dict(provenance='Actual write_stdin session 36997 exit result, transcribed before handoff; no replay',actual_exit_code=1,error='AssertionError: 7',location='test_moondream3_complete_physical_KV_journal_1128_20261119.py line 37, expected len(values)==8',scope='Independent reconstruction already returned before a wrong expected unique-file-count assertion; v2 changes only 8 to 7',original_source_SHA256=sha((P/'test_moondream3_complete_physical_KV_journal_1128_20261119.py').read_bytes()))
(P/'Moondream3_KV_storage_unit_v1_actual_failure_tool_transcript_1138_20261119.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf8')
raw_toml=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes();pre=tomllib.loads(raw_toml.decode());assert pre['id']=='910b' and pre['status']=='ACTIVE'
(P/'automation910b_before_Moondream1106_20261119.toml').write_bytes(raw_toml)
(P/'automation910b_before_Moondream1106_20261119_prompt.txt').write_text(pre['prompt'],encoding='utf8',newline='\n')
stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');H=D+'/heartbeat_handoffs/A100_verified180_Moondream_actual_CPU_NPU_all662_BF16_runtime_KV_observer_candidate_no_model_NN_'+stamp
install=[]
source_roles={
 'moondream3_own_complete_physical_KV_journal_v1_1128_20261119.py':('universal_eval/model/adapters/','Candidate complete physical KV value journal; all occurrences and values retained, independent storage replay, no model scoring'),
 'moondream3_own_original_KV_observation_hooks_v1_1132_20261119.py':('universal_eval/model/adapters/','Candidate consumer of journal wrapping the own original cache calls; no cache/model math changes, not yet runtime integrated or full NF'),
 'test_moondream3_complete_physical_KV_journal_1128_20261119.py':('scripts/tests/','Original failed storage unit test retained; only expected number of unique NPYs was wrong'),
 'test_moondream3_complete_physical_KV_journal_v2_exact_seven_arrays_1130_20261119.py':('scripts/tests/','Actual successful independent storage test, explicit dependency on original v1, fixes only expected 8 to actual 7'),
}
for name in ['build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_1138_20261119.py','prepare_archive_builder_v2_relative_static_paths_1142_20261119.py','build_moondream3_original_HF_load_precision_v2_json_sets_1116_20261119.py','build_moondream3_own_actual_NPU_loading_runtime_contract_1120_20261119.py','read_required180_Moondream_full_handoff_latest_1106_20261119_remote.py','read_current_Moondream3_CPU_load_precision_v2_1116_20261119_remote.py','read_current_Moondream3_NPU_load_runtime_1120_20261119_remote.py','read_Moondream3_own_actual_loading_runtime_KV_candidate_inventory_1135_20261119_remote.py',Path(__file__).name]:
 source_roles[name]=('scripts/code_publication_windows/','Actual no-forward loading preparation, complete own readback, or preservation builder; literal transport payloads are evidence only')
for name,(prefix,role) in source_roles.items():
 raw=(P/name).read_bytes();ast.parse(raw);install.append(dict(relative=prefix+name,SHA256=sha(raw),base64=base64.b64encode(raw).decode(),role=role))
additional=[dict(relative=v['relative'],SHA256=v['SHA256'],role='Actual original own HF CPU/NPU loading diagnostic, all662 source/cast checks and buffers; zero model forwards, original failed v1 preserved') for v in fresh['new_source_inventory']]
selected=set()
for tag in ['1106','1116','1120','1128','1130','1132','1135','1138']:
 selected.update(p for p in P.glob('*'+tag+'*20261119*') if p.is_file() and p.suffix in ['.py','.json','.stderr','.toml','.txt'])
selected.update(P/name for name in source_roles)
originals=[]
for path in sorted(selected):
 raw=path.read_bytes();assert len(raw)<8*1024**2
 originals.append(dict(original_path=str(path),archive_name=path.name,bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode(),role='Exact local source/readback/result and original failures/warnings; no model report package or learned arrays'))
static={}
for home,values in stages.items():
 for name,v in values.items():static[Path(home).name+'/'+name]=dict(path=v['path'],SHA256=v['SHA256'])
free=fresh['free_bytes'];counts=parent['canonical_counts']
summary=f'''【本轮11:06真实进展；正式评测仍停摆】{fresh['UTC']}实查没有项目模型NN/全模型数值NF或原runner.lock持有者，NPU1..15空，0外部VLLM不动；原盘{free}bytes约{free/1024**3:.2f}GiB。最早本段停摆仍为11月18日20:47:01.480853北京时间，本轮CPU/NPU权重加载、零KV/运行缓冲检查与无学习权重存储单测都不算恢复。新完整200/G/完整资格均0，180交付、165数值内/15超差/210完整权重待交付/310确缺、完整原协议66不变；165=66完整+99仅数值，390完整权重仍324需适配/资格/修复。最近完整200为Prithiv11月17日16:57:59.615811，.880对三D.87误差1.1494252873563%，已两书交付不重跑。

【Moondream3实际加载精度已闭合，不能再写仅meta推断】moondream/moondream3-preview自己的modelv2四片18537326392bytes/662BF16源与自己的11py、词表保持。原HF类from_pretrained确实按原非CPU auto分支请求torch.float16，但11月19日11:16:29.460966真实CPU加载、11:22:15.231661真实NPU4加载均证明全部662实际仍是BF16，逐源shape/所有raw位与自己原BF16权重相等；不是强制改BF16来修分。两次loading_info的missing/unexpected/mismatched/errors全空，text.wte与lm_head不绑，未多出别名。NPU加载前fresh完整4SF与tokenizer全文件SHA，随后逐662源SHA→actualNPU全值SHA。自己的AFS/共享Torch/原HF source/YAML没改，没有CPU学习层前向或NPU模型前向/成绩；no-forward actual0不能当模型真3/fullNF/资格。
CPU v1原脚本11:10:18因loading_info含set不能JSON序列化，实际1，完整原source/stderr/actual_completion保留。新v2只修set原内容记录，不翻旧完成码，仍使用同原class/from_pretrained。Jcpu={cpu}，actual0/662全数值加载匹配；Jnpu={npu}，actual0、662actualNPU匹配，原mask完整1x1x4096x4096/730前缀双向、原F32 RoPE4096x16x2全部逐位同，原24层48个1x32x4096x64 BF16物理KV全值零和SHA已核，use_flex_decoding=True保持。原运行mask/RoPE/完整零KV例证NPZ 508300bytes SHA8e89da320180d64f5e7e96042409bf2c5c356f10bc92f59afdfa04915da76f9a与所有JSON/警告全留。硬件padding32 warning是这次加载事实，不是视觉kernel或模型NN证明。真实加载进程均已自然退出/原EX已释放，不按state旧phase重启。

【KV观察器仅候选组件，不是完整adapter或NF】新增universal_eval/model/adapters/moondream3_own_complete_physical_KV_journal_v1_1128_20261119.py及moondream3_own_original_KV_observation_hooks_v1_1132_20261119.py。原KVCache.update/load_encoded_image调用/返回值不改，候选hook在原调用前保留真实输入，后读完整物理KV，逐raw位验证原位置写入和所有未写位置，再保存完整输入向量与可独立全值重建的journal；等值完整NPY逐dtype/shape/raw位相等才共享，每个语义发生仍保留。此为未来完整观察器候选，尚未接入真实模型/YAML，也没有本模型fullNF或真3。不能把无损存储组件测试当模型数学资格。未采历史A100 engine/tensor如实披露。
本地无权重unit全24层/27发生/7个完整NPY共22780512bytes独立重建过，越原写位置改值、重复position、int32position均拒，NaN/负零raw位保留。v1测试最后把等值共享后的实际7个文件误期望8而失败，原失败源/工具原记录/所有值留；v2只修期望8→7、actual0，主journal源完全同。两次本地所有原值/日志/manifest都保留，不传GitHub。hook仅AST及依赖审阅通过，尚未有真实模型运行集成测试，不称ready。

【仍需完整Moondream模型适配】已有06:24自己完整frontend200与06:31五组私有causal Flex NPU算子控制不重跑。自己的vision27/378/max12local+global/patch14，text24/H2048/Q32-KV32/D64/后20层64专家top8/51200全头/物理KV4096和原BF16 div255/sub.5/div.5保持。原HF.generate命中query后忽略保存max512，实际reasoning=True/每段768/temperature.5/top_p.9、前缀[1,15381,2]+question+[4]/answer[3]/EOS0保留披露，不改greedy/512或禁thinking来凑D。原PyTorch Flex在NPU的失败保留；私有原完整causal mask→SDPA候选接口仅无权重5例通过，未接模型，原use_flex_decoding=True不改。下一步补自己的视觉/MoE/tau/RoPE/全头/全部生成步观察与独立数值NF、表C唯一/B任务/三D和单YAML，validate/check_sources/config3/200及全部准入真过才NEW真3。当前没有完整adapter/YAML/NF/真3/首次200批准/fresh3/200/G，不能拿已加载权重、零缓存或单测冒充运行。

【资源与保留】Huihui已验3/27/fullNF0但完整200仍需152.74GiB；zai-GLM已验3/586需219.53GiB，Ovis原265.78GiB、Phi1244.15GiB。GLM原每20秒等候器PID1530341仅当前快照，实际/proc/startticks/cwd与source绑定、不持EX、不充NN；只有235720910370bytes原预算与全源/保护/NN-NF-锁门槛真过才自动新批准→NEWfresh3/fixed200，勿重复它或已过3。11:25原selector0/null，三根暂不可读保留历史不加310，不清旧protectedstate。并行检查中selector见一个other_project_python是同时执行的冻结monitor读取进程，不是NN/NF/失锁，完整argv/上下文保留，新未知检查错误不套用。原扩400GiB问题仍未获答复，不自行扩容/换盘/减guard/删未验；本轮无任何删除。
完整五件/所有逐样本input-output-raw-ref-error-token/指标统计配置/源码YAML快照日志与全部小样本/失败值不减。MedraN原FP16视觉第二层非有限和旧NF路径失败未修，不能只修路径就验收或改BF16/epsilon/滤样本；Perceptron自己的tensorstream缺、Florence原任务数据/autoF32绑定等旧pending继续，不能装包/stub/借换。

【代码】11月19日已唯一push e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d/Code-Day19，今天不第二commit。新增加载诊断/原失败/候选KV工具、真实消费hook、成功及失败测试和有用Windows控制/交接源码只放R源/scripts/configs并列完整pending，Day20逐用途/依赖/消费者/freshSHA/全diff/alloutgoing/对应检查/明确planSHA再一次正常push，不漏有用源和后续提交文件。工作区只证据，字面base64部署payload仅归档。两书/state/reg/PAUSE/POLICY/冻结runner/Gwriter/retention policy/ledger均保持。下面原全部强制规范和父祖先家族失败/例外继续，不无限拼接。

'''
first=pre['prompt'].split('\n\n',1)[0]
body=first+'\n\n'+f"当前完整交接，直接父{parent['handoff_directory']}，bodySHA256={parent['handoff_SHA256']}、facts={parent['facts_SHA256']}、closing={parent['closing_addendum_SHA256']}；全部祖先/65471字符archive继承全部原强制规范/家族失败/精确清理例外/用户决定。正文/facts是初快照，更晚closing/late_actual_notes/真实proc/内核FLOCK/退出优先，勿按旧live重启完成款。完整交接加短引用，不无限拼接、不存密码。\n\n"+summary+pre['prompt'][pre['prompt'].index('R=/root/projects/comperror'):]
snapshot=dict(UTC=fresh['UTC'],actual_project_processes=[],actual_original_FLOCK=[],free_bytes=free,actual_model_NN_or_fullNF=False,current_outage_first_proven_BJT='2026-11-18 20:47:01.480853',last_complete200_BJT='2026-11-17 16:57:59.615811',latest_model_small_NF_BJT='2026-11-18 20:47:00.909217',actual_NEW_model_forwards_this_cycle=0)
facts=dict(canonical_counts=counts,formal_NEW200_this_cycle=0,new_G_this_cycle=0,new_full_qualification_this_cycle=0,current_actual_snapshot=snapshot,
 new_owned_candidate='moondream/moondream3-preview',own_actual_CPU_loading_review=stages[cpu]['review.json'],own_actual_NPU_loading_review=stages[npu]['review.json'],own_actual_all662_NPU_load_casts=stages[npu]['all662_own_source_actual_NPU_loaded_casts.json'],own_full_original_runtime_buffers_and48_KV=stages[npu]['original_runtime_buffers_and_all48_physical_KV.json'],original_failed_load_record=stages[failed]['actual_completion.json'],local_KV_storage_unit_passed_review=unit,local_storage_original_failure=failure,all_local_unit_originals_retained=unit_arrays,all_new_runtime_control_files_retained=fresh['all_new_numeric_control_files'],effective_loaded_runtime_dtype_proven_BF16_on_CPU_and_NPU_no_forward=True,
 full_model_observer_independent_NF_YAML_and_genuine3_not_yet_present=True,candidate_KV_hook_not_yet_runtime_integrated=True,original_archive_v1_path_guard_refusal_preserved=dict(original_stdout_file='R_archive_verified180_Moondream_loading_KV_actual_1138_20261119.json',original_stderr_file='R_archive_verified180_Moondream_loading_KV_actual_1138_20261119.json.stderr',scope='Absolute archive label rejected inside() before facts/pointer publication; only copy destination label changed to relative, original source and canonical unchanged'),no_deletion_this_cycle=True,daily_code_already_pushed_commit='e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d',original_queue_current=queue,pre_update_automation=dict(TOML_SHA256=sha(raw_toml),prompt_SHA256=sha(pre['prompt'].encode()),preserved_fields={k:v for k,v in pre.items() if k not in ['prompt','updated_at']}))
closing=dict(initial_snapshot_only=False,current_actual_stage='no_project_model_NN_or_fullNF_Moondream_all662_actual_NPU_BF16_loading_and_candidate_KV_storage_completed',actual_snapshot=snapshot,original_loading_JSON_set_failure_preserved=True,original_storage_unit_wrong_expected_unique_file_count_failure_preserved=True,candidate_hooks_not_model_acceptance=True,no_deletion=True)
plan=dict(project_root=R,handoff_directory=H,expected_pointer_SHA256=fresh['pointer_SHA256'],canonical_SHA256=fresh['canonical_SHA256'],parent_handoff_directory=parent['handoff_directory'],parent_files_SHA256={'handoff.txt':parent['handoff_SHA256'],'facts.json':parent['facts_SHA256'],'closing_addendum.json':parent['closing_addendum_SHA256']},parent_pending=parent['new_primary_pending'],parent_pending_SHA256=parent['new_primary_pending_SHA256'],install_primary_sources=install,additional_existing_primary_sources=additional,local_original_artifacts=originals,static_original_files=static,handoff_text=body,new_pending=C+'/pending_after_daily_20261119_verified180_Moondream_actual_NPU_loading_KV_candidate_all_primary_v34.json',Code_Day_already_pushed='2026-11-19',next_daily_review='2026-11-20 first check: all new actual loading/source diagnostics and candidate KV journal, hook, tests and useful Windows dependencies; full SHA/diff/outgoing/consumer/tests/plan review, one normal push',facts=facts,closing=closing,pointer_updates=dict(canonical_counts=counts,verified_delivered_count=180,current_actual_model_NN_or_NF_active=False,current_actual_model_process_or_NF_active=False,current_actual_stage=closing['current_actual_stage'],current_owned_candidate='moondream/moondream3-preview',source_of_truth='Actual proc/kernel locks/completion; original loading, buffers, storage tests and candidates are not model NN/fullNF',severity_first_line='严重阻塞｜正式评测未运行'))
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(P/'archive_verified180_Moondream_loading_KV_round_plan_1138_20261119.json').write_bytes(raw);(P/'handoff_verified180_Moondream_loading_KV_current_1138_20261119.txt').write_text(body,encoding='utf8',newline='\n')
payload=dict(SHA256=sha(raw),base64=base64.b64encode(raw).decode())
remote='DATA='+repr(payload)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,json
R=Path('/root/projects/comperror');raw=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['SHA256'];plan=json.loads(raw);H=Path(plan['handoff_directory']);H.mkdir(exist_ok=False);p=H/'archive_plan_original.json';p.write_bytes(raw);sys.path[:0]=[str(R),str(R/'scripts')]
from maintenance.archive_alignment_primary_handoff_snapshot_20261117 import execute
print(json.dumps(execute(p,DATA['SHA256']),ensure_ascii=False))
'''
ast.parse(remote);(P/'execute_archive_verified180_Moondream_loading_KV_1138_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(handoff_directory=H,plan_SHA256=sha(raw),new_primary_sources=len(install)+len(additional),local_originals=len(originals),static_originals=len(static),body_chars=len(body)),ensure_ascii=False))

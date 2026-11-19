"""Prepare an explicit archival plan; no model, report, workbook, or runner mutation."""
from pathlib import Path
from datetime import datetime, timezone
import ast, base64, hashlib, json, tomllib

P = Path(__file__).resolve().parent
R = '/root/projects/comperror'
D = R + '/workspace/diagnostics/batch_910b'
C = D + '/code_publication_source_layout_20261114'
B = D + '/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda n: json.loads((P / n).read_bytes())
prior = read('R_heartbeat0114_required180_Huihui_full_handoff_and_latest_20261119.json')
parent = prior['current_pointer']
monitor = read('R_heartbeat0114_post_daily19_frozen_monitor_complete_current_20261119.json')
owned = read('R_Huihui_current_completed3_fullNF_capacity_after_daily19_20261119.json')
collection = read('R_daily19_current_declared_primary_sources_v3_20261119.json')
receipt = read('daily_20261119_receipt.json')
policy_receipt = read('R_daily19_actual_reviewed_commit_policy_CAS_receipt_20261119.json')
assert receipt['passed'] and policy_receipt['passed']
latest = monitor['latest_complete']
assert not latest['source_errors'] and not latest['inspection_errors'] and not latest['new_log_error_candidates']
assert not monitor['actual_project_processes'] and not monitor['actual_original_kernel_FLOCK']
assert owned['actual_states']['J/completion.json']['value']['actual_returncode'] == 0
assert owned['actual_states']['H/actual_completion.json']['value']['actual_returncode'] == 1
raw_toml = (P/'automation910b_before_daily19_preserved.toml').read_bytes()
automation = tomllib.loads(raw_toml.decode('utf8'))
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
H = D + '/heartbeat_handoffs/A100_verified180_Day19_pushed_Huihui_GLM_Phi_capacity_blocked_' + stamp

additional = []
for row in collection['records']:
    digest = policy_receipt['publication_policy_after_SHA256'] if row['path'] == 'configs/publication/code_only.json' else row['SHA256']
    additional.append(dict(relative=row['path'], SHA256=digest, role=row['entry']['role'] + ': ' + row['entry']['reason']))
install = []
for name, relative in [
    ('archive_alignment_reviewed_daily_transition_primary_handoff_20261119.py', 'scripts/maintenance/archive_alignment_reviewed_daily_transition_primary_handoff_20261119.py'),
    ('build_daily19_verified_push_receipt_and_R_policy_CAS_20261119.py', 'scripts/code_publication_windows/build_daily19_verified_push_receipt_and_R_policy_CAS_20261119.py'),
    (Path(__file__).name, 'scripts/code_publication_windows/' + Path(__file__).name),
]:
    raw = (P/name).read_bytes(); ast.parse(raw)
    install.append(dict(relative=relative, SHA256=sha(raw), base64=base64.b64encode(raw).decode(), role='Actual explicit Day19 publication receipt or source/CAS handoff archive; no model or canonical mutation'))

selected = set(P.glob('daily_20261119_*'))
selected.update(P.glob('R_daily19_*20261119.json'))
selected.update(P.glob('*daily19*20261119.py'))
selected.update(P/n for n in [
    'R_heartbeat0114_required180_Huihui_full_handoff_and_latest_20261119.json',
    'R_heartbeat0114_frozen_monitor_complete_current_20261119.json',
    'R_heartbeat0114_actual_resource_waiter_frozen_selector_20261119.json',
    'R_heartbeat0114_post_daily19_frozen_monitor_complete_current_20261119.json',
    'R_Huihui_current_completed3_fullNF_capacity_after_daily19_20261119.json',
    'R_own_Florence_pending_deployment_heads_and_sources_0114_20261119.json',
    'R_Florence_own_table_task_binding_and_frontend_resources_0114_20261119.json',
    'R_Florence_fresh_own_generic_saved_contract_metadata_0114_20261119.json',
    'automation910b_before_daily19_preserved.toml', 'automation910b_before_daily19_preserved_prompt.txt',
    'collect_code_publication_primary_sources.py', 'archive_alignment_reviewed_daily_transition_primary_handoff_20261119.py',
])
artifacts = []
for path in sorted(selected):
    if not path.is_file(): continue
    raw = path.read_bytes()
    row = dict(original_path=str(path), archive_name=path.name, bytes=len(raw), SHA256=sha(raw), role='Exact local review/source/actual result or pre-update official bytes; original failures retained')
    if len(raw) < 8*1024**2:
        row['base64'] = base64.b64encode(raw).decode()
    else:
        row['retained_original_not_deleted'] = True
        row['large_source_collection_contains_duplicate_explicit_source_payloads_fully_archived_as_primary_bytes'] = True
    artifacts.append(row)
counts = parent['canonical_counts']
free = monitor['actual_free_bytes']
summary = dict(actual_snapshot_UTC=monitor['UTC'], actual_project_NN_or_NF=[], actual_original_FLOCK=[],
    free_bytes=free, current_outage_first_proven_BJT='2026-11-18 20:47:01.480853',
    most_recent_complete200_BJT='2026-11-17 16:57:59.615811', most_recent_small_fullNF_BJT='2026-11-18 20:47:00.909217',
    Huihui_complete200_required_bytes=164003454340, GLM_complete200_required_bytes=235720910370,
    Ovis_original_complete200_required_bytes=285374199124, Phi_complete200_required_bytes=1335898792811,
    user_disk_expansion_question_pending_no_approval=True)
prompt = automation['prompt']
first = prompt.split('\n\n', 1)[0]
body = first + '\n\n' + f'''910B当前完整交接，直接父{parent['handoff_directory']}，正文SHA256={parent['handoff_SHA256']}、facts={parent['facts_SHA256']}、更晚closing={parent['closing_addendum_SHA256']}。父及全部祖先/65471字符archive全部规范、家族、失败、精确清理例外与用户决定完整继承；短引用不取消约束，不无限拼接、不保存密码。本文/facts为当前快照，closing/late_actual_notes/真实proc与内核FLOCK和实际退出优先。旧Huihui live、Day18代码未更新等快照已过时，不重启完成阶段。

【Day19当前实况与代码已交付】本轮新增完整NPU200/G/完整资格均0，四项165数值内/15超差/210完整权重待交付/310确缺，已交付180与完整原协议66不变。当前{monitor['UTC']}实查没有项目NN/NF/原FLOCK，原项目盘{free}bytes约{free/1024**3:.2f}GiB，NPU1..15空、0外部服务不动。Huihui已验3/27/fullNF0，完整200需164003454340bytes约152.74GiB；zai-GLM已验3/586、200需219.53GiB；Ovis已验3/1536原格式需265.78GiB；Phi已验3/5、200需1244.15GiB。所有通过的3与NF不重放；当前停摆从11月18日20:47:01.480853北京时间起，按新钟计时，最近完整200仍Prithiv11月17日16:57:59。资源等候器真实PID仅快照，自己不持EX/不充NN，每20秒原盘原预算真过才自动NEWfresh3/fixed200，不重复。原扩400GiB问题尚未答复，不擅扩容/换盘/减guard/删未验；全报告、全部小样本及失败数值、源码配置日志均保留，本轮没有删除。
北京时间11月19日已唯一正常push {receipt['commit']}，Code-Day2026-11-19，2021明确路径/109变更（108新增源或配置，只更新已有发布清单，无已有模型源码变更），远端同commit、本地干净。唯一GitHub/独立分支不变，不merge/force/导Gitee历史、不改R master/remote。R23项检查、本地19项及11MJS语法检查通过；首次collector run.py路径误拒/encoding拼写失败、首次本地临时目录5权限错误、成功同测试重试及旧源SyntaxWarning全部原件留。完整plan_SHA ea10ee445f0128237dc11289fc3edde8527e58b53212c68940b2f6982ea18679、全diff/全部outgoing完整树审核/远端核验过。receipt={policy_receipt['receipt']} SHA{policy_receipt['receipt_SHA256']}。只对R真实发布消费者configs/publication/code_only.json做a78329517e49e93433f428c3d2e68908c13a77a5d39eba080d4bb82d40df0151→b6974eebfc0b50bd7d817eafba8f23cc516960c1db854dd1f6d7421d927c057b精确CAS，publisher e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30、原模型与活GLM源/表/state/reg/原runner全部不改。旧pending的旧发布清单SHA以本次精确已审核transition继承，非白名单放宽。
Day19今日不第二提交，之后本轮有用维护源只放R scripts并完整pending，Day20首轮一起审。全部有用实现/真实配置消费者与依赖不得漏审；仅代码配置必要文档，禁止权重数据数组报告日志keys/.env/凭据/归档/旧历史，禁gitadd.或扫workspace。主实现不放workspace，旧完整报告/源码配置日志不能为清盘删除。

【本轮继续其他原权重项的实际核查】三款Florence原权重/自己的源与head可读，未进行新NN或CPU learned。microsoft/Florence-2-large-ft C唯一row416/D.1822、F16/florence2-large-ft-gufeng_v3 row248/D.154、AskUI/PTA-1 row172/D.152都G空。表B的image-text-to-text与保存部署caption image-to-text/ROUGE-L绑定须逐款闭合，不能换成MC或别的数据。large-ft原FP16/max2048/自己的51289x1024头，指定原数据R/workspace/diagnostics/a100_high_nine_20261015/reference_dataset/eval.jsonl当前缺失；不拿AA/别数据集顶替。F16/PTA原auto映射F32、原max1024/256、beam3/原caption prompts和数据保持；原florence2_adapter.py已fresh SHA ee66715347653e921f53cb2e07baba39e5238666f61969ee2a280a4893db15ca。它们的原image-to-text-short数据文件SHA ceb1670f2848955290915dbe375dca8b34b62d6980e9f6cd1abe38bf0e68963b可读，但不能擅改FP16或任务数据来交付G。此为原部署/源/数据元信息核查，不是ready、AA或NPU成绩。继续其他自身原前置闭合工作，尤其MedraN原FP16视觉图/独验缺口，未闭合前不套BF16原图。其余原pending与失败全继承。

''' + prompt[prompt.index('R=/root/projects/comperror'):]
old_start = body.index('11月18日已唯一push')
old_end = body.index('\n\n严格Windows SSH', old_start)
body = body[:old_start] + '''11月19日已唯一push e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d/Code-Day19，今日不第二commit；原Day18 02df188...与所有失败回执完整继承。当前完整源pending/facts与更晚closing为准，新增维护源Day20一并审核，不漏有用实现。唯一https://github.com/lyinguo/910b.git、codex/910b-source-layout-20261114，不merge/force/导Gitee历史/改R master remote；按用途/真实消费者/依赖/fresh并发SHA/全diff/alloutgoing/对应检查/实际publisher消费/审核planSHA精确暂存正常push核remote，有变每天一次、无变不空提交、失败续原已审commit，不等提交才跑。只代码配置必要文档，不权重数据数组报告日志凭据归档，禁gitadd.或扫workspace；R源逐SHA导独立分支，不反盖R模型源，本地README/ignore/策略留已审。字面base64运行payload仅证据，不当工作代码。''' + body[old_end:]
body = body.replace('当前盘约93.3GiB', f'父20:53时盘约93.3GiB；本轮实际盘约{free/1024**3:.2f}GiB')
facts = dict(canonical_counts=counts, formal_NEW200_this_cycle=0, new_G_this_cycle=0, new_full_qualification_this_cycle=0,
    source_of_truth='Latest real proc/kernel FLOCK and actual exit; completed stages never restarted', current_actual_snapshot=summary,
    own_completed3_fullNF=owned['actual_states']['J/completion.json']['value']['accepted'][0],
    own_completed3_capacity_actual_returncode=1, new_frontend_or_model_NN_not_claimed=True,
    daily_code=receipt, no_matrix_or_log_or_source_deletion_this_cycle=True,
    pre_update_automation=dict(SHA256=sha(raw_toml), prompt_SHA256=sha(prompt.encode()), preserved_fields={k:v for k,v in automation.items() if k not in ['prompt','updated_at']}))
static = {k:dict(path=v['path'], SHA256=v['SHA256']) for k,v in prior['full_facts']['static_original_evidence'].items()}
for key, value in owned['static_original_metadata'].items():
    static['Huihui_current_completed/' + key] = dict(path=value['path'], SHA256=value['SHA256'])
static['daily19_reviewed_push_receipt'] = dict(path=policy_receipt['receipt'], SHA256=policy_receipt['receipt_SHA256'])
closing = dict(initial_snapshot_only=False, current_actual_model_NN_or_NF_active=False, actual_snapshot=summary,
    current_actual_stage='capacity_blocked_after_completed_own_true3_fullNF', most_recent_complete200_BJT=summary['most_recent_complete200_BJT'],
    current_idle_first_proven_BJT=summary['current_outage_first_proven_BJT'], current_actual_model_processes=[], actual_original_FLOCK=[],
    last_small_fullNF_BJT=summary['most_recent_small_fullNF_BJT'], day19_pushed_commit=receipt['commit'], no_evidence_deletion=True)
pointer_updates = dict(canonical_counts=counts, verified_delivered_count=180, current_canonical_counts=counts,
    current_actual_model_NN_or_NF_active=False, current_actual_model_process_or_NF_active=False,
    current_formal_model_stage='capacity_blocked_after_completed_own_true3_fullNF', current_actual_stage='capacity_blocked_after_completed_own_true3_fullNF',
    last_complete_result_BJT=summary['most_recent_complete200_BJT'], first_proven_stoppage_BJT=summary['current_outage_first_proven_BJT'],
    severity_first_line='严重阻塞｜正式评测未运行', source_of_truth=facts['source_of_truth'], current_daily_code_commit=receipt['commit'])
plan = dict(project_root=R, handoff_directory=H, expected_pointer_SHA256=prior['current_pointer_SHA256'],
    canonical_SHA256=monitor['canonical_SHA256'], parent_handoff_directory=parent['handoff_directory'],
    parent_files_SHA256={'handoff.txt':parent['handoff_SHA256'], 'facts.json':parent['facts_SHA256'], 'closing_addendum.json':parent['closing_addendum_SHA256']},
    parent_pending=parent['new_primary_pending'], parent_pending_SHA256=parent['new_primary_pending_SHA256'],
    reviewed_daily_transition_receipt=policy_receipt['receipt'], reviewed_daily_transition_receipt_SHA256=policy_receipt['receipt_SHA256'],
    install_primary_sources=install, additional_existing_primary_sources=additional, local_original_artifacts=artifacts,
    static_original_files=static, handoff_text=body, new_pending=C+'/pending_after_daily_20261119_verified180_Huihui_GLM_Phi_capacity_all_primary_v30.json',
    Code_Day_already_pushed='2026-11-19', next_daily_review='2026-11-20 first Beijing-day check: review all new useful sources/real consumers and dependencies, exact SHA/diff/outgoing/test/plan before one normal push',
    facts=facts, closing=closing, pointer_updates=pointer_updates)
raw = (json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode()
(P/'archive_daily19_verified180_complete_current_plan_20261119.json').write_bytes(raw)
(P/'handoff_daily19_verified180_complete_current_20261119.txt').write_text(body,encoding='utf8',newline='\n')
data = dict(SHA256=sha(raw), base64=base64.b64encode(raw).decode(), utility=install[0])
remote = 'DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,json,ast,importlib
R=Path('/root/projects/comperror');item=DATA['utility'];code=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(code).hexdigest()==item['SHA256'];ast.parse(code)
target=R/item['relative'];assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(code)
raw=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['SHA256'];plan=json.loads(raw);H=Path(plan['handoff_directory']);H.mkdir(exist_ok=False);p=H/'archive_plan_original.json';p.write_bytes(raw)
sys.path[:0]=[str(R),str(R/'scripts')];m=importlib.import_module('maintenance.archive_alignment_reviewed_daily_transition_primary_handoff_20261119')
print(json.dumps(m.execute(p,DATA['SHA256']),ensure_ascii=False))
'''
ast.parse(remote)
(P/'execute_archive_daily19_verified180_complete_current_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(handoff_directory=H, plan_SHA256=sha(raw), local_originals=len(artifacts), primary_sources=len(additional)+len(install), static_files=len(static), handoff_characters=len(body)),ensure_ascii=False))

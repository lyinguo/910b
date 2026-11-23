"""Bind the exact approved publisher-policy transition and preserve the finite current handoff."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
old=(V/'preserve_heartbeat1226_verified185_resource_blocked_legacy_cleanup_full_handoff_v89_20261122.py').read_text(encoding='utf8')
repl={
'0bd677fc2ecbfd974d773895248e97f8bf1bff50331d10e1939e81423f27d2f7':'97a980f6eda2d1f6dbbbf1101f9c096e99e38e35359d4227d266559c24e27c9b',
'ba12c8d046dff5eb0d8e449740eb92f30e338a781fe75136a8786badfee65333':'aa3069e5ac5fe3c218b63debbb4a7f5d25277e86426fdeba19498a3cbff56007',
'2b3ac4b53b59fc990cd00152cd68c548e9861201f5bff02a321acf1bb20ae5a9':'bc5f4275df7abc884b3f021f096cb0a5934ce061fb6d7843f310dab587059ef8',
'verified185_heartbeat1226_resource_blocked_legacy_cleanup_full_actual_v89_20261122':'verified185_Day23_actual_commit_resource_blocked_full_actual_v90_20261123',
'pending_after_daily_20261122_verified185_resource_blocked_legacy_cleanup_all_primary_v89_20261122.json':'pending_after_daily_20261123_verified185_Day23_commit_resource_blocked_all_primary_v90_20261123.json',
'full_actual_verified185_resource_blocked_legacy_cleanup_v89_20261122.txt':'full_actual_verified185_Day23_commit_resource_blocked_v90_20261123.txt',
"Code_Day_already_pushed='2026-11-22'":"Code_Day_already_pushed='2026-11-23'",
"Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e'":"Code_Day_commit='5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9'",
'no_second_Day22_commit':'no_second_Day23_commit',
"next_daily_review='Day23":"next_daily_review='Day24"}
for a,b in repl.items():assert a in old,a;old=old.replace(a,b)
needle="    assert len(rows)==2948 and all(sha(n)==r['SHA256'] for n,r in rows.items())"
replacement="""    assert len(rows)==2964
    policy=R/'configs/publication/code_only.json';policy_row=dict(rows[str(policy)])
    syncpath=C/'daily_20261123_actual_reviewed_commit_remote_receipts/R_policy_exact_CAS_actual.json'
    sync=js(syncpath);assert sync['passed'] and sync['commit']==sync['remote_commit']=='5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9'
    assert sync['reviewed_plan_SHA256']=='66dad86f3b449fc5f3b66479c8157a837678c6657dc80af101572e92d818b992'
    assert sync['policy_before_SHA256']==policy_row['SHA256']=='39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5'
    assert sync['policy_after_SHA256']==sha(policy)=='1585e1dce7edea8a7b3c99bf27ae4b2840b715ac3b054542d55141bbc44906e9'
    assert all(sha(n)==r['SHA256'] for n,r in rows.items() if n!=str(policy))
"""
assert needle in old;old=old.replace(needle,replacement)
needle="    for n,h in q['new_primary_expected_SHA256'].items():"
extra="""    pdest=note/'primary_source_snapshots/configs/publication/code_only.json';pdest.parent.mkdir(parents=True,exist_ok=True);pdest.write_bytes(policy.read_bytes())
    rows[str(policy)]=dict(policy_row,SHA256=sha(policy),bytes=policy.stat().st_size,snapshot=str(pdest),previous_exact_source_row=policy_row,authorized_policy_only_transition_receipt=str(syncpath),authorized_policy_only_transition_receipt_SHA256=sha(syncpath))
    for n,h in q['new_primary_expected_SHA256'].items():"""
assert needle in old;old=old.replace(needle,extra)
needle="    compact['no_official_automation_write_this_round']=True"
extra="""    compact['no_official_automation_write_before_this_snapshot']=True
    compact['Day23_remote_verified_code_receipt']=q['Day23_remote_verified_code_receipt']
    compact['approved_R_publication_policy_only_transition']=sync
    compact['R_publication_policy_transition_receipt_SHA256']=sha(syncpath)
    compact['Day23_directory_allocation_metadata']=q['Day23_directory_allocation_metadata']
    compact['Day23_R_test_entry_v1_failed_before_tests_v2_correct_original_source_arg_passed39']=True
    compact['today_one_code_commit_not_new_model_NPU_acceptance']=True"""
assert needle in old;old=old.replace(needle,extra)
mainname='preserve_Day23_verified185_actual_commit_blocked_resource_full_handoff_v90_20261123.py';ast.parse(old);(V/mainname).write_text(old,encoding='utf8',newline='\n')
prior=json.loads((V/'heartbeat1226_verified185_blocked_resource_legacy_cleanup_full_handoff_plan_v89_20261122.json').read_bytes())
intro='''完整当前交接v90：北京时间2026-11-23 Day23代码已唯一正常push并核远端5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9；模型结果仍canonical185/170数值内15超差205完整权重待交付310确缺/71完整原资格，本轮新完整NPU200/G/资格/模型前向/fullNF均0。代码交付不充模型进展，资源仍阻塞，最近全数值小样本Turkish2026-11-22 17:40:38的3/6已完成，17:40:49因200约3.85TiB容量拒后停摆，以最新真实钟算时长，不重启已过3/6或任何旧job。最近合格200仍Gemma4MLX8实际ownBF16 .840对三D.8350/.8350/.835均.59880239520958%，11月22日03:34:24完整NF0/03:47:22双书全审，185两书当前SHA2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878不变。
必须先完整读根H/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1、facts1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7，本全文及同目录facts SHACURRENT_FACTS_SHA256，再closing/late/pointer更晚实际。父v89全文ada90564c5be9abb7c891fdaeccff91db5df9ae43d886f974c5fdec67e02a2cf/factsfefeb3388593b48f04faa24872b9f4018b2a335bcb431fa3cedb1929eb313414/pendingbc5f4275df7abc884b3f021f096cb0a5934ce061fb6d7843f310dab587059ef8、v87全文327a420602dc4a91090489b44827dbf4323e707dacc4923e6a16242233dc9455及全部祖先/65471字符archive所有强制规范、家族失败、精确清理例外和用户决定完整继承，短引用不取消约束。有限完整当前交接加短引用，不无限拼接、不存密码；旧live/旧Day23尚未提交均过时，后文旧时点以本段和最后actual覆盖。
【Day23已经真实唯一提交，今日不第二commit】唯一https://github.com/lyinguo/910b.git、独立codex/910b-source-layout-20261114、D:/code-lv/910b；正常push5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9/Code-Day2026-11-23，remote同SHA/worktreeclean。2972明确代码配置路径/303变更（302新增，仅已有configs/publication/code_only.json更新），全source内容/AST/用途/依赖/真实consumer/full diff6,448,620bytes SHA cf7c323c5ef23d2ac977bbf59dc80d855404b8ca852c36eeb1b133d35d38cd9a、all outgoing完整2972文件树通过，审核plan_SHA66dad86f3b449fc5f3b66479c8157a837678c6657dc80af101572e92d818b992。无merge/force/Gitee历史/R master-remote改动，无权重数据数组报告日志keys.env凭据/归档/opaque运输payload进Git，无gitadd.或workspace扫入Git。两份Meta inline input_records200协议继续准确仅Git排除，R全原字节和源码生成器/YAML都保留，不能把它们误当未保存源。
R39项、本地19项、25MJS node语法检查全0。首次R测试入口在加载原parallel NPZ测试时未传SOURCE/OUT导致FileNotFoundError，尚未进入单元执行，原源/完整stderr/失败stdout保持；v2只恢复原测试的两个显式参数和独立诊断OUT，不改模型数学或测试断言，全部39项通过。并行标准NPZ原有7项覆盖ZIP64/完整raw位/顺序/已有文件保护/不安全成员拒绝，和complete-module6项及R现有测试一起执行，不是模型NN或旧CPU200。旧库SyntaxWarning/FutureWarning完整上下文留，不说从无错误，也不泛化白名单。完整日志/plan/差异/回执保存在C/daily_20261123_actual_reviewed_commit_remote_receipts，daily_20261123_receipt.json SHA51d86b8876b24f3224d4d5c4b9d40ef841f2fd4a7acb593e0f36bd988400c554。
R唯一更改的已有源为实际publisher配置configs/publication/code_only.json：39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5→1585e1dce7edea8a7b3c99bf27ae4b2840b715ac3b054542d55141bbc44906e9，已经按真实commit/plan/全部源并发SHA、完整canonical185、原Bee v5 source/canonical不消费此配置证明后精确CAS。publisher本体e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30不改，模型/run.py/pipeline/冻结runner/Gwriter/selector/PAUSE/POLICY/state/reg/两书/ledger均不改。新pending对此唯一配置SHA合法transition准确绑定旧row/旧snapshot/真实R_CAS receipt，不能因这个例外放松任何模型或其他源SHA。
【资源实况与立即接续】本轮先完整读指定root/v87及更晚v89，再SHA冻结monitor/完整latest-checkpoint、真实proc/argv/cwd/PPid/startticks/FD/原EX/cards/实际退出/currenttrace；无NN/NF/原EX，只有原Bee v5无NN容量等候1696145及原cleanupwatcher/旧无子无锁1668102，不信号、不热改或复制它们。原盘约20.9GiB，Bee最小已准备200预算192555984252bytes约179.33GiB，自己20秒循环和全source/quiet门槛真过才NEWfresh3/fixed200，不等300分钟；扩原盘至少200GiB可用空间问题仍待用户答复，未答不是批准或已扩容，禁擅换盘/扩/降guard/流式剪证据/删未验。源暂不可读的Nemotron/Cognitive/Calm本轮冻结只读gate再次确认三root不可读，不永久加310，不清selector/protectedstate或改历史G；资源恢复仍全原冻结SHA和新私有阶段，不按旧live重启。
本轮额外只读查看项目整体和证据文件类型，非清理：原盘总929396658176bytes，R项目约837744803840bytes，其中workspace/diagnostics约823280136192bytes；model_runs约750579728384bytes，batch_910b约72716570624bytes。model_runs8513个唯一inode NPZ约623323394048allocated bytes，此外JSON约91252662272、DB约10355388416、JSONL约6679097344 bytes，多为完整数值元信息和原始profile/trace，不是可凭后缀删除的缓存；最大如旧Docci npu_trace5.59GB、msprof3.37GB、Xtuner execution每1.027GB，日志/失败/小样本/必要数值例证及ALLexternalSHA保持。无单独NPY，最大文件多数已LZMA/DEFLATE；没有采用新稀疏/重压/变格式，也没有找到可按已验正确非例证规则新增删除的项目，本轮0删除/0新释放空间。代码提交、测试、du、压缩设想和等候器不充正式NN恢复。
【继续维护】本轮全部有用R源码/实际Day23 collector、测试v1/v2、审核/并发CAS/publisher/远端验证/真实配置同步控制与原本地回执，及更新后更晚维护源码都逐原字节/SHA入CURRENT_SOURCE_COUNT完整pending；今日不第二commit，Day24首轮全审更晚新源/实际consumer/deps/fresh并发SHA/full diff-alloutgoing/相应检查/实际publisher消费及审核planSHA后只一次正常push核remote，无变不空提交，push失败续同已审commit，不等提交才运行。完整报告、每条input-output-raw-ref-error-token/统计配置运行信息、source/config快照/log及必要完整原数值永保，所有历史精确pruned例外现在不能再全数组验，旧proof字节留且如实披露。原910b ACTIVE/名称/原聊天/300分钟保持，本轮截至本snapshot只读TOML，后续如更新只能official并核实际readback归档，不能直接写TOML。
当前仍严重阻塞，最早当前停摆时点2026-11-22 17:40:49.156504北京时间；最近全数值小样本Turkish17:40:38，最近合格200 Gemma11月22日03:34:24。每轮首行“严重阻塞｜正式评测未运行”，关键检查连接失败“严重告警｜无法确认正式评测状态”；报告真实阶段/样本前向卡/已证原因/最近完整及合格结果北京时间/最早可证持续时长/解除动作，恢复当轮也报；NOTIFY文件不当聊天通知。全部资源款真原协议910B200/三D5%、确缺跳过、无pending/资格复审及双书完整报告历史例外全量复查后才关闭，远未完成。

'''
template=intro+prior['full_handoff_template'][prior['full_handoff_template'].index('【必须先读的完整继承】'):]
assert template.count('CURRENT_ACTUAL_JSON')==1
primary=json.loads((V/'daily23_actual_consumer_explicit_primary_paths_20261123.json').read_bytes())
stage={'scripts/maintenance/'+mainname:old.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
for n,b in stage.items():primary[n]=sha(b)
names=['R_heartbeat1731_verified185_required_v87_latest_v89_current_actual_20261123.json','heartbeat1731_verified185_fresh_local_book_original_automation_actual_20261123.json','daily_20261123_receipt.json','R_daily23_reviewed_policy_and_remote_receipts_actual_CAS_20261123.json','R_daily23_actual_R_source_unit_checks_20261123.json','R_daily23_actual_R_source_unit_checks_20261123.json.stderr','R_daily23_actual_R_source_unit_checks_v2_original_source_arg_20261123.json','R_daily23_actual_R_source_unit_checks_v2_original_source_arg_20261123.json.stderr','R_Day23_publication_during_actual_serial_resource_recheck_20261123.json','R_Day23_original_three_resources_current_readonly_20261123.json','R_Day23_project_directory_allocation_depth2_actual_20261123.json','R_Day23_diagnostics_allocation_depth1_actual_20261123.json','R_Day23_model_runs_nonarray_allocation_metadata_actual_20261123.json','probe_R_project_directory_allocation_depth2_readonly_20261123_remote.py','probe_R_diagnostics_allocation_depth1_readonly_20261123_remote.py','probe_model_runs_nonarray_allocation_metadata_only_20261123_remote.py']
local=[]
for n in names:
    b=(V/n).read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
q={k:prior[k] for k in ['cleanup_discovery','cleanup_discovery_SHA256','official_actual_readback','original_three_owned_exact_locations','legacy_schema_reaudits','auxiliary_reader_failure','numeric_allocation']}
q.update(new_primary_expected_SHA256=primary,local_original_records=local,full_handoff_template=template,
         original_three_resources_fresh_readonly=json.loads((V/'R_Day23_original_three_resources_current_readonly_20261123.json').read_bytes()),
         current_local_and_automation=json.loads((V/'heartbeat1731_verified185_fresh_local_book_original_automation_actual_20261123.json').read_bytes()),
         Day23_remote_verified_code_receipt=json.loads((V/'daily_20261123_receipt.json').read_bytes()),
         Day23_directory_allocation_metadata={n:json.loads((V/n).read_bytes()) for n in ['R_Day23_project_directory_allocation_depth2_actual_20261123.json','R_Day23_diagnostics_allocation_depth1_actual_20261123.json','R_Day23_model_runs_nonarray_allocation_metadata_actual_20261123.json']})
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();plan='Day23_verified185_actual_commit_blocked_resource_full_handoff_plan_v90_20261123.json';(V/plan).write_bytes(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(plan)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+mainname+"'),str(p),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(s);(V/'stage_Day23_verified185_actual_commit_full_handoff_v90_20261123_remote.py').write_text(s,encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb),explicit_primary_paths=len(primary),local_originals=len(local),finite_full_template_chars=len(template))))

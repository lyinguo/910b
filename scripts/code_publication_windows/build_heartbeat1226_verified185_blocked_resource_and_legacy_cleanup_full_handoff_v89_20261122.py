"""Preserve the exact current blocked-resource facts and every useful new read-only implementation."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
old=(V/'close_verified185_Turkish_fullNF0_capacity_official_and_cleanup_v88_20261122.py').read_text(encoding='utf8')
replacements={
'd80837dd904a0a0d22c166f2cb662b52ab1ff37705eb6da2be7d7203d5277790':'0bd677fc2ecbfd974d773895248e97f8bf1bff50331d10e1939e81423f27d2f7',
'68b5bf24388887aed46973f9eedacd5494d067941c36691c753389575aa7c0bd':'ba12c8d046dff5eb0d8e449740eb92f30e338a781fe75136a8786badfee65333',
'35777b97e7c5242dac244c813c6b7cbdc841318230b625bbe8d16aa86b870a48':'2b3ac4b53b59fc990cd00152cd68c548e9861201f5bff02a321acf1bb20ae5a9',
'len(rows)==2942':'len(rows)==2948',
'verified185_Turkish_completed_fullNF0_capacity_official_actual_v88_20261122':'verified185_heartbeat1226_resource_blocked_legacy_cleanup_full_actual_v89_20261122',
'pending_after_daily_20261122_verified185_Turkish_completed_official_cleanup_all_primary_v88_20261122.json':'pending_after_daily_20261122_verified185_resource_blocked_legacy_cleanup_all_primary_v89_20261122.json',
'late_actual_Turkish_fullNF0_capacity_official_cleanup_v88_20261122.txt':'full_actual_verified185_resource_blocked_legacy_cleanup_v89_20261122.txt'}
for a,b in replacements.items():assert a in old;old=old.replace(a,b)
a=old.index("    text='''本轮正式已完成阶段及官方维护晚注v88")
b=old.index("    full=note/",a)
old=old[:a]+"    text=q['full_handoff_template']\n"+old[b:]
old=old.replace("text.replace('CURRENT_FACTS_SHA256',sha(facts))", "text.replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_FACTS_SHA256',sha(facts))")
needle="    facts=note/'facts.json';put(facts,compact)"
extra="""    bee_cfg=R/'configs/queues/BeeSFT_completed_storage_NF_capacity_v5_verified185_exact137_wait_actual_capacity_serial_20261122.json'
    assert sha(bee_cfg)==rows[str(bee_cfg)]['SHA256']
    bq=js(bee_cfg);wait=Path(bq['wait_home']);bs=js(wait/'state.json')
    assert bq['required_free_bytes']==192555984252 and bs['required_free_bytes']==192555984252
    compact['existing_BeeSFT_v5_current_actual']=dict(config=str(bee_cfg),config_SHA256=sha(bee_cfg),wait_home=str(wait),state=bs,new_NN_job_exists=Path(bq['new_job']).exists(),new_continuation_home_exists=Path(bq['home']).exists(),not_NN_or_formal_NF=True)
    compact['original_three_resources_fresh_readonly']=q['original_three_resources_fresh_readonly']
    compact['same_owned_ID_alternative_exact_class_locations_only']=q['original_three_owned_exact_locations']
    compact['legacy_schema_full_numeric_reaudits']=q['legacy_schema_reaudits']
    compact['auxiliary_v1_KeyError_preserved_and_v2_original_registry_path_actual0']=q['auxiliary_reader_failure']
    compact['numeric_physical_allocation_metadata_only']=q['numeric_allocation']
    compact['current_local_workbook_and_original_automation_readonly']=q['current_local_and_automation']
    compact['no_new_physical_storage_method_adopted_or_protected_matrix_changed']=True
    compact['no_official_automation_write_this_round']=True
    facts=note/'facts.json';put(facts,compact)"""
assert needle in old;old=old.replace(needle,extra)
needle="current_resource_stop_earliest_UTC='2026-11-22T09:40:49.156504+00:00')"
extra="""current_resource_stop_earliest_UTC='2026-11-22T09:40:49.156504+00:00',current_full_actual_note=str(full),current_full_actual_note_SHA256=sha(full),current_full_facts=str(facts),current_full_facts_SHA256=sha(facts),current_BeeSFT_existing_waiter_actual=compact['existing_BeeSFT_v5_current_actual'],current_legacy_numeric_schema_readonly_reaudits=q['legacy_schema_reaudits'],current_original_three_resources_readonly=q['original_three_resources_fresh_readonly'],current_numeric_allocation_metadata=q['numeric_allocation'])"""
assert needle in old;old=old.replace(needle,extra)
mainname='preserve_heartbeat1226_verified185_resource_blocked_legacy_cleanup_full_handoff_v89_20261122.py'
ast.parse(old);(V/mainname).write_text(old,encoding='utf8',newline='\n')
prior=json.loads((V/'verified185_Base_failed_Turkish_same_serial_full_handoff_plan_v87_20261122.json').read_bytes())['full_handoff_template']
intro='''完整当前交接v89（2026-11-22 20:26心跳实际）：仍verified185，185交付/170数值内15超差205完整权重待交付310确缺/71完整原资格，本轮新完整200/新G/新资格/新增NN或全数值NF均0。本轮不是恢复运行：同Turkish已经于17:40:38完整真3/6 fullNF0，17:40:49完整200容量拒后全NN/NF/EX自然释放，当前停摆从该时点按最新钟算，不能以旧running/ready/等候器/CPU或元信息充NN。原Gemma4MLX8本日03:34:24完成的200/.840、三D.835相对.5988023952%和03:47:22双书全审是最近完整合格200，不重复计新增。Turkish小样本Accuracy1.0/对三D.705差41.84397%只是3条披露，不是完整200误差或首次200额外门槛。GLM4.1 Base视觉relativeRMSE .04171977>.02原正式NF仍失败，CPU诊断0不等于NF0，原失败及全部原值不改/不删/不翻0。
必须先完整读根H/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1及同facts1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7；本全文及同目录facts SHACURRENT_FACTS_SHA256、closing/late/pointer更晚实际优先。直接父完整v87正文327a420602dc4a91090489b44827dbf4323e707dacc4923e6a16242233dc9455/facts473cc4719104e22179085557bd6d696b0f3889e0a27cc8f7d8dffda4a5ad23e6，v88晚注83cd8c2ed8d37976f60842e93fedacde03876727b73acb41946b5d8af7084c41/facts6adc072c5bab4b9e06244621fa3b03901d287251327107c8901c107e5f106789/pending2b3ac4b53b59fc990cd00152cd68c548e9861201f5bff02a321acf1bb20ae5a9、全部祖先/65471字符archive全部强制规范、家族失败、精确清理例外和用户决定完整继承。有限完整当前交接加短引用，不无限拼接、不存密码。
本轮先完整读取以上指定文件及v88closing/late/pointer后，SHA冻结monitor/full latest-checkpoint与真实proc/argv/cwd/PPid/startticks/原内核FLOCK/cards/实际退出检查均成功：没有项目NN/NF/原EX，1..15空卡/0外部服务不动，原盘约20.9GiB，原180及旧source错误不能套本轮。20:34 monitor source_errors/inspection/newlog皆空，最新以末尾actual为准。原BeeSFT1696145仅无NN/无EX容量等候，每20秒核原盘192555984252bytes约179.33GiB门槛/全源/quiet+原EX，未NEWfresh3/200，旧1668102同样无子无EX不信号。缺口约158.4GiB，原盘增加至少200GiB问题仍待用户/管理员答复，未答不算批准或已扩容，禁擅换盘/扩/降guard/流式剪证据。
本轮进一步查152份旧schema证明：不能把缺少original_protocol_final_qualification_complete字段一律称为仅数值或未数值通过；它只是当前清理scanner的保守过滤，不修改71完整原资格。只读辅助v1因为旧discovery skipped行没有proof字段而KeyError，0删除/0NN，原source和错误留；新v2仅用原冻结ret179 proof_for从C唯一model的state/reg定位准确路径，actual0。逐SHA核13份明确全200/fullsource/fullKV或全forward数值证明，5款SmolDocling/T5Gemma/GraniteMLX/两InternVL35Flash和8款Cognitive/Calm/Nemotron/Heretic/Pali3Pt896/Smol500/GraniteMLXHF/MedGemma，其12个已识别系列均已按原精确例外清过，现存unpruned_nonexample_files=0；SmolDocling旧sample_000.npz三位布局不套其它索引，现存33个NPZ合约1.259GB含3个AA静态控制，全部保护，不因未识别布局删。两Smol22/Agentic旧proof只first3完整KV而其余metadata SHA，未作为全KV清理资格。旧pruned数组不能全数组再验，旧proof字节保持，所有错/invalid/empty/trunc/error/小样本/未验/失败与完整报告源配置日志/必要完整组保留，本轮新候选0/临时suffix0/删除0，不能宣称释放了新空间。
原cleanup watcher32bd...20:37:55 actual0/no_duplicate_or_formal_work_started_defer/no_array_or_file_change，当前活跃只做字节全同hardlink物理去重/no_new_unique_array_pruning，不是模型NN。其active_resource_waiters为空因为旧字段只识别required_bytes/needed_bytes而Bee实际字段required_free_bytes；本轮如实披露此元信息不匹配但未热改活watcher/source/config，160GiB proactive门槛仍每次低水位触发物理去重，Bee自身20秒循环不受该字段影响。最后实际没有重复可回收，不把这个0当正式NF或已聊天通知。
进一步只读model_runs物理stat和最大60个ZIP central directories：8513个唯一inode的NPZ，logical623301799545/allocated623323394048bytes约580.5GiB，没有单独NPY，最大文件多数ZIP-LZMA/DEFLATE已压缩；上限抽样中ZIP_STORED仅约4.57GB，不据此宣称全盘同分布。未实施新稀疏/重压/变更格式或修改任何受保护数值，不把设想当释放容量。空间问题是存不下证据，不是NPU/RAM不足；当前16x64GiB/128CPU/~2TiBRAM不能替代磁盘，串行max1不等于能绕容量。
原三根资源本轮20:55冻结readonly gate再次0但三root都unavailable；另只查这三个同modelID在已知Gemma3ForConditionalGeneration/Unknown两个类目录的6个精确路径，全部不可读，未扫描/借用别款checkpoint，不永久加310，不清原selector27daea/protectedstate，不复跑或改历史G。资源若恢复必须全原冻结file/tensor SHA和原前置门槛后新私有阶段，不按旧live重启。
原910b ACTIVE/原名称/原聊天/300分钟本轮只读复核均同，TOML SHA3f364fc3feceacd42be1241618b0bc4f27513df02a08b5f3b55471cf9ced2e6e；未再次写官方或TOML。local freshSHA/stat仍185书2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878/6099931bytes/mtime1795290387531571700/ctime1795290387529570400，server/state/reg/binding/ledger等canonical原SHA不变，changes0不写表。新有用R主源/实际metadata消费者/Windowsbuilder、旧失败版本及本地原回执已逐字节入明确CURRENT_SOURCE_COUNT全清单，Day22唯一43a149bb...不第二提交，Day23完整用途/真实consumer/依赖/fresh并发SHA/全diff/outgoing/相应检查和实际publisher plan_SHA审核后精确暂存正常push核remote；不漏源，不上传数组报告日志权重数据凭据归档或opaque transport。
下面共同原规范与完整家族决定继续；其中旧时刻和阶段一律以上述本轮完成/停摆及末尾完整actual覆盖，不拼接旧actual历史。全部资源款真实原协议910B200/三D5%、确缺跳过、无pending/资格复审且全量复查双书完整报告历史例外才可关闭，远未完成。

'''
template=intro+prior[prior.index('【必须先读的完整继承】'):]
template=template.replace('【Turkish自己的原模板/FP16构造闭合后，已接新私有串行真3，当前全NF以真实actual为准】','【Turkish真3/6已完整fullNF0，完整200容量拒，禁止按旧live重启】')
template=template.replace('NF实际退出另看下文。','NF和runner已在17:40:38.937624北京时间0，proofcf89fde1ab3067eba798cde73d0cd37f2798bc21d283b86c0bcabd4e6086a4a1；17:40:49.156504 capacity actual1，无首次200审批/fresh3/200/G。')
assert template.count('CURRENT_ACTUAL_JSON')==1
plans=['heartbeat1226_verified185_full_handoff_actual_reader_primary_plan_20261122.json','verified185_legacy_numeric_proof_schema_metadata_read_primary_plan_20261122.json','verified185_legacy_numeric_proof_schema_metadata_v2_primary_plan_20261122.json','verified185_exact5_legacy_all200_numeric_metadata_primary_plan_20261122.json','verified185_exact8_legacy_allstep_numeric_metadata_primary_plan_20261122.json','project_numeric_physical_allocation_metadata_primary_plan_20261122.json','original_three_owned_exact_locations_readonly_primary_plan_20261122.json']
primary={}
for n in plans:primary.update(json.loads((V/n).read_bytes()))
stage={'scripts/maintenance/'+mainname:old.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
for n,b in stage.items():primary[n]=sha(b)
names=['R_heartbeat1226_verified185_required_v87_closing_v88_current_actual_20261122.json','heartbeat1226_verified185_fresh_local_book_and_original_automation_actual_20261122.json','verified185_legacy_proof_schema_metadata_v1_observed_failure_20261122.json','R_verified185_legacy_proof_schema_cleanup_metadata_v2_actual_20261122.json','R_verified185_exact5_legacy_all200_numeric_metadata_reaudit_actual_20261122.json','R_verified185_exact8_legacy_allstep_numeric_metadata_reaudit_actual_20261122.json','R_project_numeric_physical_allocation_metadata_actual_20261122.json','R_heartbeat1226_verified185_original_three_resources_fresh_readonly_20261122.json','R_heartbeat1226_original_three_owned_exact_locations_readonly_actual_20261122.json']
local=[]
for n in names:
    b=(V/n).read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
five=json.loads((V/names[4]).read_bytes());eight=json.loads((V/names[5]).read_bytes());assert five['candidate_count']==eight['candidate_count']==0
allocation=json.loads((V/names[6]).read_bytes());fresh=json.loads((V/names[7]).read_bytes());samepaths=json.loads((V/names[8]).read_bytes())
q=dict(new_primary_expected_SHA256=primary,local_original_records=local,cleanup_discovery=eight['discovery_path'],cleanup_discovery_SHA256=eight['discovery_SHA256'],official_actual_readback=json.loads((V/'official910b_verified185_Turkish_completed_v87_actual_readback_receipt_20261122.json').read_bytes()),full_handoff_template=template,original_three_resources_fresh_readonly=fresh,original_three_owned_exact_locations=samepaths,legacy_schema_reaudits=dict(exact5= {k:five[k] for k in ['UTC','candidate_count','candidate_bytes','skipped','discovery_path','discovery_SHA256']},exact8={k:eight[k] for k in ['UTC','candidate_count','candidate_bytes','skipped','discovery_path','discovery_SHA256']},numeric_proof_missing_final_qualification_flag_not_reclassified=True),auxiliary_reader_failure=json.loads((V/names[2]).read_bytes()),numeric_allocation=allocation,current_local_and_automation=json.loads((V/names[1]).read_bytes()))
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();plan='heartbeat1226_verified185_blocked_resource_legacy_cleanup_full_handoff_plan_v89_20261122.json';(V/plan).write_bytes(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nq=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(plan)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif q.exists():assert q.read_bytes()==b\nelse:q.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+mainname+"'),str(q),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_heartbeat1226_verified185_blocked_resource_full_handoff_v89_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb),explicit_primary_paths=len(primary),local_originals=len(local),finite_full_handoff_template_chars=len(template))))

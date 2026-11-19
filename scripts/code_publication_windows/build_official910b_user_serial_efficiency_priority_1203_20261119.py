"""Update only the existing heartbeat prompt with the actual user serial policy."""
from pathlib import Path
import hashlib,json,tomllib
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes();old=tomllib.loads(raw.decode())
assert old['id']=='910b' and old['status']=='ACTIVE'
assert sha(raw)=='fc23db82024ccf891a01e1fcfdd751469c0b6d2b6b94bc8f2a2278f55e2ce77d'
launch=json.loads((P/'R_Huihui_serial_efficiency_policy_actual_launch_1158_20261119.json').read_bytes())
assert launch['serial_only'] and launch['max_active_models']==1 and launch['current_state']['phase']=='wait_for_single_model_complete_original_capacity'
first,tail=old['prompt'].split('\n\n',1)
decision='''【最新真实用户决定：串行优先、效率优先】用户原话“1. 平行保证不了那就串行一切以效率为准！别耽误我时间哇”。默认只运行一个模型的NN/完整独验，结束并真锁释放后立即接下一前置齐款；不等凑两款并行的总空间、不等定期检查，也不把并行或扩400GiB作为串行首款的额外前置。原16卡两路小样本验证只保留能力事实，当前不是凑并行的目标；以后仅有实际吞吐收益证据且总预算/隔离真过才另考虑。全部原模型数学、任务输入、dtype/预算/停止/评分、五件/数值/源/配置/日志、真实FLOCK和三D5%资格不减。
实际已新增并消费configs/queues/serial_efficiency_user_priority_20261119.json（SHA68fe0e21793c88e840d8f43d63e3e935f7ca90f002fff4fbd423b436bb900c9e），调度器scripts/serial_efficiency_dispatch_policy_20261119.py，9项串行/原锁/旧接续避让检查均过。新scripts/continue_huihui_completed3_serial_efficiency_wait_capacity_fresh3_fixed200_v2_1155_20261119.py SHA2059c83e466d0feb78af89d7cc7a7ed041c75370a417aefd10a4cb61b8790252已实际启动，只消费自己已过真3/27/proof7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59，不重跑已过真3。H=B/HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119，真实PID1557351仅快照，后续按proc/源SHA/PPid/startticks/实际state与EX续查，不重复启动。它每20秒按单款164003454340bytes=152.740GiB门槛与全部NN/NF/原EX检查，满足后新私有首次200测量批准→NEWfresh3/fixed0:199 NEW200；不改704BF16/2048/原thinkingFalse/outer4serial1/EOS/全数值/评分。新J200=B/HuihuiGLM46_completed_true3_single_model_serial_approved_NEWfresh3_fixed200_v2_real_FLOCK_1155_20261119目前未创建。控制器不写G/state/reg，等候不持原EX、不算NN/NF或恢复。
原zai-GLM无NN容量等候器1530341/源96dfdd09b7b14a4845e142cbcc6dcc4384130a362ff1bea777f9fcf2a059637b不热改、不信号；若它已在派发或235720910370bytes原预算恢复且其源/真实进程绑定通过，新Huihui先避让它完成，再串行接续，避免争锁。原global runner EX保持58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90，不启动旧并行协调器、不借分卡绕表写。新等待控制器cwd=R会在冻结monitor归other_project_python，必须按真实源/PID/state识别为无NN资源进程，不能冒称推理；旧严格清理或发布若拒它，先按真实NN/EX/阶段安全处理，不忽略未知进程、不热改冻结源。
11月19日11:52:33北京时间新state实查free99224961024bytes约92.41GiB、NN/NF/原EX全无，phase=wait_for_single_model_complete_original_capacity，首款仍约60.33GiB缺口；它不是“已经跑200”。Huihui3个原完整NPZ的333个整NPY记录预检无可直接共享重复项，未重跑真3/独验/压缩或采用新格式，也未删除任何矩阵。继续其它可闭合的自身适配与精确合规清理，不能全队等并行/AA，也不能减原guard/删未验/换盘硬跑。原完整报告/源码/配置/日志/必要例证全保留。原盘扩容尚无用户批准，这次串行决定不等于批准扩容/换盘/减少数值证据。
新有用串行控制/真实配置消费者/9项测试/部署与只读原记录源码要在R源码/scripts/configs并列最新完整pending，Day20全审不漏；11月19日已唯一commit不第二提交。完整交接正文/facts仍为本轮初快照，下述同目录closing/late_actual_notes里的本次真实用户决定、串行源/回执/currentstate优先，短引用不取消所有父祖先规范。
'''
prompt=first+'\n\n'+decision+'\n'+tail
(P/'automation910b_before_user_serial_efficiency_20261119.toml').write_bytes(raw)
(P/'automation910b_before_user_serial_efficiency_prompt_20261119.txt').write_text(old['prompt'],encoding='utf8',newline='\n')
args=dict(mode='update',id=old['id'],kind='heartbeat',name=old['name'],status=old['status'],rrule=old['rrule'],targetThreadId=old['target_thread_id'],prompt=prompt)
if 'notification_policy' in old:args['notificationPolicy']=old['notification_policy']
(P/'official910b_user_serial_efficiency_arguments_1203_20261119.json').write_text(json.dumps(args,ensure_ascii=False),encoding='utf8')
(P/'prompt_official910b_user_serial_efficiency_1203_20261119.txt').write_text(prompt,encoding='utf8',newline='\n')
print(json.dumps(dict(prompt_chars=len(prompt),requested_normalized_SHA256=sha(prompt.rstrip('\n').encode()),only_existing_prompt_update=True,serial_only=True)))

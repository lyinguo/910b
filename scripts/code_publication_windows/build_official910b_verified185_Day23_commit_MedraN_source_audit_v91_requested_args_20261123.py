"""Prepare a prompt-only official update; never write the live TOML."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tomllib
V=Path(__file__).resolve().parent
r=json.loads((V/'R_Day23_verified185_actual_commit_full_handoff_v90_preserved_20261123.json').read_bytes())
audit=json.loads((V/'R_MedraN_completed_FP16_native_source_contract_no_model_20261123.json').read_bytes())
assert r['passed'] and r['source_count']==2973 and audit['passed'] and not audit['installed_timm_available']
p=Path('C:/Users/lyg/.codex/automations/910b/automation.toml'); raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='3f364fc3feceacd42be1241618b0bc4f27513df02a08b5f3b55471cf9ced2e6e'
old=tomllib.loads(raw.decode('utf8'))
assert old['id']=='910b' and old['status']=='ACTIVE' and old['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
assert old['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300'
(V/'official910b_before_verified185_Day23_MedraN_source_v91_20261123.toml').write_bytes(raw)
args=json.loads((V/'official910b_verified185_Turkish_completed_v87_requested_args_20261122.json').read_bytes())
previous=args['prompt']; tail=previous[previous.index('【唯一canonical185】'):]
start=tail.index('\nDay22已唯一'); end=tail.index('\n全部祖先家族pending',start)
day='''
Day23北京时间11月23日已唯一正常push5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9/Code-Day2026-11-23，今日不第二commit。2972明确代码配置路径/303变更（302新增，仅已有configs/publication/code_only.json更新），remote同SHA/worktreeclean；R39项、本地19项、25MJS语法检查全过，初R测试入口缺SOURCE/OUT参数失败及v2恢复原参数通过全部原件保留。审核plan_SHA66dad86f3b449fc5f3b66479c8157a837678c6657dc80af101572e92d818b992/full diff SHA cf7c323c5ef23d2ac977bbf59dc80d855404b8ca852c36eeb1b133d35d38cd9a/alloutgoing完整树过。receipt C/daily_20261123_actual_reviewed_commit_remote_receipts/daily_20261123_receipt.json SHA51d86b8876b24f3224d4d5c4b9d40ef841f2fd4a7acb593e0f36bd988400c554。R真实publisher配置code_only.json唯一已审核精确CAS39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5→1585e1dce7edea8a7b3c99bf27ae4b2840b715ac3b054542d55141bbc44906e9；publisher本体e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30与模型/run.py/pipeline/原runner/Gwriter/PAUSE/两书/state/reg/ledger均不改，不由此放宽任何runtime源SHA。两份Meta inline200真实输入协议只Git准确排除，R完整原件及生成器/YAML保留，不把数据上传Git。
当前2973初主源和本轮新MedraN只读诊断/官方维护与更晚closing完整pending全部原字节逐SHA保留，Day24首轮逐用途/consumer/deps/fresh并发SHA/完整diff-alloutgoing/对应checks/实际publish_code_daily.py消费code_only及审核planSHA后一次正常push核remote；无变不空提交，push失败续同已审commit，不等提交才跑。唯一https://github.com/lyinguo/910b.git、独立codex/910b-source-layout-20261114、D:/code-lv/910b，不merge/force/Gitee历史/R master-remote改动，不gitadd.或扫workspace，不传权重数据数组报告日志keys.env凭据归档旧历史或字面base64运输payload。主实现只R源码/scripts/configs，workspace仅证据，R源freshSHA导独立分支不反盖模型源，本地README/ignore/策略保持已审。
'''
tail=tail[:start]+day+tail[end:]
H='/root/projects/comperror/workspace/diagnostics/batch_910b/heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
mandatory=f'''必须先完整读{H}/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1，再同目录facts.json SHA1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7；必须完整读当前全文{r['late_actual_note']} SHA{r['late_actual_note_SHA256']}及同目录facts.json SHA{r['facts_SHA256']}，再H/closing_addendum.json、late_actual_notes与D/A100_ALIGNMENT_HANDOFF_CURRENT.json更晚交接。v90为Day23代码实际提交/资源阻塞完整交接，旧Day23尚未提交说法已过时。父v89及v87全文327a420602dc4a91090489b44827dbf4323e707dacc4923e6a16242233dc9455与全部祖先/65471字符archive全部强制规范、家族失败、精确清理例外和用户决定完整继承，短引用不取消约束。更晚真实proc/原EX/退出/closing及最新用户决定优先，不按旧live重启完成款。完整有限交接加短引用，不无限拼接、不存密码。'''
current=f'''【当前严重阻塞｜正式模型NN/全数值NF未运行】
最新实际monitor/proc/原FLOCK/cards于11月23日02:22北京时间确认无项目模型NN/NF/原EX，NPU1..15空、0外部VLLM不动，原盘22341648384bytes约20.81GiB。当前停摆最早可证11月22日17:40:49.156504北京时间Turkish200容量拒，按最新钟算时长，最近完整合格200仍Gemma4MLX8名下实际ownBF16于11月22日03:34:24/两书03:47:22；最近全数值小样本Turkish17:40:38。本轮新增完整200/G/资格/模型前向/fullNF均0，Day23代码提交/R单元检查/CPU源码审计不充正式评测恢复。
BeeSFT v5原无NN等候器1696145仅旧PID快照，仍20秒原盘预算192555984252bytes约179.33GiB真实满足且全source/canonical/全部NN-NF-controller-EX释放才新私有内部批准→NEWfresh3/fixed200；不重复/热改/杀原waiter或已过3/storageNF，旧无子无EX1668102不动。cleanupwatcher1629242只是完整raw相同硬链接物理去重，无新增unique正确非例证清理。扩原项目盘至少200GiB问题仍待答复，未答不批准或已扩容，不换盘/扩/降guard/流式剪证据/删未验。继续有自身原前置的未适配与超差，不全队等在线A100/定期，不重复已过小样本凑运行。
【已完成阶段，不重放】Turkish own4SF686BF16存储→原autoFP16、自己32Llama/CLIP24/select-2/full128258/唯一EOS128009/2048/outer10/无system-date，C唯一row690/三D.7050/.7050/.705/G空。J2真3/6/NPU4完整NF0 proofcf89fde1ab3067eba798cde73d0cd37f2798bc21d283b86c0bcabd4e6086a4a1，200经验4228496264073bytes约3.85TiB容量拒；全3必要数值12.56GB/五件/源日志保留，不复跑/重压/删。GLM41Base own概率协议真3/6的全NF1仍失败：last_hidden_state relativeRMSE.04171977283130191>原.02，CPU诊断localoperators过不改变全视觉失败，根因未证；原全失败数组/五件/source/日志保持，不放宽阈值、改数学/提取、翻旧job或盲重跑。Hancom3/6 fullNF0但3.94TiB、Xtuner3/6144但14.53TiB、More3/3但3.96TiB、BeeRL/Ovis/Phi等所有旧已过阶段不重复，原预算/EOS/cache/组序/评分及全部家族约束详见全文。
【MedraN新只读源码核查，不是修复或NN】自己的原FP16旧真3/NF失败与vision_blocks.2非有限全部保持。新R/scripts/diagnostics/review_medran_original_FP16_graph_and_loader_source_contract_20261123.py SHAe0bc21686108b589d3edea7acb4e22036da39b782522437edeb241b9984c0a34于11月23日02:22:43实际source-audit0，无权重数组加载/learnedCPUforward/模型NPUforward。{audit['home']}/review.json SHA{audit['review_SHA256']}。明确旧CPU诊断重放同一parameterized图，不是原HF runtime等价证明；当前原HF视觉TimmWrapper要求真实timm，安装包与原图注明的两个精确原source路径都不存在。未证明keepFP32或eps新根因，未构造/加载原HF模型，不改dtype/BF16/eps/数学/阈值/输入/预算、无新资格/G，不借Unsloth权重或装包/stub。只能继续原source可证闭合，不能因CPU同失败宣称原A100也失败。
本轮已复核旧完整numeric schema和正确非例证/临时元信息：可新清candidate0/tmp0，0删除。原盘R约837.7GB，model_runs约750.6GB，其中uniqueNPZ约623.3GB，JSON约91.25GB/DB10.35GB/JSONL6.68GB主要是完整数值元信息与原始trace/profile，不能按后缀当缓存删除。完整五件、全部逐条明细、源码/配置/日志、全部小样本/失败/错invalidtrunc和必要完整数值例证保持；不要把既往pruned数组说成仍在。
'''
prompt='持续原910b ACTIVE/原名称/原聊天/每300分钟，串行max1继续实际A100工作簿D→910B相对5%适配。锁释放且原前置与资源真过立即接下一款，不等定期。保留完整结果、源码、配置、日志和必要完整数值，停摆/错误及时普通可见中文报。\n\n'+mandatory+'\n\n'+current+'\n'+tail
args.update(mode='update',id='910b',kind='heartbeat',prompt=prompt.rstrip(),name=old['name'],status=old['status'],
            rrule=old['rrule'],targetThreadId=old['target_thread_id'],notificationPolicy=None)
out=V/'official910b_verified185_Day23_MedraN_source_v91_requested_args_20261123.json'
out.write_text(json.dumps(args,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(path=str(out),prompt_chars=len(args['prompt']),prompt_SHA256=hashlib.sha256(args['prompt'].encode()).hexdigest(),
    original_name_status_thread_cadence_preserved=True),ensure_ascii=False))

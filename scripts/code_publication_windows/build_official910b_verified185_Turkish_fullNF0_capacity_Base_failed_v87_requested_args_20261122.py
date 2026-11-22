"""Prepare official-only heartbeat prompt update using actual completed numerical acceptance."""
from pathlib import Path
import hashlib, json, tomllib
V = Path(__file__).resolve().parent
r = json.loads((V/'R_verified185_Base_failed_Turkish_same_serial_full_handoff_v87_preserved_20261122.json').read_bytes())
assert r['passed'] and r['source_count'] == 2942
path = Path('C:/Users/lyg/.codex/automations/910b/automation.toml')
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'abb8fe4531593f121a4f959489bd045fc1184b00b6d593ea834e950d2fe70c90'
old = tomllib.loads(raw.decode('utf8'))
assert old['id'] == '910b' and old['status'] == 'ACTIVE'
(V/'official910b_before_verified185_Turkish_completed_v87_20261122.toml').write_bytes(raw)
args = json.loads((V/'official910b_verified185_GLM41Base_serial_v86_requested_args_20261122.json').read_bytes())
previous = args['prompt']
tail = previous[previous.index('【唯一canonical185】'):].replace('2902', '2942')
H = '/root/projects/comperror/workspace/diagnostics/batch_910b/heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
mandatory = f"必须先完整读{H}/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1，再同目录facts.json SHA1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7；还必须完整读本轮全文{r['full_actual_note']} SHA{r['full_actual_note_SHA256']}及同目录facts.json SHA{r['facts_SHA256']}，再读H/closing_addendum.json、late_actual_notes和D/A100_ALIGNMENT_HANDOFF_CURRENT.json更晚交接。父v86全文f9c21c78daf820377dc4bf2fd13009cbccf46cc94255d9561cf819a0f4bf29eb/facts7e74100de0d4dda189621ad31e1549713912a4fedd31901a6bb7d511871d11f7及全部祖先/65471字符archive所有强制规范、家族失败、精确清理例外与用户决定完整继承，短引用不取消约束。正文/facts是初快照，更晚真实proc/原内核EX/自然退出/closing与最新用户决定优先，不按旧live重启完成款。2942初主源和更晚完整pending/真实YAML-protocol-smoke/本地原回执与原prompt/TOML全部原字节逐SHA保留，不漏有用后续提交文件。完整交接加短引用，不无限拼接、不存密码。"
section = '''【当前严重阻塞，Turkish已经真3全数值通过，200容量拒，不重复】
ytu-ce-cosmos/Turkish-LLaVA-v0.1当前C唯一row690/三D.7050/.7050/.705/G空，不能用旧state row689。自己的4SF16709608888bytes/686BF16存储→原nonCPUauto请求FP16，32Llama Q32KV8x128/CLIP24-336/select-2/576patch/plain theta500000/context8192/full128258头（原resize128257→128258在legacy remap之前）/图ID128257/pad128256/唯一EOS128009/原Greedy2048/outer10内serial1保持。自己的模板没有system/date，不套Hancom的头/三个EOS/date。源/输入/自己额外行RNG/原F32RoPE与全部686 own source-cast-actualNPU SHA闭合，不借另一款checkpoint/数据/词表/字体/输出，历史AAengine/tensor没采如实披露。
原v1独立模板错误0NN拒、v2全200像素/token/source通过后旧物理row689元信息拒，旧失败/数值/源码日志留。v3只准确重绑C690/三D，frontend200重放0；自己原CPU构造v2实际0/learnedforward0，sourceproofc331ec7cd0feb34c916936a961b504af49f5e5c6f60aa78356cdf63ff694d891/额外行NPZ7112aecadff2eb4fef36a4bc61004e25688c4c6db78fd8ecbff4448b8c921f30。所有4329 R/AFS filterTrue-False顺序和全200图/pixels/Rusttokens/686meta/4SF SHA完成，不重放。原未使用J1/v1在Base失败后自然拒08:56:43UTC，0NN，别重启。
J2=B/TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122 job60b82a03905e82b79e6e0429a0958f85dbfa0c42f2ec54fce67c164d6da355ab，报告20261122_170517/NPU4/真实3条6forward/raw[B,B,A]/自己soleEOS每条2步。11月22日17:40:38.937624北京时间model1仅完整baselinegap/完整数值NF0/runner0，proofcf89fde1ab3067eba798cde73d0cd37f2798bc21d283b86c0bcabd4e6086a4a1，8034完整数值发生、全部686source→actualNPU/full128258argmax/32物理KV/视觉和全部externalSHA过，2231kernel仅首vision-prefill。error/empty/invalid/trunc0，恰完整五件永久留。Accuracy1.0只3条，对完整D差41.84397%只是小样本披露，不是200硬件误差/资格，不新增小3对完整D5%门槛；不能把runneraccepted3当完整原资格。
H=B/TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122，controller b897c07959fcd32d5bcf07eb21f1b4747f8fd753ada5269d3f01589e721cbad5/真实queue bacb9bf2590dc89181ff288e8a4e196b3d648caa8f9fcf74d5478883f0f36d51，17:40:49.156504 actual1仅正式200容量拒。capSHA a029a47b242df3e0b411e501a1f79a5cef4f604d77c6633dcce9230f15d22605，3完整NPZ12557175205+meta4966980bytes/全200自己原形状/原unknown保守1.2603504693758312/原factor4+6GiB与硬3-log1-pre.5-sample2GiB保护不降，经验4228496264073bytes约3.85TiB>实盘22721392640bytes约21.16GiB。未首次200批准/NEWfresh3/200/G，全部NN/NF/controller/原EX自然退出；旧running state过时，不重复3/NF/controller、不热改source/job/YAML/deadline/信号，三个12.56GB完整小样本必要例证不能当已验可清非例证。
容量恢复只新私有接续准确引用同proof→内部首次200测量批准→NEWfresh3/fixed200，原预算/头/cache/停止/组序/数学/提取不改。reader scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py完整核monitor/真实proc/内核EX/退出/current trace/cards/proof/cap。全200/fullNF/原资格/三D5%全过及全NN-NF-controller-worker-EX释放后才原writer11/ArtifactTool实际author+PNG实看/全备份/真原EX跨server-local精确CAS只本G/fulltargetaudit新186，不预写或伪增资格。

【GLM4.1 Base自己的原概率协议真3/6已成，但全NF失败，不接受】
zai-org/GLM-4.1V-9B-Base row695/三D.9100/.9100/.91/G空，own4SF20585644104bytes/704BF16/40text Q32KV2x128/24vision/full151552头/context65536。原adapter c95d7286200f93a000f7232cd807ec64e80bb4e9ac5911f06976258ed328184d的option TEXT teacher-forced平均prompt logprob，manual prompt/ownk/plp[-(k+2):-2]均值max/tie/每option temperature0-max_tokens1-prompt_logprobs0/outer10/保存fallback2048不变，不概率换生成、单字母logit、强字母或调slice/数学/预算/禁thinking/筛样本凑D。自己前端200/525请求/4329顺序/200图片/独立完整pixels/token/4SF SHA/704meta已过，不重做。无借checkpoint/数据/词表/字体，自己PIL原RGB不是其它款白底。
同J3=B/GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122 job390cf828750dde9f8fc6f19e0aff10ae18f54b4db43d819e598edaf8c2357ad4/report20261122_160752/NPU4真3/6，小Accuracy1.0仅3，不是200成绩。北京时间16:56:31.981387 runner actual1/model1/NF1/accepted[]，原completion112bf778aa5a960357c2c10570223df1d3d9383d5ffb2c0be7f1450260327d50/NFstderr0a0fb2a00fc540c02b83e92c79a9e7da3b82d34507e253da9abb58acb4b18189：('0/A/model.visual/last_hidden_state','relative_RMSE',0.04171977283130191,0.6875)>预先固定.02。这是独立张量误差，不是评分D5%。自己的原MODEL ffc7d6f868514c2a5778580567745b7fdec2e2c1c27f4e73e0cac3076e627e55/NF6828f4bc390d3fda8ad81e4b6f5c3c25afe0afbaa4eab3d79bdc8b97f167eddf/全值capture82936193223d4dfa42e41276c2cf9df790921febd37b683b97254d6ce776aa47不改；三个完整NPZ4754658636/4773874381/4100881021bytes和全五报告源配置日志/失败值全部保护，不能删/翻0/重复3或放宽阈值接受。
诊断 scripts/diagnose_glm41base_completed_visual_full_value_CPU_reference_drift_v1_20261122.py c1d37481704e0cf7f6da972ec5f7fa19edda26f0cc7f20964b2a3063f0dcb4a3于UTC09:15:41.213993实际diagnostic0，279module/304tensor/255230976完整值/自身181vision参数与全部source实际SHA核同；local_operator_failure_checks空但整视觉last_hidden_state仍.04171977，pooler.01954896。formal_acceptance=False/qualification=False/新增NPUforward0，旧NF仍失败，根因未证明。不能删除全视觉检查、提高.02或把CPU诊断0充NN/NF恢复。后续若真实图或参考错误被证明，新的私有不可变阶段验证，绝不热改/翻旧失败job。

【已完成Hancom和其它原小样本不重放；实际资源与停摆】
Hancom3/6全NF0于11:57:44.370153，proof950652f9abd8184cb66143c2946865b01e232d106ff8c392a66d5fad87bb32a6；2004330453003134bytes约3.94TiB拒，全部12.95GB小样本保护。Xtuner3/6144 fullNF0但14.53TiB、More3/3 fullNF0但3.96TiB同样完成不重启。原BeeSFT已过3和完整存储NF，v5合法185容量接续器仍只等原盘预算192555984252bytes约179.33GiB且全部source/quietEX真过再NEWfresh3/200；等候器不持EX、不充NN，不重复。0卡其它VLLM不动，16卡各64GiB/128CPU/~2TiBRAM，用户串行max1，以实际存储与准备程度调度，不能空卡声称评测在跑。
当前停摆最早可证Turkish200容量拒17:40:49.156504北京时间，时长按当前钟算；17:46实际全项目NN/NF/原EX皆无，监视source_errors/inspection/newlog全部空。最近完整200合格仍Gemma4MLX8 11月22日03:34:24.274295/双书全审03:47:22.926705，.840对三D.8350/.8350/.835差.59880239520958%；最近小全数值是Turkish17:40:38。不能延续父Base停摆跨过Turkish真实恢复，也不能用加载、CPU诊断、前端、meta、算子、ready、空卡、旧结果或等候器充正式运行。严重停摆首行“严重阻塞｜正式评测未运行”，关键检查/连接失败首行“严重告警｜无法确认正式评测状态”，写真实阶段/样本前向卡/已证原因/最近完整与合格结果北京时间/最早可证时长与解除动作，恢复当轮也报，磁盘NOTIFY不是聊天通知。
本轮已再查185基线完整验收证明下的正确非例证和临时文件，metadata discovery2eec9de701b82477a646d7e2762277c73ca21f5fda93185cdf4d1673f63da068，候选0/tmp0/新删除0；不能把未显式完整原协议声明或旧pruned例外误当仍全数组可复验，也不能扩大删未验/失败/小样本/错invalidtrunc/必要例证。现有watcher32bd...仅全raw相同硬链接物理去重/no_new_unique_array_pruning，不冒称定期删unique矩阵。用户原400GiB扩容问题未答不是批准，不擅扩/换盘/降guard/流式剪证据。原报告/全部逐条明细/源码配置日志及必要全数值保持；继续其它有自身原前置且空间真够项目，不等AA/定期、不盲重跑已过3。当前新完整200/G/资格0，仍185/170内15超205待310缺/71完整原资格，数值内≠完整资格。

'''
prompt = '持续原910b ACTIVE/原名称/原聊天/300分钟，串行max1继续实际910B适配。锁真实释放即接其它自己的原权重有前置且资源真过款，不等定期；完整报告、源码、配置、日志及必要完整数值例证保留，错误/停摆及时普通可见中文报。\n\n'+mandatory+'\n\n'+section+tail
args.update(mode='update', id='910b', kind='heartbeat', prompt=prompt.rstrip(), name=old['name'], status=old['status'],
            rrule=old['rrule'], targetThreadId=old['target_thread_id'], notificationPolicy=None)
out = V/'official910b_verified185_Turkish_completed_v87_requested_args_20261122.json'
out.write_text(json.dumps(args, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(dict(path=str(out), prompt_chars=len(args['prompt']), prompt_SHA256=hashlib.sha256(args['prompt'].encode()).hexdigest(),
                     original_name_status_thread_and_cadence_unchanged=True)))

"""Preserve all explicit useful sources and rewrite finite current handoff for the same task."""
from pathlib import Path
import ast, base64, hashlib, json
V = Path(__file__).resolve().parent

def sha(b):
    return hashlib.sha256(b).hexdigest()

old = json.loads((V/'verified185_Hancom_completed_GLM41Base_own_probability_full_handoff_plan_v86_20261122.json').read_bytes())['full_handoff_template']
intro = '''完整当前交接v87：verified185未变；185交付、170数值内/15超差/205完整权重待交付/310确缺、71完整原资格。本轮新增完整200/G/资格0。Hancom真3/6 fullNF0已完成但200约3.94TiB拒；GLM4.1 Base自己的原概率协议已真实3/6完整落盘，随后全数值NF视觉relativeRMSE .0417197728313超过预先固定.02而正式失败，不能接受、改阈值或翻旧job0。已完整诊断255230976原值、279模块304检查，局部单算子无越界但整段视觉仍失败，诊断0不等于NF0，根因未证明。下一款Turkish-LLaVA自己的全部前端200、原CPU构造及source readiness通过后，旧串行v1自然只读拒绝已保留；新私有v2准确审查Base失败并等所有前序NN/NF/controller/原EX自然释放，已经NEW真3/6在NPU4完成，实际当前全数值独验/自然退出/容量以末尾facts和更晚真实proc为准，不按旧running重启。串行max1，不重复任何已完成3/NF；完整报告、所有逐条input-output-raw-ref-error-token、源码、配置、日志和必要完整数值例证保留，本轮无删除，不等五小时才接续。
直接父v86 full正文SHA f9c21c78daf820377dc4bf2fd13009cbccf46cc94255d9561cf819a0f4bf29eb/facts7e74100de0d4dda189621ad31e1549713912a4fedd31901a6bb7d511871d11f7/pending d369ebbf961a9276fc7e70c2f49d88d5e99c1740e2dce2e4b4115a1c5f8af34d，及根handoff/facts/全部祖先/65471字符archive全部强制规范、家族失败、精确清理例外与用户决定完整继承。短引用不取消约束，本全文有限重写当前段，不无限拼接、不存密码。官方原910b ACTIVE/原名称/原聊天/300分钟保持，仅official工具维护，活跃TOML只读。当前完整主源/config CURRENT_SOURCE_COUNT个，全部明确路径/原字节/SHA与本轮本地原件/之前官方prompt-TOML原件已归档，后续Day23全审不漏，今天Day22不第二commit。

'''
text = intro + old[old.index('【必须先读的完整继承】'):]
a = text.index('【下一款GLM4.1 Base自己的原概率协议'); b = text.index('【Bee SFT：', a)
base = '''【GLM4.1 Base：自己的原概率协议真3已完成，正式全数值失败，不接受/不重跑】
zai-org/GLM-4.1V-9B-Base C唯一row695/三D.9100/.9100/.91/G空，自己的4SF20585644104bytes/704BF16，40text/H4096/Q32-KV2x128/151552完整头、24vision/H1536/patch14/temporal2/merge2/context65536。原R YAML606ac2677328d9b7dfe02dc0e5ef546ea1d56bce4d9c69485278a2e21fb35860、原glm41v_base_discriminative.py c95d7286200f93a000f7232cd807ec64e80bb4e9ac5911f06976258ed328184d/父c435cdcfa181637ed69181ab1d9aa5ca0aaaca17efb6f0015be74e400f10e86c/generic da65f91d00696a2becf43121c793de985eb8364fd18f2a0164ff8d53541602db保持。不是生成/单字母logit；原option TEXT teacher-forced mean prompt logprob，原[gMASK]<sop><|user|>与图标记/clean原题/Answer:X.+空格选项文本/<|assistant|>换行，原k=自己空格cont的token数，plp[-(k+2):-2]均值/max/tie顺序保持；每选项temperature0/max_tokens1/prompt_logprobs0，fallback2048/outer10保持。全200均有选项，共525请求，无fallback。自己PIL直接RGB，不套其它款白底；EOS资产[151329,151336,151338,151348]原字段保留。历史AA engine/tensor未采如实披露，不用CPU/MLU充AA或NPU成绩。
自己无权重前端200于11月22日11:51:27.334479 full0，FRONT=B/GLM41Base_own_original_option_logprob_native_frontend200_meta_no_NN_v1_20261122，source4501062eb42a73331530ce950b0bdd835b5126800e67dfc67b57d0d3c94fdd9c，frontend0d4777bbaaab10a0701bc27296850e9441e07c3284717e52bc30f8f089176228。全4329 R/AFS filterTrue-False原prompt/ref顺序/all200图片字节/全部525原manual prompts-token score区/独立TV2uint8BICUBIC+NumPyF32normalize→nativeBF16全位/704meta/4SF全SHA过，0/99/199各option完整输入例证保留，不重做前端或借输入权重。
MODEL universal_eval/model/adapters/glm41base_own_original_option_logprob_BF16_native_v1_20261122.py ffc7d6f868514c2a5778580567745b7fdec2e2c1c27f4e73e0cac3076e627e55，INPUT b6ebc5660645aa49607a3312dd253105c38db15bce894c79db2eecc424078dd8；全值utils/complete_module_values_tree_20261122.py82936193223d4dfa42e41276c2cf9df790921febd37b683b97254d6ce776aa47，NF scripts/verify_glm41base_own_original_option_logprob_BF16_complete_values_v1_20261122.py6828f4bc390d3fda8ad81e4b6f5c3c25afe0afbaa4eab3d79bdc8b97f167eddf固定BF16相对RMSE.02/F32logprob1e-4。所有leaf/composite输入kwargs输出全值/每次实际copy/原完整头/40KV/原切片均值和auxmax1/full source-cast-actualNPU/externalSHA，只有全raw位+shape+dtype相同才共享完整NPY，所有语义发生保留，不丢值/改阈值。6项真实artifact/cache/corrupt/nonfinite/schema拒绝测试均0；原完整FortranNPY+ZIP-LZMA无损存储，数学/精度/预算/位置/组序/所有mask不变。
同J3=B/GLM41Base_own_original_option_probability_BF16_true3_v1_real_FLOCK_20261122，job390cf828750dde9f8fc6f19e0aff10ae18f54b4db43d819e598edaf8c2357ad4，报告20261122_160752/NPU4/真实3条6选项原前向。UTC提交0A08:14:41.774577/0B08:14:47.933771/1A08:26:59.588064/1B08:27:05.323335/2A08:39:25.350153/2B08:39:30.233846；小Accuracy1.0只3条不当200误差/资格。五件恰完整/error-empty-invalid-trunc0，model1仅完整baselinegap。但独立全数值NF在首whole visual失败：('0/A/model.visual/last_hidden_state','relative_RMSE',0.04171977283130191,0.6875)。这是张量误差，不是A100分数差5%。首textembedding 1384448值全同不能取消visual失败。
原controller H=B/GLM41Base_own_original_probability_true3_capacity_fresh3_fixed200_serial_v1_20261122 于北京时间16:56:31.994131实际1；runner completion16:56:31.981387 actual1/model1/NF1/accepted[]，completion SHA112bf778aa5a960357c2c10570223df1d3d9383d5ffb2c0be7f1450260327d50，NFstderr0a0fb2a00fc540c02b83e92c79a9e7da3b82d34507e253da9abb58acb4b18189。原三个完整NPZ4754658636/4773874381/4100881021bytes/全部必要小样本和失败数值/原五报告源YAML日志保持，不删/重压/翻0/重复真实3，不存在Base批准/fresh3/200/G。
只读CPU诊断 scripts/diagnose_glm41base_completed_visual_full_value_CPU_reference_drift_v1_20261122.py c1d37481704e0cf7f6da972ec5f7fa19edda26f0cc7f20964b2a3063f0dcb4a3，H=B/GLM41Base_completed_genuine3_visual_full_value_CPU_drift_diagnostic_only_v1_20261122，于UTC09:15:41.213993实际diagnostic0。自己181个vision参数/source-cast-actualNPU SHA核同；279原module/304完整张量检查/255230976值，whole last_hidden_state仍.04171977283130191、pooler.01954896251754227，trajectory与local分开，local_operator_failure_checks=[]，checklog229d2221f2c42ecbf8469fc23ab5403e625553680593452c73e4495c6b77aeba。formal_acceptance=False/qualification=False/新增NPU前向0，旧NF失败仍必须通过；只能说全视觉误差被复现，不能凭局部通过猜根因、增阈值、删除全局检查或声称NF0。若未来有真实数学/接口纠正，必须新不可变源/阶段验证，不热改活源或接受旧失败。

【Turkish自己的原模板/FP16构造闭合后，已接新私有串行真3，当前全NF以真实actual为准】
ytu-ce-cosmos/Turkish-LLaVA-v0.1当前C唯一row690/三D.7050/.7050/.705/G空，旧state row689不改但不能发布。自己的4SF16709608888bytes/686BF16，按原nonCPU auto请求FP16/32Llama Q32KV8x128/CLIP24-336/select-2/576patch/plain theta500000/context8192/2048/outer10内serial1/Greedy。自己原resize在legacy remap之前128257→128258，实际完整head128258/图ID128257/pad128256/唯一EOS128009，不套Hancom三个EOS/head/date。自己的chat模板无system/date，原BOS+user头/strip原prompt/eot+assistant头双换行，不注入日期。自己原transformers_vlm.py507b940.../YAML2a5e440def37d2d882d587d8077de0e2ea280de46d708213b966944ba30e88b7保持，不借另一款checkpoint/数据/词表/字体/输出，不改预算/停止/提取/数学凑分。
frontend v1独立模板错误沿用Hancom system/date，在第一条0NN拒绝，原source918038fcc5f5986d60ae0bc27e17f9f724729a404a7e50c67a5d8fa4d2aef43b/失败全留。v2只改独立参考为自己真实模板，source1d3d39f619aff6e837fe2755a9414d8d5a3edaac5a049062098178ac4ab14e22，自己全200独立pixels/token/source于UTC08:29:49完成，最后旧物理row689元信息断言拒，旧v2仍actual1不翻0。已完成前端字节4fe9c7ee3d26954dcb56d64f8bd78ba3445756c56754a8c96c554668d5c79f54，全4329 R/AFS filterTrue-False顺序+200images+Rusttokens+独立像素逐位+所有686meta/4SF SHA正确，0/99/199完整输入保留。metadata v3 c2a17c9b031bffb483633b9e073f9ed35e6d924ff362876a0bc54a2e7f241134于UTC08:40:43.807286实际0，只重绑C690/三D，前端200重放0。旧CPU v1因原前端v2 actual1根本未启动；新原CPU构造v2 source239a8ba3a2d54b3a2b75c39a32122b1d2d9d872991a8a981036fc478289c50f2于08:46:43.536279实际0/learnedforward0，自己的686 BF16→原请求F16/resize-before-remap/全部行位/source remap/自己额外行RNG/原F32RoPE全SHA证明。CPUsourceproofc331ec7cd0feb34c916936a961b504af49f5e5c6f60aa78356cdf63ff694d891/自己额外行NPZ7112aecadff2eb4fef36a4bc61004e25688c4c6db78fd8ecbff4448b8c921f30均完整留；不把CPU构造当前置NN成绩。
独立MODEL universal_eval/model/adapters/turkish_llava_own_original_Transformers_FP16_2048_native_v1_own_original_20261122.py 7fdb55b9530d0f83065c264902a4c3f3f637c7ea780b688fdd34b15f55c99553；GRAPH universal_eval/model/turkish_llava_own_saved_Transformers_graph_20261122.py25713577a94fc7d6c871a798d3652b7dd771d0446036dc628b61db5a1fa999e6；NF scripts/verify_turkish_llava_own_original_Transformers_FP16_2048_full_values_v1_own_original_20261122.pyb6120a53e7107b5d40fdeea49b4278d7ff67517c598aae84c1fe504cfb2278e2。观察器和完整独验使用已验架构经验，但每个source/input/checkpoint实际都是自己686与自己的额外行，不借学得数组。全部原32物理KV/full128258头/每前向argmax/全部视觉/F32RoPE/所有source-cast-actualNPU和ALLexternalSHA原数值门槛保持。
YAML configs/LlavaLlamaForCausalLM/ytu-ce-cosmos--Turkish-LLaVA-v0.1-own-original-Transformers-FP16-2048-native-v1-own-original-20261122.yaml40783aba4b25cad2718e2abf910b55d46297667ce168102df397310ea58dd7ca；protocolf984bdbda745e335a183073046bd273ad4f1d7db692a398e0e9145086b2803e8。readiness08:49:43.143351 full0，IModel/import/full686meta/真实cache接口/validate_config/check_sources/config3/200及三D/current185/source绑定过。不重跑前端/CPU构造/ready。原已准备J1 5c5656e0b71c6c04fd2f0a7f41eba35efb61e97701649ddde5f8ba52cdc12154没有启动NN；controller v1按Base失败准确自然拒08:56:43.963710 actual1保留。
新controller scripts/continue_turkish_llava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122.py b897c07959fcd32d5bcf07eb21f1b4747f8fd753ada5269d3f01589e721cbad5，真实queue bacb9bf2590dc89181ff288e8a4e196b3d648caa8f9fcf74d5478883f0f36d51，H=B/TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122。它只精确核Base原completion/失败全文SHA+旧unusedJ1/v1自然退出，review24a02bc46a2e45ebf08b27c6f0c9fd82fff33bcbe001bbc9f786209183fefc1b，不把Base接受/放宽阈值。新J2=B/TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122 job60b82a03905e82b79e6e0429a0958f85dbfa0c42f2ec54fce67c164d6da355ab，于UTC09:05:03.637251 controller1719822/runner1719834真原EX/model1719846仅启动快照，不热改/信号/重复它。报告20261122_170517/NPU4真实3条6forward已完成，各2步原soleEOS，raw[B,B,A]全部error/empty/invalid/trunc0；小Accuracy1.0不当200误差或资格，model1只有完整baselinegap，NF实际退出另看下文。三个NPZ4188249195/4293326239/4075599771bytes与所有原五报告和必要小样本全留，不作可清非例证。
完成真3完整NF0后才按所有ZIP/全200自己原shape估容量，factor4+6GiB/硬3-log1-pre.5-sample2GiB/outer10 guard不变，未知轴保守原最大比，不以空卡冒称预算通过。真足额→内部首次200测量批准→NEWfresh3/fixed0:199 NEW200，已有原3/NF不重复。完整200+原资格+三D5%与全部NN/NF/worker/controller/原EX退出才原writer11/ArtifactTool实际author+PNG实看/备份/双书真原EX-CAS/全targetaudit交付新186；控制器不写G，不预定合格。若NF或容量拒，原错误/五件/全部小样本值/source/config/log保留，不能翻旧job或用CPU诊断充正式恢复。按实际退出接其它前置齐款，不等定期。最新v8reader完整actual/进程/锁/实际自然退出优先，不按本段旧PID/state重启。

'''
text = text[:a] + base + text[b:]
a = text.index('【真实运行/停摆必须可见'); b = text.index('【LLaVA-Mini真3', a)
text = text[:a] + '''【真实运行/停摆与资源，末尾actual和更晚真实优先】
最近合格完整200仍Gemma4MLX8于11月22日03:34:24.274295北京时间全NF0/03:47:22.926705双书全审。Hancom11:57:44 full3/6通过后11:57:55容量拒，Base16:14:41真实前向恢复；Base16:56:31 fullNF失败后再次停摆，Turkish17:05加载不冒称NN，17:08左右真实vision-prefill/最迟17:15已4forward/17:19:08.896482最后第6forward，已可见中文报恢复，不继续跨恢复累计旧停摆。当前为同Turkish完整NF或已自然完成，以末尾actual/真实proc/原EX为准。诊断/元信息/前端/压缩/emptycard/ready/等候器不能充NN/NF，所有未知错查完整上下文，不泛用旧rootcwd/警告解释。原Base CPU诊断已经完成且无新增NPU，不充资格恢复。
最新完整reader UTC09:34原盘free22734032896bytes约21.17GiB，主要必要小样本/失败和已验数值矩阵，不是RAM或算力不足；机器16独立64GiB NPU/128CPU/~2TiBRAM，用户要求串行max1，0卡外部VLLM不动。当前小样本已全部落盘，不能因空间不足删除未验/失败/小样本/必要原例证。原400GiB扩容问题未答不是批准，不自行扩/换盘/tmpfs/降低guard/流式剪证据。本轮没有删除，旧精确pruned例外与ledger5f20ac...完整继承，不能再称其全数组仍可验。仅全数值验收+完整targetaudit+逐SHA/foreignFD/quiet真EX/全部保护前后核同的正确非例证按原policy精确清，报告/全部逐条记录/源码配置日志/小样本及错例证永保。
已有cleanup watcher32bd7209129b3cf22bac0056584847262e9793b3fab6b52dfaccc7b3de801401/真实config0bf674ae2fc971e34a020a1fbc8d724d72fe53645e44a8441e3e14557d7e6f29只完整raw相同物理hardlink去重，no_new_unique_array_pruning=True，不能宣称其已清唯一非例证。新未知error/停摆及时普通中文首行“严重阻塞｜正式评测未运行”；关键连接/检查失败“严重告警｜无法确认正式评测状态”。列真实模型阶段/样本/前向/卡/已证原因/最近完整及合格结果北京时间/最早可证时间持续时长（未知直说）/解除动作，恢复当轮也报；磁盘NOTIFY不是聊天通知。四项170/15/205/310、185交付和71完整资格必须分清，新增完整200=0要解释真3/NF与容量阶段，数值内不等于原资格。

''' + text[b:]
text = text.replace('本全文、closing_addendum.json', '本全文及同目录facts.json SHACURRENT_FACTS_SHA256、closing_addendum.json')
assert text.count('CURRENT_ACTUAL_JSON') == 1
primary = {}
for n in ['TurkishLlava_own_original_native_fullNF_after_Base_serial_v1_primary_plan_20261122.json',
          'TurkishLlava_exact_Base_visual_failure_ready_serial_v2_primary_plan_20261122.json',
          'GLM41Base_existing_visual_failure_CPU_diagnostic_primary_plan_20261122.json',
          'GLM41Base_Turkish_serial_readers_official_controls_v87_primary_plan_20261122.json']:
    primary.update(json.loads((V/n).read_bytes()))
extra_scripts = [
 'prepare_TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py',
 'prepare_TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122.py',
 'recover_TurkishLlava_completed_frontend200_current_unique_C690_metadata_only_v3_20261122.py',
 'inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py',
 'continue_TurkishLlava_own_frontend_v2_then_original_CPU_constructor_no_forward_v1_20261122.py',
 'inspect_TurkishLlava_own_original_auto_FP16_loader_resize_remap_no_forward_v2_current_unique_C690_20261122.py',
 'continue_TurkishLlava_completed_frontend200_metadata_v3_original_CPU_constructor_v2_no_NN_20261122.py']
builders = [
 'build_TurkishLlava_own_original_frontend200_meta_no_NN_v1_20261122.py',
 'build_TurkishLlava_own_original_frontend200_v2_exact_own_template_no_NN_20261122.py',
 'build_TurkishLlava_own_original_CPU_constructor_after_frontend_v2_no_forward_20261122.py',
 'build_TurkishLlava_completed_frontend200_unique_C_metadata_v3_CPU_constructor_v2_no_replay_20261122.py']
for n in extra_scripts + builders:
    b = (V/n).read_text(encoding='utf8').encode('utf8')
    ast.parse(b)
    primary[('scripts/code_publication_windows/' if n in builders else 'scripts/')+n] = sha(b)
stage = {}
for n in ['read_TurkishLlava_serial_v2_after_Base_failure_and_visual_CPU_diagnostic_actual_v7_20261122_remote.py',
          'read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py',
          'preserve_verified185_Base_failed_Turkish_same_serial_full_handoff_v87_20261122.py']:
    b = (V/n).read_text(encoding='utf8').encode('utf8')
    ast.parse(b)
    stage['scripts/maintenance/'+n] = b
stage['scripts/code_publication_windows/'+Path(__file__).name] = Path(__file__).read_text(encoding='utf8').encode('utf8')
for n, b in stage.items():
    primary[n] = sha(b)
original_names = [
 'R_TurkishLlava_own_original_frontend200_no_NN_v1_actual_launch_20261122.json',
 'R_TurkishLlava_frontend200_v2_exact_own_template_no_NN_actual_launch_20261122.json',
 'R_TurkishLlava_own_original_CPU_constructor_no_forward_actual_launch_20261122.json',
 'R_TurkishLlava_exact_unique_current_binding_completed_frontend_v2_20261122.json',
 'R_TurkishLlava_frontend_metadata_v3_CPU_constructor_v2_no_replay_actual_launch_20261122.json',
 'R_TurkishLlava_own_original_fullNF_after_GLM41Base_serial_v1_actual_launch_20261122.json',
 'R_TurkishLlava_ready_after_exact_Base_visual_NF_failure_serial_v2_actual_launch_20261122.json',
 'R_GLM41Base_existing_visual_CPU_diagnostic_only_actual_launch_20261122.json',
 'R_GLM41Base_Turkish_serial_full_numeric_progress_actual_v6_20261122.json',
 'R_TurkishLlava_serial_v2_Base_visual_CPU_diagnostic_actual_v7_followup_20261122.json',
 'official910b_before_verified185_GLM41Base_serial_v86_20261122.toml',
 'official910b_after_verified185_GLM41Base_serial_v86_20261122.toml',
 'official910b_verified185_GLM41Base_serial_v86_requested_args_20261122.json',
 'official910b_verified185_GLM41Base_serial_v86_official_tool_result_20261122.json',
 'official910b_verified185_GLM41Base_serial_v86_actual_readback_receipt_20261122.json']
local = []
for n in original_names:
    b = (V/n).read_bytes()
    local.append(dict(name=n, SHA256=sha(b), base64=base64.b64encode(b).decode()))
q = dict(expected_pointer_SHA256='d56937c6702fd5821d0846db892d4b882fc85ce60a2a2abcfbc5895a0db48f1e',
         expected_closing_SHA256='e0bfb7a53793d25aaf39ab45ec1384bb53e1addbcb05d5a83117f1fe542607cb',
         expected_pending_SHA256='d369ebbf961a9276fc7e70c2f49d88d5e99c1740e2dce2e4b4115a1c5f8af34d',
         round_name='verified185_Base_failed_Turkish_same_serial_full_actual_v87_20261122',
         pending_name='pending_after_daily_20261122_verified185_Base_failed_Turkish_same_serial_all_primary_v87_20261122.json',
         full_note_name='full_actual_verified185_Base_failed_Turkish_same_serial_v87_20261122.txt',
         new_primary_expected_SHA256=primary, local_original_records=local, full_handoff_template=text)
qb = json.dumps(q, ensure_ascii=False, indent=2).encode('utf8')
planname = 'verified185_Base_failed_Turkish_same_serial_full_handoff_plan_v87_20261122.json'
(V/planname).write_bytes(qb)
payload = {n:dict(SHA256=sha(b), base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier = "import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n;ast.parse(b)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nT=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans';q=T/"+repr(planname)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif q.exists():assert q.read_bytes()==b\nelse:q.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/preserve_verified185_Base_failed_Turkish_same_serial_full_handoff_v87_20261122.py'),str(q),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier)
(V/'stage_preserve_verified185_Base_failed_Turkish_same_serial_v87_20261122_remote.py').write_text(carrier, encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb), explicit_new_primary_paths=len(primary), local_original_records=len(local), finite_handoff_template_chars=len(text))))

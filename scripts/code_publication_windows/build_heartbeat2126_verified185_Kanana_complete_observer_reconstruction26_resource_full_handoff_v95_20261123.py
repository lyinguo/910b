"""Preserve finite current observer progress, real resources and all error records."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
s=(V/'preserve_heartbeat1627_verified185_Kanana_frontend200_NPU_operator_only_resource_full_handoff_v94_20261123.py').read_text(encoding='utf8')
changes={
 'de1380ab457195c04c2a94bb46f58210e1c7698a6c5d33505b833f99205e9c0d':'4a60aea2bd92f45d98c285eb6254655103a83b0bfa82d83fd344d5c6c198bcb9',
 '11c79eaea6011022f9b0d58dca60e33d5188542387c840bd556e338893b2885a':'7e654fa39e0497d9ceb5d43c2cf0a9f2f6a1cadc4394570c0962600b9489474b',
 'ab1480b3d6df58fda3f48e778ad4854dddfa96b4fb82d45d82429270246d109b':'c1eaa9fde8b98649ecf9d516b3911ff59cefff57f059ccf14db183d44f3cdae3',
 'len(rows)==3035':'len(rows)==3048',
 'verified185_heartbeat1627_Kanana_frontend200_NPU_operator_only_resource_full_actual_v94_20261123':'verified185_heartbeat2126_Kanana_complete_observer_reconstruction26_resource_full_actual_v95_20261123',
 'full_actual_verified185_Kanana_frontend200_NPU_operator_only_resource_v94_20261123.txt':'full_actual_verified185_Kanana_complete_observer_reconstruction26_resource_v95_20261123.txt',
 'pending_after_daily_20261123_verified185_Kanana_frontend200_NPU_operator_only_resource_all_primary_v94_20261123.json':'pending_after_daily_20261123_verified185_Kanana_complete_observer_reconstruction26_resource_all_primary_v95_20261123.json',
 'Actual Kanana independent original frontend200, exact input/source binding, untrained original RegStage NPU interface, preserved pre-stage failures and readonly maintenance controls; no learned model or canonical mutation':'Actual Kanana complete immutable native tensor/KV observation, separate full-value reconstruction, real synthetic/meta test consumers and current readonly resource/cleanup preservation; no learned model or canonical mutation',
 'Own independent original input200 accepted and six untrained NPU RegStage graph controls completed; no complete model adapter/fullNF/NEW NPU3 or score yet':'Own input200 and six untrained NPU RegStage controls previously complete; new native full-value observer 14 tests and independent reconstruction 12 tests passed, but model arithmetic verifier/NPU loader/YAML/NEW NPU3 still incomplete'
}
for a,b in changes.items():assert a in s,a;s=s.replace(a,b)
needle="    oldfacts=Path(ptr['current_full_facts']);assert sha(oldfacts)==ptr['current_full_facts_SHA256']"
insert="""    new_contracts={}
    for key,count in [('kanana_complete_observer_tests',14),('kanana_independent_reconstruction_tests',12)]:
        home=Path(q[key]['home']);rp=home/'review.json';assert sha(rp)==q[key]['review_SHA256'];rv=js(rp)
        assert rv['passed'] and rv['check_count']==count and len(rv['checks'])==count
        assert rv['learned_CPU_GPU_NPU_forwards']==0 and not rv['formal_NEW_NPU3'] and not rv['formal_fullNF_or_200_or_qualification']
        assert all(not n.startswith('/') and sha(R/n)==h for n,h in rv['source_sha256'].items())
        cp=home/'actual_completion.json';assert js(cp)['actual_returncode']==0 and js(cp)['no_model_NN_or_fullNF']
        new_contracts[key]=dict(review=str(rp),review_SHA256=sha(rp),complete_value=rv,actual_completion=str(cp),actual_completion_SHA256=sha(cp),actual_completion_value=js(cp))
    oldfacts=Path(ptr['current_full_facts']);assert sha(oldfacts)==ptr['current_full_facts_SHA256']"""
assert s.count(needle)==1;s=s.replace(needle,insert)
needle="    facts=note/'facts.json';put(facts,compact)"
insert="""    compact['current_Kanana_complete_native_observer_and_independent_reconstruction_contract_tests']=new_contracts
    compact['this_round_14_plus12_contract_checks_not_model_NPU3_or_fullNF_or_score']=True
    compact['transient_permission_and_local_sandbox_errors_exact_recovery']=q['transient_errors_and_recovery']
    facts=note/'facts.json';put(facts,compact)"""
assert s.count(needle)==1;s=s.replace(needle,insert)
needle="    for key in ['current_model_process_active'"
insert="""    upd['current_Kanana_complete_native_observer_and_independent_reconstruction_contract_tests']=new_contracts
    upd['transient_permission_and_local_sandbox_errors_exact_recovery']=q['transient_errors_and_recovery']
    for key in ['current_model_process_active'"""
assert s.count(needle)==1;s=s.replace(needle,insert)
mn='preserve_heartbeat2126_verified185_Kanana_complete_observer_reconstruction26_resource_full_handoff_v95_20261123.py'
ast.parse(s);(V/mn).write_text(s,encoding='utf8')
parent=json.loads((V/'heartbeat1627_verified185_Kanana_frontend200_NPU_operator_only_full_handoff_plan_v94_20261123.json').read_bytes())
norms=parent['full_handoff_template'];norms=norms[norms.index('【必须先读的完整继承】'):]
intro='''完整当前交接v95，原910b ACTIVE/原名称/原聊天/每300分钟/用户串行max1继续。必须完整读根H/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1/facts1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7，强制Day23v90全文62226a97a8e3a86e9e9b17c5f0916c5aa23d14ef3c3919d9727a92488ca25e21/factsb2b5a43e08e59a855f0ad23a05ee4c56e616becac1ff61c2c372099d055d6990，本全文/同目录facts SHACURRENT_FACTS_SHA256，再closing/late/pointer更晚实际。直接父v94全文faed6314853f2be95bc33ba21e03102998f102b8b30ffeb19b0332d6649fb655/facts1e391f7be7a26fd144b474d30635cabe8fe322557e5f0cac922e481f98f7f062/pendingc1eaa9fde8b98649ecf9d516b3911ff59cefff57f059ccf14db183d44f3cdae3，v93/v92/v91及全部祖先/65471字符archive全部强制规范、家族失败、用户决定和精确pruned例外完整继承，短引用不取消约束。完整有限当前交接加短引用，不无限拼接、不存密码；更晚真实proc/原EX/实际退出/closing优先，不按旧live或旧尚无观察器重跑完成阶段。
【当前仍严重存储阻塞，26项代码检查不是正式评测恢复】11月23日21:29本轮完整冻结monitor/proc argv-cwd-PPid-startticks/原FLOCK/cards确认无项目模型NN/fullNF/原EX，NPU1..15空/0外部VLLM不动，free21911277568bytes约20.4GiB；21:46重新资源/清理与末actual更晚优先。停摆最早可证仍11月22日17:40:49.156504北京时间Turkish200容量拒，按最新钟计时；本轮仅新观察器/独立格式重建的合成值CPU检查及真实meta挂钩注册，不含原模型权重加载或模型前向，不跨恢复/算子阶段误计。最近合格完整200仍Gemma4MLX8名下实际自己BF16 11月22日03:34:24.274295/两书03:47:22.926705，200/13383/NPU4/.840对三D.8350/.8350/.835均0.59880239520958%；最近小NF仍Turkish17:40:38真3/6，200约3.85TiB拒；必要全3/12.56GB保持，均不重放。
本轮新完整NPU200/G/完整资格/真实模型样本及提交前向/fullNF均0。唯一185交付/170数值内15超差205完整权重待正式交付310确缺/71完整原协议资格不变，170=71完整+99仅数值，390完整权重仍319需适配/资格/修复。两书2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878/state47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3/reg846e7e4c02f91cb900f6856b20d1e6a91c69fe588c758d7df690492981f1da8f/bindingd25d0d3f284a601117ab75a10978d35f1b992236fdc91ed692296b859426c39c均不写。原D700956.../当前D14ab07...保护非G/公式/样式/行序/ZIPmeta/外变；changes0不写，186前freshlocalSHA/stat/700模型真实计数/29精确round9/310确缺空G核，禁旧导出覆盖185。
BeeSFT v5原无NN等候器仍每20秒核原盘192555984252bytes约179.33GiB与原source/canonical/全NN-NF-controller-EX门槛，真过才新私有内部首次200测量批准→NEWfresh3/fixed200，不持EX、不复制/热改/信号/重复自己的原3/storageNF。旧1668102无子无EX和cleanupwatcher1629242不动；后者仅完整raw相同物理硬链接无新增unique prune。扩原盘至少200GiB问题未答不是批准或已扩，不擅扩/换盘/降guard/流式剪证据/删未验；原main363384用户停/PAUSE原owner不重启接管，不scratchstate/清G骗保护，0卡他人VLLM不动。串行max1，前置资源真齐锁释放立即接不等周期。对其余前置未闭合款继续源码适配，不能笼统说205都可马上跑200，也不能用仅等候/空卡/算子/前端或完成小样本充live。
【Kanana前已完成与本轮新代码边界】MID kakaocorp/kanana-1.5-v-3b-instruct当前C唯一row379/三D.8900/.8900/.89/G空，旧state348/原snapshot315/保存YAML baseline.88不能作当前行或D。自己的2SF7335888032bytes/739BF16头、32Llama Q32/KV8/head_dim128/32Qwen2视觉/H1280/16heads/128259全头、自己的DynamicCAbstractor原RegStage depth2x2/PatchMerge2/SiLU/LayerNorm2d/原pos插值、theta8000000/原配置cacheFalse不变；自己5py仅既有3timm导入为真实原私有算子外所有模型非import AST同，原AFS/共享库不写，不装包或stub，不借checkpoint/数据/词表/字体。旧739实际meta已通过，旧跨AA Torch2.11与R CPU2.10六组合成逐位控制仍失败，未翻0或放阈值；原AA graph是原source合成CPU控制不是历史A100模型现场。
前端200已于父v94完整独立通过，Hfront=B/Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123，reviewdb892b6f4e4948825d55d5d75208454a7360e56639b7c0fdda3acc8b483d922f/frontend2fdff83189f3ed0ae6e38db4744c5a66fa4ed2cee11b3e2543317cbe6de0c585。原R和AFS4329数据同SHA2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451，原专用conv/双Human段/原Jinja空白/BPE既有修复/F32像素每位/全部token-mask-grid/负1图像位/自己的EOS128009-pad128001/Greedy2048/outer1/max_input8192保持。0/99/199必要完整NPZ、原v1对话参考失败、v2最小独立参考修复和v3来源路径失败全留，不重跑CPU200。只读来源scope v4 Haccept=B/Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v4_exact_pipeline_20261123/review1dddc30c79bf076871546c6ee1e21da978f15600684a0f713a51eb61f30db755已0，R相对source含YAML/verifier/run.py/真实universal_eval/orchestrator/pipeline.py，AFS/库/证据另external，旧来源字段不翻改。
父六组无训练权重原RegStage图实际NPU4测试已16:43:09完成diagnostic0，reviewfeb6efa05c572a0aeb6a7ee424977e320e3b79feacd9e11c3aa83316ea5a0dcb，58个合成参数actualNPU全SHA同，6组fullraw值/差异全留，相对AA CPU .001842..004482、非逐位同；没有加载checkpoint或原模型前向/fullNF，不能再当新NPU3、资格或恢复，不重放此算子测试。
【本轮完整张量/KV观察器14项检查实际0】新增R/universal_eval/model/kanana_own_original_graphtext_v2_20261123/complete_native_observer_values_20261123.py SHA d05a1460fe83778a9ab32dc328f5fb6f805a034ddc5411da36eb294ed2b8f87e，仅register_forward_pre_hook/post_hook读取原native模块，不替forward/generate/prepare/cache update/math/dtype/预算/停止/输入/提取。每个tensor occurrence都独立完整detach-copy到CPU并保存dtype/shape/所有raw位SHA；只有全dtype/shape/rawSHA与全部raw位实际再比相等才共享一个完整标准NPY，所有语义引用/发生次数保持。未抽样/截断/量化/仅存索引或argmax/选择头，完整标准FortranNPY+ZIP-DEFLATE写入；每完整数组全量重开逐位同。3GiB硬/1GiB日志写入保护保持，本模块明确不替代未来模型2GiB每sample/.5GiB预检/factor4+6GiB以及真实小样本和200全部形状容量门槛，不凭此改原预算。
真实消费者scripts/tests/test_kanana_complete_native_observer_values_no_model_20261123.py SHA1e5edf397b0d03a41c8d928351646d6ac8bd7bf79ae3c3d94342fd2adf1093c8，Htest=B/Kanana_complete_native_observer_synthetic_values_and_real_meta_registration_tests_v1_20261123于11月23日21:38:40.253020北京时间actual0，review3ea092c400e335679619c6d951f327a30d62a8bb4933a214b7129da76d76a181。14项实查包含BF16所有NaN payload/Inf/正负零和finite原位、同bytes不同dtype/shape不混、输入后来原地变更不改已存值、标量rank0/空shape(0,8)/非连续view、完整128259列头、真实HF DynamicCache的32层64完整K/V和append前后、未知对象明确拒、全部NPY独立重开、合成未训练nn.Linear有无hook输出逐位同、移除只自有hooks而保留原hooks；对真正自己的739 meta模型816原模块注册，32视觉/32Llama/projector/fullhead齐，所有739 state和每个原forward/generate/prepare函数对象前后同。meta重构只用于新观察器真实拓扑挂钩检查，复用原739来源证明，不算又一次模型验证；未加载checkpoint、原模型forward0、仅2次未训练CPU Linear操作，不当learned CPU/NPU/AA成绩。全部合成两组完整NPZ/manifest/events均保留。
【独立读取/完整32KV重建12项实际0，不是模型数值fullNF】新增R/universal_eval/model/kanana_own_original_graphtext_v2_20261123/independent_complete_capture_values_20261123.py SHA6029407eeef7ad73aaf0c04cb2787dbc3702f6a094d3a919651610941b2254be，不import生产journal，独立按标准NPZ/NPY和完整SHA/dtype/shape/raw位解码所有数组及每个语义occurrence，检查没有漏值/orphan/分母/顺序变化或伪dtype-shape。真实消费者scripts/tests/test_kanana_independent_complete_capture_reconstruction_no_model_20261123.py SHA5c8df97d5f64542229921df0b6d89aaa86bdcf3140832332a2227731fe4de493，H=B/Kanana_independent_complete_capture_and_full32KV_reconstruction_synthetic_only_v1_20261123于21:43:38.209950实际0，review91dba8a2df8d919d6f0f559e2bc4341fc0607b43dac130d3735ab35d9d42ef47。12项独立检查恢复sample0全部135个unique完整array/136 occurrence和sample1全部2/2；BF16全位、dtype/shape分开、每个引用独立copy、标量/空/strided/全头、全部32层K/V完整prefix与append逐位同、后来livecache变更不改旧prefix、伪造dtype/shape拒、原pre/post事件与call配对齐。没有重放14测试/前端200/meta或原模型，原值/错误/分母不减，不因12检查0称模型fullNF通过。
【必须保留的原cache事实】本轮读取本机原GenerationMixin.generate完整源码到Htest/installed_original_GenerationMixin_generate_external_source.txt，原分支明确decoder-only且model_input_name==inputs_embeds时generation_config.use_cache=True；Kanana自己原generate确以prepare_mm_inputs返回inputs_embeds进入原Llama generate。因此不能仅凭保存config.use_cacheFalse就省略32层KV、强关cache或声称原A100同样没cache。原配置与原source都没改，这只是原installed生成入口语义，真实新NPU运行仍必须实际捕获确认cache类型/物理全值/前缀/位置mask/append/全部源SHA。下一步完整逐层数学重建、自己的739source-cast-actualNPU加载、原全头/视觉/kernel/真实cache独验、单YAML/IModel和validate_config/check_sources/config3/200及资源门槛还未闭合；没有适配ready/NEW真3/内部首次200批准/fresh3/200/G。以上两套格式/观察契约检查26项不能当原模型全数值验收，不能降低旧任何模型阈值或重跑已过阶段凑运行。
【本轮资源、清理和错误处理】21:46已审185已全验正确非例证+临时发现器仅输出目录变化/AST同重查，候选/bytes/tmp以末facts为准；原13legacy完整numeric的5+8定向meta复审继承，不因original资格字段缺误称没验，不重放原precisely pruned数组。三原根同一冻结只读门槛本轮又核，暂不可读跳过/保留G、不加310、不清selector27daea...或合法发布引起protectedstate。本轮0删除，不动任何权重/数据/表、原小样本/失败/未验/错invalidtrunc/error、必要首中末整组/五件/全部逐条明细/source/YAML快照/日志。原ledger5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0/policy66857.../consent06eba...保持，全部祖先精确pruned例外原proof字节不删不称仍全数组。物理extent dedupe errno95不支持/未采用保持不重复压缩或试新盘。
首次当前实际检查因自动权限审核超时被工具拒且没有执行，明确系统原文“不因timeout推断unsafe，允许retry once”；已按允许的一次同固定只读SSH重试成功21:29完整核状态，没有重复模型或杀进程。另一次仅本地V生成脚本用默认受限令牌因split writable roots能力不支持而未执行，后同范围经审核执行成功。两个错误原观察与恢复文件均留，无用户拒绝授权/模型故障/旧job翻0，不泛白名单；后新连接/monitor真实成功优先，当前能确认正式仍未运行。父旧辅助SSH255自然reset与原R控制actual1已闭合，不再当活NN/NF或当前连接故障。
【日更与全部原件】11月23日已唯一正常push5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9/Code-Day23/2972paths303changes/remote同SHA，R39/本地19/MJS25及最初R入口失败/v2修复回执保留；今天不第二commit。publisher code_only唯一已审1585e1dce7edea8a7b3c99bf27ae4b2840b715ac3b054542d55141bbc44906e9/publisher e67fc049...不变，两个Meta含inline200数据protocol仅Git精确排除，R原件和生成器/YAML在。全部CURRENT_SOURCE_COUNT主源与真实consumer配置/原IR逐SHA/字节完整pending，包含本轮4个新观察/独立重建模块及测试、清理发现器、全部本地原回执和有用维护源。Day24首轮审核用途/consumer/deps/fresh并发SHA/全diff-alloutgoing/对应checks/真实publish_code_daily.py消费与审核planSHA后一次正常push核远端，无变不空提交/失败续原已审commit/不等提交才跑。唯一GitHub https://github.com/lyinguo/910b.git/独立codex/910b-source-layout-20261114/D:/code-lv/910b本轮21:46仍clean，不merge/force/Gitee历史/R master-remote改动，不gitadd.扫workspace或上传权重数据/任何合成或模型数组/报告日志keys.env凭据归档/字面base64运输payload。工作实现只R源码/scripts/configs，workspace仅证据，后续提交文件不可丢。
原910b本轮只读TOML590801a94c1493e88f304f20f35bf5f31fd3a77e5548f346c735179ca0c52929，ACTIVE/原名称/原线程/300分钟同，无official更新或raw写。local book fresh2981.../6099931bytes/mtime1795290387531571700/ctime1795290387529570400同，全部canonical/原runner58de6.../Gwriter83134.../PAUSE/POLICY/真实run.py和orchestrator/pipeline不写。其它所有家族与原用户决定按父完整继承：MedraN原FP16 nonfinite不借当前AA不同BF16配置改精度/eps数学，GLM41Base原fullNF1未过不放阈值，Moondream原query768采样reasoning/PixArialVQA4096/Florence自己头数据dtype/StructTable1500/TrOCR词表/LoRA/量化与确缺规则不变；所有已交付款/原已过3不重复。
严重停摆/未恢复异常首行“严重阻塞｜正式评测未运行”，关键当前连接检查失败“严重告警｜无法确认正式评测状态”；每轮报真实模型阶段/样本前向/卡/原因/最近完整合格结果北京时间/最早可证时长和解除动作，真实恢复当轮也报，NOTIFY磁盘文件不是聊天通知。空卡/加载/算子/CPU前端/观察器与meta契约检查/等待进程不充正式NN/fullNF。全部有资源款真实原协议910B200/fullNF/三D5%、确缺明确跳过、无pending/资格复审且全量复查双书完整报告历史例外才关闭，远未完成。

'''
template=intro+norms;assert template.count('CURRENT_ACTUAL_JSON')==1
primary={}
for n in ['Kanana_complete_native_observer_primary_source_SHA_heartbeat2126_20261123.json','Kanana_independent_complete_capture_primary_source_SHA_heartbeat2126_20261123.json','heartbeat2126_verified185_cleanup_discovery_primary_SHA_20261123.json']:
    for p,h in json.loads((V/n).read_bytes()).items():
        if p in primary:assert primary[p]==h
        primary[p]=h
stage={'scripts/maintenance/'+mn:s.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes()}
for n,b in stage.items():ast.parse(b);primary[n]=sha(b)
names=['R_heartbeat2126_required_root_v90_latest_v94_actual_20261123.json','R_heartbeat2126_required_root_v90_latest_v94_actual_20261123.json.stderr',
 'R_Kanana_complete_native_observer_synthetic_tests_heartbeat2126_20261123.json','R_Kanana_complete_native_observer_synthetic_tests_heartbeat2126_20261123.json.stderr',
 'R_Kanana_independent_complete_capture_tests_heartbeat2126_20261123.json','R_Kanana_independent_complete_capture_tests_heartbeat2126_20261123.json.stderr',
 'R_heartbeat2126_verified185_current_safe_cleanup_metadata_discovery_20261123.json','R_heartbeat2126_verified185_current_safe_cleanup_metadata_discovery_20261123.json.stderr',
 'R_heartbeat2126_original_three_resources_frozen_readonly_current_20261123.json','R_heartbeat2126_original_three_resources_frozen_readonly_current_20261123.json.stderr',
 'heartbeat2126_fresh_local_book_TOML_code_readonly_20261123.json','heartbeat2126_permission_timeout_and_local_sandbox_capability_errors_observed_recovered_20261123.json',
 'R_user_other_models_and_cleanup_current_required_v94_actual_20261123.json','R_user_other_models_and_cleanup_current_required_v94_actual_20261123.json.stderr',
 'R_user_other_models_original_three_resources_current_20261123.json','R_user_other_models_original_three_resources_current_20261123.json.stderr']
local=[]
for n in names:
    b=(V/n).read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
q=dict(parent,new_primary_expected_SHA256=primary,local_original_records=local,full_handoff_template=template,
 current_round_local_and_automation=json.loads((V/'heartbeat2126_fresh_local_book_TOML_code_readonly_20261123.json').read_bytes()),
 fresh_cleanup_discovery=json.loads((V/'R_heartbeat2126_verified185_current_safe_cleanup_metadata_discovery_20261123.json').read_bytes()),
 current_original_three_resources_readonly=json.loads((V/'R_heartbeat2126_original_three_resources_frozen_readonly_current_20261123.json').read_bytes()),
 kanana_complete_observer_tests=json.loads((V/'R_Kanana_complete_native_observer_synthetic_tests_heartbeat2126_20261123.json').read_bytes()),
 kanana_independent_reconstruction_tests=json.loads((V/'R_Kanana_independent_complete_capture_tests_heartbeat2126_20261123.json').read_bytes()),
 transient_errors_and_recovery=json.loads((V/'heartbeat2126_permission_timeout_and_local_sandbox_capability_errors_observed_recovered_20261123.json').read_bytes()))
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();pn='heartbeat2126_verified185_Kanana_complete_observer_reconstruction26_full_handoff_plan_v95_20261123.json';(V/pn).write_bytes(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,os,subprocess\nR=Path('/root/projects/comperror');payload=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,z in payload.items():\n b=base64.b64decode(z['base64']);assert hashlib.sha256(b).hexdigest()==z['SHA256'];ast.parse(b);p=R/n;p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(pn)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+mn+"'),str(p),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_heartbeat2126_verified185_Kanana_observer_reconstruction_full_handoff_v95_20261123_remote.py').write_text(carrier,encoding='utf8')
(V/'heartbeat2126_Kanana_all_new_primary_before_full_handoff_v95_20261123.json').write_text(json.dumps(primary,indent=2),encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb),new_primary_count=len(primary),local_original_count=len(local),finite_full_template_chars=len(template),no_new_model_NPU_or_score_claim=True)))

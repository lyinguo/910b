"""Preserve finite new input/operator progress and all exact failures, no model claim."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
s=(V/'preserve_heartbeat1131_verified185_Kanana_graph_meta_drift_resource_full_handoff_v93_20261123.py').read_text(encoding='utf8')
changes={
 '17c7404eccf2784ef9c7647e3e791116170abb5c4b212b00ed9198ccbe6e20f1':'de1380ab457195c04c2a94bb46f58210e1c7698a6c5d33505b833f99205e9c0d',
 'd3c900e5833326cded6a65c27f6031e7477bc44af1de07662863982a09075a79':'11c79eaea6011022f9b0d58dca60e33d5188542387c840bd556e338893b2885a',
 'f3dc4620f1d8417c003003298e2cfd13464a20922175df4e1ddc1135d66f0b8d':'ab1480b3d6df58fda3f48e778ad4854dddfa96b4fb82d45d82429270246d109b',
 'len(rows)==2987':'len(rows)==3035',
 'verified185_heartbeat1131_Kanana_graph_meta_CPU_drift_resource_full_actual_v93_20261123':'verified185_heartbeat1627_Kanana_frontend200_NPU_operator_only_resource_full_actual_v94_20261123',
 'full_actual_verified185_Kanana_graph_meta_CPU_drift_resource_v93_20261123.txt':'full_actual_verified185_Kanana_frontend200_NPU_operator_only_resource_v94_20261123.txt',
 'pending_after_daily_20261123_verified185_Kanana_graph_meta_CPU_drift_resource_all_primary_v93_20261123.json':'pending_after_daily_20261123_verified185_Kanana_frontend200_NPU_operator_only_resource_all_primary_v94_20261123.json',
 'Actual Kanana private original graph interfaces, strict synthetic control failures, meta and filesystem diagnostics, readonly resource checks and full preservation controls; no learned model or canonical mutation':'Actual Kanana independent original frontend200, exact input/source binding, untrained original RegStage NPU interface, preserved pre-stage failures and readonly maintenance controls; no learned model or canonical mutation'
}
for a,b in changes.items():assert a in s,a;s=s.replace(a,b)
# The original 58.9 MB AA stdout was already archived byte-identically in v93.
# Reference its existing full control plan instead of duplicating it this round.
a=s.index('    # Reconstruct the byte-identical original AA stdout')
b=s.index("    for n,h in q['new_primary_expected_SHA256'].items():",a)
s=s[:a]+s[b:]
needle="    oldfacts=Path(ptr['current_full_facts']);assert sha(oldfacts)==ptr['current_full_facts_SHA256']"
insert="""    front=Path(q['kanana_frontend_acceptance']['home']);fp=front/'review.json'
    assert sha(fp)==q['kanana_frontend_acceptance']['review_SHA256'];fv=js(fp)
    assert fv['passed'] and fv['samples']==200 and fv['actual_NEW_model_forwards']==fv['actual_NEW_frontend_computations']==0
    assert not fv['actual_NPU_inference'] and not fv['formal_fullNF_or_200_or_qualification']
    assert fv['current_C_unique_row']==379 and fv['current_three_A100_D']==dict(server='0.8900',local='0.8900',preserved_original='0.89')
    assert all(not n.startswith('/') and sha(R/n)==h for n,h in fv['source_sha256'].items())
    op=Path(q['kanana_untrained_NPU_operator']['home']);opproof=op/'review.json'
    assert sha(opproof)==q['kanana_untrained_NPU_operator']['review_SHA256'];ov=js(opproof)
    assert ov['passed'] and ov['all6_original_graph_NPU_executions_completed'] and ov['all58_untrained_parameter_source_actual_NPU_SHA_verified']
    assert ov['checkpoint_values_loaded']==ov['learned_CPU_GPU_NPU_forwards']==ov['actual_NEW_model_forwards']==0
    assert not ov['formal_NEW_NPU3'] and not ov['formal_fullNF_or_200_or_qualification']
    assert len(ov['controls'])==6 and len(ov['synthetic_parameter_sources'])==58
    current_closed={}
    for home in ['Kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123','Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123','Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v3_20261123','Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v4_exact_pipeline_20261123','Kanana_original_parameterized_RegStage_actual_NPU_synthetic_interface_diagnostic_v4_20261123']:
        p=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'/home/'actual_completion.json'
        current_closed[str(p)]=dict(SHA256=sha(p),complete_value=js(p))
    assert [z['complete_value']['actual_returncode'] for z in current_closed.values()]==[1,0,1,0,0]
    oldfacts=Path(ptr['current_full_facts']);assert sha(oldfacts)==ptr['current_full_facts_SHA256']"""
assert s.count(needle)==1;s=s.replace(needle,insert)
needle="    facts=note/'facts.json';put(facts,compact)"
insert="""    compact['current_Kanana_completed_frontend200_input_only_acceptance']=dict(review=str(fp),review_SHA256=sha(fp),complete_value=fv)
    compact['current_Kanana_untrained_NPU_operator_interface_only']=dict(review=str(opproof),review_SHA256=sha(opproof),complete_value=ov)
    compact['current_Kanana_frontend_source_scope_operator_actual_completions']=current_closed
    compact['old_local_control_transport_naturally_closed255_remote_actual1']=q['old_local_transport_actual']
    compact['no_new_model_NPU3_fullNF_or200_despite_input200_and_untrained_NPU_operators']=True
    facts=note/'facts.json';put(facts,compact)"""
assert s.count(needle)==1;s=s.replace(needle,insert)
needle="    for key in ['current_model_process_active'"
insert="""    upd['current_Kanana_completed_frontend200_input_only_acceptance']=compact['current_Kanana_completed_frontend200_input_only_acceptance']
    upd['current_Kanana_untrained_NPU_operator_interface_only']=compact['current_Kanana_untrained_NPU_operator_interface_only']
    upd['current_owned_candidate']='kakaocorp/kanana-1.5-v-3b-instruct'
    upd['current_owned_candidate_stage']='Own independent original input200 accepted and six untrained NPU RegStage graph controls completed; no complete model adapter/fullNF/NEW NPU3 or score yet'
    upd['old_local_control_transport_naturally_closed255_remote_actual1']=q['old_local_transport_actual']
    for key in ['current_model_process_active'"""
assert s.count(needle)==1;s=s.replace(needle,insert)
mn='preserve_heartbeat1627_verified185_Kanana_frontend200_NPU_operator_only_resource_full_handoff_v94_20261123.py'
ast.parse(s);(V/mn).write_text(s,encoding='utf8')
parent=json.loads((V/'heartbeat1131_verified185_Kanana_graph_meta_CPU_drift_resource_full_handoff_plan_v93_20261123.json').read_bytes())
norms=parent['full_handoff_template'];norms=norms[norms.index('【必须先读的完整继承】'):]
intro='''完整当前交接v94，原910b ACTIVE/原名称/原聊天/每300分钟/用户串行max1继续。必须先完整读根H/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1/facts1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7、强制Day23完整v90全文62226a97a8e3a86e9e9b17c5f0916c5aa23d14ef3c3919d9727a92488ca25e21/factsb2b5a43e08e59a855f0ad23a05ee4c56e616becac1ff61c2c372099d055d6990，再本全文/同目录facts SHACURRENT_FACTS_SHA256、closing/late/pointer更晚真实实际。直接父v93全文76edb482c9264e8f3bd4d2b2731343bdad5d270c48584f8cc261096e7fa302e0/facts1b925b790a575cb1fd6a90efa5526fe12c831a528f467c59747d6e64e24b0d4a/pendingab1480b3d6df58fda3f48e778ad4854dddfa96b4fb82d45d82429270246d109b，v92/v91及全部祖先/65471字符archive全部强制规范/家族失败/精确清理例外/用户决定完整继承，短引用不取消约束，不无限拼接/保存密码。更晚实际proc/原EX/真实退出/closing优先，不按旧live或旧“尚无前端200”重启已完成阶段。
【当前仍严重阻塞，正式模型NN/fullNF没有恢复】本轮11月23日16:28冻结完整monitor/实际proc/argv/cwd/PPid/startticks/FLOCK/cards确认项目模型NN/fullNF/原EX皆无、NPU1..15空/0外部VLLM不动、盘约20.5GiB；本段后最终actual更晚优先。16:43仅原全局EX下NPU4无训练权重算子诊断，绝非正式恢复，阶段完成后锁自然释；16:49原三根只读gate仍无NN/NF/EX。停摆最早可证仍11月22日17:40:49.156504北京时间Turkish200容量拒，按最新钟计时，不把算子/CPU输入/源码或meta当NN/fullNF。最近合格完整200仍Gemma4MLX8名下实际自己BF16 11月22日03:34:24.274295/双书03:47:22.926705，200/13383/NPU4/.840对三D.8350/.8350/.835误差均0.59880239520958%，不重放。最近完整小NF仍Turkish17:40:38真3/6/200约3.85TiB拒；全3/12.56GB必要数值保护。
本轮新完整NPU200/G/完整原资格/真实模型样本/提交前向/fullNF均0。唯一185交付/170数值内15超差205完整权重待交付310确缺/71完整原资格保持；170=71完整+99仅数值，390完整权重仍319需补适配/资格/修复。空G失败超差测量不伪增15或185。双书SHA2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878/state47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3/reg846e7e4c02f91cb900f6856b20d1e6a91c69fe588c758d7df690492981f1da8f/bindingd25d0d3f284a601117ab75a10978d35f1b992236fdc91ed692296b859426c39c均不写。原D700956.../当前D14ab07...不改，完整非G/公式/样式/行序/ZIPmeta/外变保护，changes0不写，新186必须freshlocalSHA/stat/真实700模型计数/29精确round9/310缺空G核，不旧导出覆盖185。
BeeSFT v5原无NN等候器仍每20秒核原盘192555984252bytes约179.33GiB及完整source/canonical/全部NN-NF-controller-EX门槛，真满足才新私有内部首次200测量批准→NEWfresh3/固定200；当前20.5GiB不够，不复制/热改/信号/用waiting当模型，不重复自己的原3/storageNF。旧1668102无子无EX不动，cleanupwatcher1629242只全raw相同物理硬链接/no unique prune。扩原盘至少200GiB问题仍未答，未答不批准/已扩，不擅扩/换盘/降guard/流式剪证据/删未验。原main363384用户停/PAUSE原owner保持，不重启接管或scratchstate/清G骗保护，0卡他人服务不动；锁释放与原前置/容量真过即接续不等周期。
【Kanana本轮自己的完整输入200已独立通过，三D精确绑定已纠正；不重跑前端】MID kakaocorp/kanana-1.5-v-3b-instruct实际首sheet C唯一row379/三D server.8900/local.8900/保留原.89/G空。旧state row348/保存YAML baseline.88及原snapshot row315是历史，不能当当前D或行。自己的原YAML4eb9bb0dc87e8fd6ff212e1cc7fa7e0ccf8c5591fa878eab476cac2809cc5121/HighNineKananaCanonical91f6ea468dea13f79cc1f94fe10f15f3a9f7a6b5ffd3276c7599eda2dfa47fd8和base22286203e26940a7f6a97da841b39c285b0d9a5f070ffb5352e9d9100d4467db保持；原BF16-auto/Greedy2048/outer1/专用Processor conv+image/max_length8192/BPE merges既有修复/new-tokens-only不二次切片/视觉SDPA均不变。实际自己的KananaVTokenizerFast构造并恢复原Rust BPE后eos128009/pad128001/bos128000，beforeprobe字符IDs到afterprobe[791,4062,14198,39935,35308,927,279,16053,5679,13]只恢复原已有canonical修复，不新借词表。保存原AFS dataset与自己的R相同200全部4329原行，两个完整文件均SHA2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451且本轮AFS原路径可读，未借AA或另一款数据/像素。
新prepare_kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123.py先本地预检因sharedStrings不存在、空v.text=None映射拒0NN，原观察/失败builder全留；v2 local builder只处理真实str-v空G和独立三D绑定，未写表。服务器输入v1第0条像素全位同，但独立对话参考误假设BOS后换行/图文无换行/assistant前单换行而拒actual1，原原始对话差异/完整trace/源码全留不翻0。新prepare_kanana_own_original_BF16_dedicated_frontend200_v2_original_whitespace_20261123.py SHA b84c16628b284ac30d6d7a96a0720c2d12afa486c8a85888033c8cae7b5052a2只独立reference文本恢复原Jinja实际空白（BOS直接intro、图与文Human段间换行、AI前双换行），所有其他函数AST完全同，自己的processor/chat_template/model math未改。
Hfront=B/Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123真实完整200通过，reviewdb892b6f4e4948825d55d5d75208454a7360e56639b7c0fdda3acc8b483d922f，own_original_frontend200.jsonl SHA2fdff83189f3ed0ae6e38db4744c5a66fa4ed2cee11b3e2543317cbe6de0c585。全部200原prompt/ref/imageSHA/原Greedy尾句/专用完整对话/全部input_ids/attention_mask/负-1图像token/每个F32pixel bitSHA/原grid及顺序独立全值比较同；自己的PIL uint8BICUBIC/原HF float64-rescale→F32再F32norm/原两帧复制和(通道,时序,14,14)patch轴序不改。0/99/199三组完整RGB/实际与独立F32像素/token/mask/grid必要NPZ均在，不重放CPU200。
只读输入来源绑定v3误写universal_eval/pipeline.py，在读源路径时拒actual1，未重做输入或改原成功/失败。新scripts/maintenance/accept_Kanana_completed_frontend200_exact_source_scope_no_replay_v4_real_pipeline_20261123.py SHA27be8a5263812c3486be570c83dfae10dbfd66cf10b121b8c73f35cb4ebe6976仅修真实universal_eval/orchestrator/pipeline.py，Haccept=B/Kanana_completed_original_frontend200_source_external_scope_acceptance_no_replay_v4_exact_pipeline_20261123实际0，准确reviewSHA见末facts。本款frontend200原件与三必要完整例证SHA复核、全部R source相对路径含原YAML/两frontend/verifier/run.py/真实pipeline，AFS/库/证据另external；原v2自述绝对source字段原件保留，未来正式protocol消费v4正确schema而非翻旧job。actual_NEW_frontend_computations=0/actual_NEW_model_forwards=0/actual_NPU_inference=False/formal_fullNF_or_200_or_qualification=False，passed只输入来源闭合，非模型资格。
【真实NPU原RegStage接口六组完成，但不是模型真3或fullNF】自己的原739BF16头/2SF7335888032bytes/源5py、自己32Llama/32Qwen2视觉/128259头、theta8000000/config cacheFalse保持。父v93已经739实际meta name/shape/BF16同；原AA Torch2.11与R CPU2.10的六组合成逐位控制仍失败，.0001648..003624 relativeRMSE及旧parse/inline/rawbit失败全部留，未将它翻0或推断原A100现场。
新scripts/diagnostics/diagnose_Kanana_original_parameterized_RegStage_actual_NPU_synthetic_interface_v4_20261123.py SHAe26bd810333b2132242b73c32bf921bb951f1644d151c90fac8f4926a78ad4cf，于11月23日16:43:09.053240北京时间diagnostic0。H=B/Kanana_original_parameterized_RegStage_actual_NPU_synthetic_interface_diagnostic_v4_20261123，reviewfeb6efa05c572a0aeb6a7ee424977e320e3b79feacd9e11c3aa83316ea5a0dcb。原真runner.lock EX/实际/proc/空NPU4/npu-smi准入下，只执行自己原两个RegStage完整无权重JIT图，使用父已完整保存的42348160bytes随机未训练合成参数（不是checkpoint）；58个name/shape/BF16/source→actualNPU完整rawSHA全部同。输入形状每stage为4x6/6x4/8x8共6组，原AA CPU与R原图CPU和实际NPU全输入/输出NPZ及SHA完整留。
六组实际NPU相对原AA CPU的relativeRMSE .0039996481/.0018420373/.0041006525/.00448216997/.0032094388/.0036756871，maxabs .046875..078125，均非逐位同；只报告完整差异，不建立/放宽模型接收阈值。此0仅接口执行成功、全finite/shape/dtype和58合成源SHA过；checkpoint_values_loaded=0/learned_CPU_GPU_NPU_forwards=0/actual_NEW_model_forwards=0/formal_NEW_NPU3=False/formal_fullNF_or_200_or_qualification=False。没有模型权重加载/模型样本/真实模型视觉kernel/fullhead/32KV验证，不拿这个NPU算子0或CPU200当正式恢复。未安装timm/包、无stub、原AFS/共享Torch/自己5py除既有私有import外不变，无模型dtype/math/eps/mask/position/cache/预算/停止/提取调优。
下一步Kanana必须自己的完整native IModel单YAML/只读全模型观察器、自己的739 source-cast-actualNPU SHA、完整头/原视觉/原cache与全部数值独验闭合，validate_config/check_sources/config3/200及真实资源门槛真过再NEW真3；当前未ready/无首次200批准/fresh3/200/G。保留已过输入200和原meta及六算子结果，不重复凑运行，也不等AA在线作为额外前置。原engine现场未采如实披露，CPU/MLU不充A100/G。
【清理及错误处置真实结果】16:39已审185当前已全验正确非例证/临时发现器新H独立复查0candidate/0bytes/temporary[]，discovery98a58ef14d117dd17523bdd4a093e2f9be1c963540c9fdd4c77dc20109c65432，原13legacy完整numeric 5+8定向metadata复审结果继承，不仅因缺original资格字段误称没验，也不重放已精确pruned数组。本轮0删除，未对旧权重/数据/表/失败/未验/活动/小样本/必要完整首中末组/五件/逐条明细/source/config/log动手。旧原盘小合成文件extent dedupe errno95不支持/no adopted方法不重试；未换盘/扩容/降guard/流式剪证据。原ledger5f20ac2b35d58b09e5c4a3a8ca6a5dacf1446b9cb39876678b24458b0f1ef4b0不改，所有祖先精确pruned不能再全数组验，原proof字节保持不能称仍全矩阵。
原Nemotron/Cognitive/Calm三根16:49冻结只读门槛仍temporarily unavailable/dispatchnull，不加永久310、不清selector27daea...或合法发布造成protectedstate，不借其它同名权重。父旧本地session32886/Python35140-ssh28384传输本轮自然退出：SSH255/client_loop Connection reset，helper1，stdout0，完整stderr保留。服务器原privatev1早于11:54:30已actual1/a7cc316...，从来不是模型NN/NF；本轮没有信号/误杀/重启，后续多次新固定SSH/完整monitor均正常，不能把旧辅助255当当前模型退出或当前服务器不可确认。原来源v3路径失败和前端v1参考失败各自明确新v4/v2闭合，旧错误不删/不翻0；未知新错误仍需实查完整上下文，不泛白名单。
【来源、日更和自动化保护】Day23已唯一正常push5370d46a9bb27a62ce8152a92dfc72fa9c1f2fd9/2972paths303changes/remote同SHA，R39/本地19/MJS25过及初失败回执保留；今天不第二commit。R publisher配置唯一已审CAS至1585e1dce7edea8a7b3c99bf27ae4b2840b715ac3b054542d55141bbc44906e9/publisher e67fc049...和全部原runtime冻结源/runner58de6.../Gwriter83134.../run.py/真实pipeline/PAUSE/POLICY/两书/state/reg/ledger均不改。两个Meta inline200协议仅Git精确排除，R全原件/生成器/YAML在。
本轮CURRENT_SOURCE_COUNT主源码和真实consumer配置/原IR，全部新增前端失败版/修复版/源绑定失败与v4/六算子诊断/清理发现器/Windows builder与控制reader/本地原stderr回执逐SHA原字节归档，Day24首轮全审用途/consumer/deps/fresh并发SHA/full diff/alloutgoing/对应checks/实际publish_code_daily.py消费清单和审核planSHA后一次正常push核远端，无变不空提交/失败续原已审commit/不等提交才跑。唯一https://github.com/lyinguo/910b.git/独立codex/910b-source-layout-20261114/D:/code-lv/910b本轮16:50仍clean，不merge/force/Gitee历史或改R master remote，不gitadd.扫workspace/传权重数据/合成或模型数组/报告日志keys.env凭据归档或base64运输payload。主实现仅R源码/scripts/configs，workspace仅证据，必要所有后续提交文件保持。
原910b本轮只读TOML590801a94c1493e88f304f20f35bf5f31fd3a77e5548f346c735179ca0c52929，ACTIVE/原名称/原线程/300分钟全同，无official改动。local双书本轮freshSHA2981.../6099931bytes/mtime1795290387531571700/ctime1795290387529570400同，非G/三D/ZIPmeta不改。所有家族/失败/用户决定按下述完整规范及父全文继承，包括MedraN当前AA配置BF16≠自己冻结FP16仅四dtype元信息差，不改自己FP16/eps数学或据此推断历史D；Moondream原query768采样reasoning、Florence/Pix/LoRA/词表字体/任务数据等不借换装包。
严重停摆或未恢复异常每轮首行“严重阻塞｜正式评测未运行”；关键当前连接检查失败“严重告警｜无法确认正式评测状态”，列实际模型阶段/样本前向/卡/原因/最近完整合格结果北京时间/最早可证时长/解除动作，真实恢复当轮也报。空卡/算子/CPU前端/加载/等待进程/已完成小样本不充NN/fullNF，磁盘NOTIFY不当聊天已通知。全部有资源款真实原协议910B200/fullNF/三D5%、确缺明确跳过、无pending/资格复审且全量复查双书完整报告历史例外才关闭，远未完成。

'''
template=intro+norms;assert template.count('CURRENT_ACTUAL_JSON')==1
primary={}
for n in ['Kanana_own_frontend200_v1_primary_source_SHA_heartbeat1627_20261123.json','Kanana_own_frontend200_v2_primary_source_SHA_20261123.json','heartbeat1627_verified185_cleanup_discovery_primary_SHA_20261123.json','Kanana_original_RegStage_NPU_synthetic_interface_v4_primary_SHA_20261123.json','Kanana_frontend200_source_external_acceptance_v3_primary_SHA_20261123.json','Kanana_frontend200_source_external_acceptance_v4_primary_SHA_20261123.json']:
    for p,h in json.loads((V/n).read_bytes()).items():
        if p in primary:assert primary[p]==h
        primary[p]=h
stage={'scripts/maintenance/'+mn:s.encode(),
       'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes(),
       'scripts/maintenance/read_Kanana_dedicated_frontend_actual_contract_failures_no_model_20261123.py':(V/'read_Kanana_dedicated_frontend_actual_contract_failures_no_model_20261123.py').read_bytes(),
       'scripts/code_publication_windows/build_Kanana_own_frontend200_binding_and_source_v1_heartbeat1627_20261123.py':(V/'build_Kanana_own_frontend200_binding_and_source_v1_heartbeat1627_20261123.py').read_bytes()}
for n,b in stage.items():ast.parse(b);primary[n]=sha(b)
names=['R_heartbeat1627_required_root_v90_latest_v93_actual_20261123.json','R_heartbeat1627_required_root_v90_latest_v93_actual_20261123.json.stderr',
 'heartbeat1627_fresh_local_book_TOML_code_readonly_20261123.json','heartbeat1627_fresh_local_book_TOML_code_readonly_final_20261123.json',
 'Kanana_frontend_binding_local_v1_pre_stage_failures_observed_20261123.json','Kanana_frontend_binding_builder_v2_actual_stdout.log','Kanana_frontend_binding_builder_v2_actual_stderr.log',
 'Kanana_frontend_v2_reference_only_source_review_20261123.json','Kanana_frontend_binding_builder_v3_actual_stdout.log','Kanana_frontend_binding_builder_v3_actual_stderr.log',
 'R_Kanana_own_dedicated_frontend200_v1_heartbeat1627_20261123.json','R_Kanana_own_dedicated_frontend200_v1_heartbeat1627_20261123.json.stderr',
 'R_Kanana_dedicated_frontend_v1_actual_failure_and_contract_20261123.json','R_Kanana_dedicated_frontend_v1_actual_failure_and_contract_20261123.json.stderr',
 'R_Kanana_own_dedicated_frontend200_v2_original_whitespace_20261123.json','R_Kanana_own_dedicated_frontend200_v2_original_whitespace_20261123.json.stderr',
 'R_Kanana_dedicated_frontend_v2_full_actual_records_20261123.json','R_Kanana_dedicated_frontend_v2_full_actual_records_20261123.json.stderr',
 'R_Kanana_original_RegStage_NPU_synthetic_interface_v4_20261123.json','R_Kanana_original_RegStage_NPU_synthetic_interface_v4_20261123.json.stderr',
 'R_Kanana_frontend200_source_external_acceptance_v3_20261123.json','R_Kanana_frontend200_source_external_acceptance_v3_20261123.json.stderr',
 'R_Kanana_frontend200_source_external_acceptance_v4_20261123.json','R_Kanana_frontend200_source_external_acceptance_v4_20261123.json.stderr',
 'R_heartbeat1627_verified185_current_safe_cleanup_metadata_discovery_20261123.json','R_heartbeat1627_verified185_current_safe_cleanup_metadata_discovery_20261123.json.stderr',
 'R_heartbeat1627_original_three_resources_frozen_readonly_current_20261123.json','R_heartbeat1627_original_three_resources_frozen_readonly_current_20261123.json.stderr',
 'heartbeat1627_old_private_transport_naturally_closed255_server_actual1_preserved_20261123.json',
 'R_Kanana_private_real_RegStage_complete_synthetic_controls739_meta_v1_20261123.json','R_Kanana_private_real_RegStage_complete_synthetic_controls739_meta_v1_20261123.json.stderr']
local=[]
for n in names:
    b=(V/n).read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
q=dict(new_primary_expected_SHA256=primary,local_original_records=local,full_handoff_template=template,
 official_actual_readback=parent['official_actual_readback'],
 current_round_local_and_automation=json.loads((V/'heartbeat1627_fresh_local_book_TOML_code_readonly_final_20261123.json').read_bytes()),
 fresh_cleanup_discovery=json.loads((V/'R_heartbeat1627_verified185_current_safe_cleanup_metadata_discovery_20261123.json').read_bytes()),
 current_original_three_resources_readonly=json.loads((V/'R_heartbeat1627_original_three_resources_frozen_readonly_current_20261123.json').read_bytes()),
 filesystem_capability=parent['filesystem_capability'],
 old_local_transport_actual=json.loads((V/'heartbeat1627_old_private_transport_naturally_closed255_server_actual1_preserved_20261123.json').read_bytes()),
 Kanana_source_meta_diagnostic=parent['Kanana_source_meta_diagnostic'],
 kanana_frontend_acceptance=json.loads((V/'R_Kanana_frontend200_source_external_acceptance_v4_20261123.json').read_bytes()),
 kanana_untrained_NPU_operator=json.loads((V/'R_Kanana_original_RegStage_NPU_synthetic_interface_v4_20261123.json').read_bytes()))
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();pn='heartbeat1627_verified185_Kanana_frontend200_NPU_operator_only_full_handoff_plan_v94_20261123.json';(V/pn).write_bytes(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,os,subprocess\nR=Path('/root/projects/comperror');payload=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,z in payload.items():\n b=base64.b64decode(z['base64']);assert hashlib.sha256(b).hexdigest()==z['SHA256'];ast.parse(b);p=R/n;p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(pn)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+mn+"'),str(p),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_heartbeat1627_verified185_Kanana_frontend200_NPU_operator_full_handoff_v94_20261123_remote.py').write_text(carrier,encoding='utf8')
(V/'heartbeat1627_Kanana_all_new_primary_before_full_handoff_v94_20261123.json').write_text(json.dumps(primary,indent=2),encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb),new_primary_count=len(primary),local_original_count=len(local),finite_full_template_chars=len(template),no_duplicate_original_AA_raw_stdout_copy=True)))

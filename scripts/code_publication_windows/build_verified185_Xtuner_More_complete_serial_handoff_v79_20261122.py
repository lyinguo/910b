"""Build finite full handoff and precise Day23 source additions; payload is evidence."""
from pathlib import Path
import ast, base64, hashlib, json

V = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
binding = json.loads((V / 'R_verified185_Xtuner_More_serial_actual_followup_20261122_0620.json').read_bytes())
ptr = binding['pointer']
assert ptr['current_verified_count'] == 185
assert ptr['current_primary_source_count'] == 2782
assert binding['all_current_canonical_fresh_SHA256'] == ptr['canonical_SHA256']
full = json.loads((V / 'verified185_Xtuner_native_v2_IModel_full_handoff_plan_v77_20261122.json').read_bytes())['full_handoff_template']
full = full.replace('完整当前交接：verified185。', '完整当前交接v79：仍verified185，本次新增完整200/G/完整资格均0。')
full = full.replace('新完整200/新G/新完整资格各1，', '本次新增完整200/新G/新完整资格均0；Gemma4MLX8本日较早完成的各1属于历史已交付，不能重复计数。')
full = full.replace('直接父v76全文SHA1079785379844d60361d7d56f452011d10314e92a8de4b592f83eac975360fb9，', '直接父v78更晚实际全文SHA30114c22c102c37e383075e8635fead89d9f35180271abafc598528a05eea370，父v77完整全文SHA2ff99c0093965c86e582f23e1ca1afdbb7873ea21edb9843bdb4ac465da51f81，再父v76全文SHA1079785379844d60361d7d56f452011d10314e92a8de4b592f83eac975360fb9，')
section = '''【本次真正进度：同Xtuner v2真3正在保存全部原数值，不重复，不热改】
同原J3 f422bbc90f71e573de81f5d62e008375b64785da92dfbdcf24efe24ba35fa114/报告20261122_045843仍以真实proc、内核EX、committed_step时间、FD/io和actual退出为准。06:07:22.423829北京时间已提交index0和index1各2048次前向，累计4096；第1条完整NPZ14518613628bytes已关闭，第2条在06:16:41约2482142267bytes且仍增长。此时正在CPU无损压缩保存完整实际数值，模型任务和原EX仍在，不是新NN/NF或真3验收完成；第3条和fullNF须以更晚实际为准。没有NEW200/G/资格，不把运行3条说成已验3条，不改原2048/唯一EOS128001，不能强加128008/128009早停。专门再次以本款真实AutoTokenizer和原TransformersVLM6011:6045分支证明有效EOS仍128001，额外EOS合并是空dict，无learned CPU/NPU。
Frozen monitor的owner_not_in_project_process_snapshot仅本次真实runner1697730/PPid1697588/argv同J3/真实startticks/内核EX/cwd=/root证明被原111:122过滤排除；原warning完整留，不改monitor/泛白名单。本轮source_errors/new_log_error_candidates为空；若更晚新错误必须读完整上下文，不套旧解释。数字保存阶段不能用CPU压缩声称正在提交新的NPU前向；没有信号、热改deadline或重复runner。原当前串行LZMA保存函数a858f771182c1e8240e8e94d158d6089a4c7ee421b7cae94ab4301c9fb2be485逐字节未动。
本轮同时读取的5款原LLaVA元信息不是NN：Hancom HiLLaVA、SpaceLLaVA、Turkish LLaVA、v1.5-13b和以下LLaVA_MORE。旧state row不作表位置；自己源/任务/精度和原分支前置未闭合的不能派发或称ready。v1.5-13b路径无SF不等于确缺，禁止外借视觉塔。

【LLaVA_MORE原概率协议完整实现已准备，源准备真实0，接续器只等同Xtuner完成和原EX释放】
MID=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning，首sheet C唯一实际row159/B=image-text-to-text/三D.7900/.7900/.79/G空；state旧row99不能发布。自己的4SF16709591824bytes/686全部F16，自己的CLIP和两层projector均在，不借tower/输入/参考。原R YAML a1052e5e48c71e6b2de7277674c588866596afa4bd5d5236e486826524e37e39，原adapter universal_eval/model/adapters/llava_more_vlm.py SHA819af5fbdbda709c179b3b5f1256f19162f43b94d49182f3054a7e0b29af50a1，继承TransformersVLM507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3。自己original mode=discriminative_letter，原use_cacheFalse每样本一次完整前向，fullhead最后位F16→F32后，自己Rust字母与空格字母全部ID取maxlogit，候选顺序tie依原Python max；不是生成协议，不能硬套其它款生成或softmax。原max2048/Greedy尾句/outer10内serial1保留，但本原概率分支不用生成、generation_count0，无KVcache，不能伪称生成或物理KVcache。原EOS[128001,128008,128009]保留但此分支不用。
原_padded RGB(122,116,104)正方形、原CLIP336 uint8BICUBIC/F32normalize/实际F16像素、原system/user/assistant模板/原_strip_eval_instruction均保留。自己的32Llama/H4096/Q32-KV8x128/CLIP24/patch14/select-2/576patch/2GELU投影。原savedTF legacy构造省略rawLlama3.1 rope_scaling，因此自己的实际图是plainRoPE500000/context131072，不能改成raw3.1缩放去凑分。自己的原head/embedding128256先原resize128257再legacy remap；自己的原构造extra图像/头行与完整RNG保持，F32解析RoPEbuffer不转F16。历史A100 engine/tensor未采和旧/mnt2/afs/download_models路径不可读如实披露，不冒称AA。
FRONT=B/LlavaMore_own_original_FP16_discriminative_letter_pad_frontend200_meta_no_NN_v1_20261122，于05:31:57.803476 actual0。frontend SHA0a02f42ae2d23459e6aa47c801fa9327ca94cb570fa6172caed118fd228d749c/meta f8838dc6eebeb1dd732862a713795982b7112eef1db30c1a38139ee96ad7c76e；全200自己原prompt/ref/imageSHA、2..5字母选项ID集合、Rust全token/576展开、独立NumPy pad/TV2像素/实际F16逐位同，所有4329数据与实际原filterTrueFalse同集已证；自己的全4SF全文件SHA/686meta同，不重跑CPU200。source scripts/prepare_LlavaMore_own_original_FP16_letter_pad_frontend200_meta_no_NN_v1_20261122.py SHA91a7af12af35e06a971a5295c1503744915f74194df3b7e51038cbdf85a37067。首launch receipt key里误把SHA占位全局替换导致key source_<hash>256，payload实际SHA断言/执行源91a7与actual0真实正确；旧回执保留不修写不重跑，非模型数学错误。
CPU=B/LlavaMore_own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122，05:42:55.286112 actual0，无learnedforward。source scripts/inspect_LlavaMore_own_original_FP16_loader_resize_remap_no_forward_v1_20261122.py SHA936b998938c533caa136806a8d41d4bf879ac72a3b2f521c1fc4bcd1040892a2，自己的686源/cast/实际CPU remap全位同，sourceproof266e0a9ccaa3178e9d5699eaee39abbb36412f2a7bc08ebe6720357f303ed422；原构造extrahead/embedding/RNG全NPZ5aa9faee75f47f692dfc7c7a107256f5652f6d8e7868bfac162d4b48ed6ec9e8。原HF初685 missing/unexpected就是legacy布局，原remap后全部686恢复，完整warning留；不能把meta或CPU构造说成NN/NF。
新标准完整无损CPU并行NPZ：universal_eval/utils/complete_standard_LZMA_NPZ_parallel_values_20261122.py SHAd1f8594c0dfa6b1fc9f549e9bde6cb634d661de08fbea70acdfb201da2e3c668，tests scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py SHA1441c14aeed7d219dd003c7ba128af0e76a48630819e0f3acac283979cd6eb03。最多8 CPU workers各完整标准FortranNPY/ZIP-LZMA member，主线程按原顺序写标准ZIP64记录，所有原array名/shape/dtype/rawbit/发生引用/完整值不减，无抽样/改精度/共享丢值。Windows和server各7项检查过，含NaN rawpayload/异端/非连续/空/标量/1vs8worker/标准ZIP64边界/目标x拒覆盖；未来本款0/99/199自己完整输入NPZ与原串行wholefile SHA逐字节同。Hdiag=B/complete_standard_parallel_NPZ_own_input3_all_values_storage_diagnostic_only_v1_20261122保留所有原件与全值，只storage诊断、没有模型NN或资格。只用于新More，当前Xtuner活source/job/旧保存函数绝不热改。CPU压缩并行不是模型并行，模型仍max1；真实模型全NF必须再过，不能拿7unit代替。
新MODEL=universal_eval/model/adapters/llava_more_own_original_FP16_discriminative_letter_native_v1_20261122.py SHAa922fc8957c26ac09e6504080d2277117f74be9a90f1147289c1fabef191caac，OwnLlavaMoreOriginalFP16Letter(IModel)；graph universal_eval/model/llava_more_own_original_saved_Transformers_FP16_graph_20261122.py SHAa780ddda49f857bde308e61bdd2d4e59172bb15e54eded43ddf5d3e8682c0e20；ALGO91d6c4964306a7f1404c18a502aab378f9192c07751668505706892936b84b4f。实际调用原actor.predict/_predict_letter/_prepare_inputs/_pad_image/_build_text原函数；私有observer的_letter_scores只调用super原maxscore后完整记录，不改原概率数学。原686ownsource/owncast/actualNPU/extra完整头参数全SHA、所有原module输入输出、32层QKV/全mask/位置/576融合、所有sequencepositions完整128257头、实际原F32最后位原字母ID/maxscore、原无cache均保存并独验。
NF=scripts/verify_LlavaMore_own_original_FP16_discriminative_letter_complete_values_v1_20261122.py SHA97f30e70a99f0b516f7cc242db548d825c0d47faef5fe574f85a213910551b8b，原F16 atol/rtol .01/.01首ownNN前固定，不放宽；独立原物理686+extrahead/自己pad像素/tokens/32层/所有vision/全序列head/alllogits/全部变体ID max/tie/无KVcache/1forward每条、全5件逐条分母/allsource/externalSHA。原generated_ids为空合法概率评分，不当生成。NF stdout仅单proofJSON，状态进stderr，避免旧ExtraData；CPU独验16threads首ownNN前绑定，不能运行失败改阈值或滤样本。
PREP scripts/prepare_LlavaMore_own_original_FP16_letter_true3_source_ready_v1_20261122.py SHAd4ac52b85747839d231135ee42f86aa780a78151833ad0c4fdfc47eb7e71685f。READY=B/LlavaMore_own_original_FP16_letter_source_readiness_v1_20261122于06:06:23.373095真实0，real ModelLoader.load_class/IModel无abstract及原函数identity/meta686/完整图/source/ownconstructor/input/proof全过，validate_config/check_sources/config_for3/200/import过；还不是模型真3。YAML=configs/LlavaLlamaForCausalLM/aimagelab--LLaVA_MORE-llama_3_1-8B-finetuning-own-original-FP16-letter-pad-native-v1-20261122.yaml SHA717c82d5ef1985eae80b47ba872d103d735bd6069a33b0ca545a680aedd3691b；protocol configs/protocols/llava_more_own_original_FP16_letter_pad_native_v1_20261122.json SHA6a172f9b0b4ec17f58c8191046ca7b09721bac49967107fa9d141aa93470dc70；真独立单YAML直接run.py。
J3=B/LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122，jobSHA1a51187709216cf527dbd9da9f9627e8fc41813359a135a0837c513c6a0aa1f1，尚未NN。CONT scripts/continue_LlavaMore_own_original_letter_after_Xtuner_actual_release_true3_capacity_fresh3_fixed200_serial_v1_20261122.py SHAf26d34b7d2510eba49f1f2cb09340d42055416ee458e513f94470313285b834c；实际consumer configs/queues/LlavaMore_own_original_letter_after_Xtuner_actual_release_serial_v1_20261122.json SHAd15e16ad172a904ebb0ae18c54a1300b8b57047f1c90044f49eba248bc7a7b4b；H=B/LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122，snapshot PID1700499只等同Xtuner实际退出/全部NN-NF-controller/原EX释放与卡空、自己不持原EX，每20秒检查，勿重复。若前Xtuner完整200合格，必须等双书交付再MoreNN；合法canonical185→186需要新私有准确重绑定，不能清protected绕过。若前3容量拒或失败，则全退出后自然转ownNEW3；通过才全部自身ZIP部件/全200pixel-token及未知轴maxratio平方/原factor4+6GiB+hard3/log1/.5pre/2GiB×actual1，真容量过→内部首次200测量批准→NEWfresh3/固定0:199 NEW200。小3对完整D不加5%门槛，容量拒保留proof/不重做3。J200=B/LlavaMore_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122以真实存在/实际退出优先。控制器不写G，完整200/fullNF/原资格/三D5%且全部NN-NF-controller-EX退出后另ArtifactTool实写实看/原writer11/跨双书真EX备份CAS发布，不先写表或把ready当200。
本轮新有用所有native/NF/graph/算法/源准备/实际consumer/CPU保存utility和7tests/Windows3builders/8只读readers/自动YAML-protocol-smoke和本完整维护源均在R源码/scripts/configs，当前完整清单逐SHA原字节pending Day23，不漏依赖/后续提交文件。不第二Day22commit；opaque stage_launch运输base64只证据不当Git代码。本轮未删除任何模型值/报告/源/配置/日志。所有失败和原first-middle-last/错invalid/trunc/error/small/未验保护继续。

'''
at = full.index('【Bee SFT：')
full = full[:at] + section + full[at:]
stage = {}
reader_names = [
    'read_heartbeat_verified185_Xtuner_required_latest_handoff_full_20261122_remote.py',
    'read_verified185_pending_Llava_original_deployment_only_priority_no_NN_20261122_remote.py',
    'read_next_owned_Llava_five_original_protocol_header_metadata_no_NN_20261122_remote.py',
    'read_LlavaMore_original_adapter_complete_source_no_NN_20261122_remote.py',
    'read_Xtuner_own_actual_tokenizer_original_TF_EOS_branch_metadata_only_20261122_remote.py',
    'read_verified185_Xtuner_current_LlavaMore_frontend_only_no_second_NN_20261122_remote.py',
    'read_verified185_Xtuner_current_LlavaMore_source_constructor_v2_no_second_NN_20261122_remote.py',
    'read_verified185_Xtuner_LlavaMore_serial_ready_frozen_monitor_actual_20261122_remote.py',
    'record_verified185_Xtuner_More_full_serial_actual_handoff_v79_20261122.py',
]
for n in reader_names:
    stage['scripts/maintenance/' + n] = (V / n).read_bytes()
stage['scripts/code_publication_windows/' + Path(__file__).name] = Path(__file__).read_bytes()
primary = json.loads((V / 'LlavaMore_own_original_probability_true3_all_primary_v1_20261122.json').read_bytes())
primary.update({n: sha(b) for n, b in stage.items()})
primary.update({
    'scripts/prepare_LlavaMore_own_original_FP16_letter_pad_frontend200_meta_no_NN_v1_20261122.py': '91a7af12af35e06a971a5295c1503744915f74194df3b7e51038cbdf85a37067',
    'scripts/inspect_LlavaMore_own_original_FP16_loader_resize_remap_no_forward_v1_20261122.py': '936b998938c533caa136806a8d41d4bf879ac72a3b2f521c1fc4bcd1040892a2',
})
local_names = [
    'R_heartbeat_0518_verified185_Xtuner_required_full_handoff_20261122.json',
    'R_next_owned_Llava_five_original_protocol_header_metadata_no_NN_20261122.json',
    'R_Xtuner_own_actual_tokenizer_original_TF_EOS_branch_metadata_20261122.json',
    'R_LlavaMore_own_original_frontend200_meta_v1_actual_launch_20261122.json',
    'R_LlavaMore_original_CPU_constructor_no_forward_v1_actual_launch_20261122.json',
    'R_heartbeat0518_Xtuner_current_LlavaMore_ctor_v2_actual_20261122.json',
    'R_parallel_NPZ_complete_own_input3_storage_only_v1_20261122.json',
    'R_LlavaMore_original_probability_serial_after_Xtuner_v1_actual_launch_20261122.json',
    'R_heartbeat0518_Xtuner_LlavaMore_ready_frozen_monitor_actual_20261122.json',
    'R_verified185_Xtuner_More_serial_actual_followup_20261122_0620.json',
    'LlavaMore_own_native_probability_primary_paths_v1_20261122.json',
    'LlavaMore_own_probability_fullNF_primary_paths_v1_20261122.json',
    'LlavaMore_own_original_probability_true3_all_primary_v1_20261122.json',
    'parallel_NPZ_complete_storage_primary_paths_20261122.json',
]
local = []
for n in local_names:
    b = (V / n).read_bytes()
    local.append(dict(name=n, SHA256=sha(b), base64=base64.b64encode(b).decode()))
q = dict(expected_pointer_SHA256=binding['pointer_SHA256'], expected_pending_SHA256=ptr['new_primary_pending_SHA256'],
    expected_closing_SHA256=binding['closing_SHA256'], expected_parent_source_count=2782,
    round_name='verified185_Xtuner_same_v2_More_original_probability_ready_serial_complete_v79_20261122',
    full_note_name='full_actual_verified185_Xtuner_same_v2_More_original_probability_ready_serial_v79_20261122.txt',
    pending_name='pending_after_daily_20261122_verified185_Xtuner_More_probability_complete_all_primary_v79_20261122.json',
    new_primary_paths=sorted(primary), new_primary_expected_SHA256=primary,
    full_handoff_template=full, local_original_records=local)
qb = json.dumps(q, ensure_ascii=False, indent=2).encode()
name = 'verified185_Xtuner_More_probability_complete_serial_handoff_plan_v79_20261122.json'
(V / name).write_bytes(qb)
qh = sha(qb)
payload = {n: dict(SHA256=sha(b), base64=base64.b64encode(b).decode()) for n, b in stage.items()}
carrier = "import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode(" + repr(base64.b64encode(json.dumps(payload).encode()).decode()) + "))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n\n if p.suffix=='.py':ast.parse(b)\n p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nD=R/'workspace/diagnostics/batch_910b';T=D/'heartbeat_handoff_plans';T.mkdir(exist_ok=True);q=T/" + repr(name) + ";b=base64.b64decode(" + repr(base64.b64encode(qb).decode()) + ");assert hashlib.sha256(b).hexdigest()==" + repr(qh) + "\nif q.exists():assert q.read_bytes()==b\nelse:q.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/record_verified185_Xtuner_More_full_serial_actual_handoff_v79_20261122.py'),str(q)," + repr(qh) + "],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier)
(V / 'stage_record_verified185_Xtuner_More_probability_complete_handoff_v79_20261122_remote.py').write_text(carrier, encoding='utf8')
print(json.dumps(dict(plan_SHA256=qh, full_template_chars=len(full), new_primary_paths=len(primary), local_original_records=len(local), original_local_bytes=sum(len(base64.b64decode(r['base64'])) for r in local))))

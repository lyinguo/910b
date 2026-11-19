"""Prepare the existing heartbeat update, preserving all non-prompt settings."""
from pathlib import Path
import hashlib,json,tomllib
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
a=read('R_archive_verified180_Moondream_loading_KV_v3_actual_1145_20261119.json')
base=read('archive_verified180_Moondream_loading_KV_round_plan_1138_20261119.json')
assert a['passed'] and a['canonical_files_unchanged']
raw=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes()
assert raw==(P/'automation910b_before_Moondream1106_20261119.toml').read_bytes()
old=tomllib.loads(raw.decode());assert old['id']=='910b' and old['status']=='ACTIVE'
first=old['prompt'].split('\n\n',1)[0]
ref=f"必须先完整读{a['handoff_directory']}/handoff.txt，UTF8/LF SHA256={a['handoff_SHA256']}；再读同目录facts.json SHA256={a['facts_SHA256']}、closing_addendum.json、late_actual_notes和D/A100_ALIGNMENT_HANDOFF_CURRENT.json更晚交接。正文/facts为初快照，更晚真实proc/内核FLOCK/实际退出/closing和最新真实用户决定优先，勿按旧live重启完成款。父正文d47ffcf7243cc7839eccb84b343b5f9537bae420fc2be1a427d894e5df153465/factsb5692b6140796fc5dbfb3738bbb193ebae6be656a0d3f12c28a6569d9c88f001及全部祖先/65471字符archive继承全部原强制规范、家族失败、精确清理例外和用户决定，短引用不取消约束。本轮2064初主源/config、55本地原件元信息/原字节及15静态原件、原prompt/TOML已归档；大旧plan/payload不递归拼接，原失败目录和原字节保留，更晚closing/pending优先，不漏有用代码或后续提交文件。完整交接加短引用，不无限拼接、不存密码。"
delta='''【11:06本轮实际进展；仍严重阻塞】11月19日11:25:52北京时间完整冻结monitor/实际proc/原FLOCK确认没有项目模型NN/全模型NF/原EX，NPU1..15空、0外部VLLM不动，原盘99361284096bytes约92.54GiB。本轮新完整200/G/完整资格均0，180交付、165数值内/15超差/210完整权重待交付/310确缺、完整原协议66不变；165=66完整+99仅数值，390完整权重仍324需适配/资格/修复。最近完整200仍Prithiv11月17日16:57:59.615811，.880对三D.87均1.1494252873563%，已两书交付不重跑。本段停摆仍从11月18日20:47:01.480853算，不能把下面加载/缓存/存储测试当恢复。Huihui完整200需152.74GiB，zai-GLM219.53、Ovis原265.78、Phi1244.15，都超过当前盘。原GLM每20秒无NN等候器仍真实存活、不持EX，235720910370bytes和全部原门槛真过才自动NEWfresh3/fixed200，不重复或拿waiting当NN。11:25 selector0/null，三Gemma根仍暂不可读，不加310/不清旧protectedstate。原扩400GiB问题未答不是批准，不自行扩容/换盘/减guard/删未验。本轮无删除。

【Moondream3的实际加载精度已证明BF16，不再仅凭meta】自己的modelv2四片18537326392bytes/662BF16、11py/词表与原YAML/共享源码均保持。按原非CPU auto分支请求torch.float16，实际原HF类11:16:29 CPU加载和11:22:15 NPU4加载均为全部662BF16，逐源shape/全部raw位/SHA与自己原权重相等，missing/unexpected/mismatched/errors全空，text.wte与head未绑；这不是强改dtype。NPU加载前fresh全四SF与tokenizer全文件SHA，所有source→actualNPU逐值SHA通过；原完整4096mask/730前缀双向、F32 RoPE和24层48个1x32x4096x64 BF16零KV全数值过，use_flex_decoding=True不改。Jcpu=B/Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_v2_json_sets_1116_20261119；Jnpu=B/Moondream3_own_actual_NPU_loading_all662_runtime_buffers_no_forward_1120_20261119。两者actual0但model_forward_calls=0，不是模型真3/fullNF/资格或正式恢复。CPU v1实际1因loading_info的set不能JSON序列化，完整原错误/源/日志留；v2只修记录。NPU padding32警告全留，不冒称视觉kernel。原运行mask/RoPE/完整零KV NPZ508300bytes SHA8e89da320180d64f5e7e96042409bf2c5c356f10bc92f59afdfa04915da76f9a留；均自然退出/EX已释，不重启完成的加载诊断。

【完整模型尚未ready】已有自己的frontend200与5组无权重Flex算子控制不重跑。原vision27/378/max12local+global/patch14、text24/H2048/Q32-KV32/D64/后20层64专家top8/51200全头/物理KV4096、BF16逐步div255/sub.5/div.5不变。原HF.generate的query分支忽略保存max512，原reasoning=True/每段768/temperature.5/top_p.9/首[1,15381,2]+question+[4]/answer[3]/EOS0都保留披露，不能强改greedy512或禁thinking凑D。原Flex在NPU API拒绝和旧诊断actual1保持，私有完整causal mask→SDPA接口目前只5组无权重对照过，原use_flex_decoding=True/原源码/共享库不改，尚未接模型。
新增候选KV journal及原cache调用观察hook在universal_eval/model/adapters；原update/load_encoded_image调用及返回不改，完整actual物理KV与真实输入逐raw位核后，仅在逐dtype/shape/全部位同才共享完整NPY，每个发生保留，独立全值重建。无权重本地24层/27发生/7完整NPY22780512bytes通过，越原写位置改值/重复position/改int32都拒，NaN/负零位保持。v1测试把实际共享后7文件误期望8的失败和所有原值留，v2只改期望8→7、actual0。hook仅候选/AST和依赖核对，尚未真实模型集成；没有完整adapter/单YAML/视觉-MoE-tau-RoPE-head-全部生成步独立NF/真3/批准/fresh3/200/G。下一步补齐并绑定表C唯一/B/三D，validate/check_sources/config3/200和全部准入真过才NEW真3，不用原加载/零KV/单测冒充NN。历史AA engine/tensor未采如实披露。

【保留与代码】本轮原CPU加载JSONset失败、KV测试7/8计数失败、首次归档副本绝对路径guard拒和随后本地大旧payload递归归档guard拒均完整留；新归档仅修相对副本标签和有原字节的大旧plan/payload不重复嵌套，原五件/源/配置/日志/数值/表/state/POLICY不动。55本地原件和15静态原件、2064源已归档，更晚closing补官方回执与源。全部报告、所有小样本/失败值/必要例证保持，本轮没有删除。MedraN旧FP16视觉非有限/旧NF路径错误仍未合规修复，不能改BF16/epsilon或只修路径验收；Isaac真实tensorstream缺、Florence原任务/数据/autoF32绑定等旧pending继承，不能装包/stub/借换。
11月19日已唯一push e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d，今天不第二commit。本轮有用加载/候选KV/真实消费hook/失败与成功测试/Windows控制源码都在R主源码/scripts，不在workspace只存快照；Day20按用途/真实消费者/依赖/freshSHA/全diff/alloutgoing/检查/planSHA一次正常push，不漏有用实现。当前初pending=C/pending_after_daily_20261119_verified180_Moondream_actual_NPU_loading_KV_candidate_all_primary_v34.json，SHA1cb1f88e5c83cdbbafc7ee59db6535a9d176111b18685d831488ddd64df8fe10，更晚closing完整pending优先；字面base64运行payload只归档证据，不传GitHub。全部原规范如下及父祖先继承，完成加载不等于运行成功。
'''
tail=old['prompt'][old['prompt'].index('R=/root/projects/comperror'):]
prompt=first+'\n\n'+ref+'\n\n'+delta+'\n'+tail
assert 'RRULE' not in prompt
args=dict(mode='update',id=old['id'],kind='heartbeat',name=old['name'],status=old['status'],rrule=old['rrule'],targetThreadId=old['target_thread_id'],prompt=prompt)
if 'notification_policy' in old:args['notificationPolicy']=old['notification_policy']
(P/'official910b_Moondream1106_update_arguments_20261119.json').write_text(json.dumps(args,ensure_ascii=False),encoding='utf8')
(P/'prompt_official910b_Moondream_loading_KV_current_1106_20261119.txt').write_text(prompt,encoding='utf8',newline='\n')
print(json.dumps(dict(prompt_chars=len(prompt),prompt_normalized_SHA256=sha(prompt.rstrip('\n').encode()),preserved={k:v for k,v in args.items() if k!='prompt'}),ensure_ascii=False))

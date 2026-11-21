"""Write a complete current handoff and preserve new useful sources, without touching active jobs."""
from pathlib import Path,PurePosixPath
import json,hashlib,base64,ast
V=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
rec='scripts/maintenance/record_verified184_Meta117_BeeRL_serial_handoff_v3_20261121.py'
s=(V/'record_verified184_Meta117_Bee_v4_handoff_v2_20261121.py').read_text('utf8')
s=s.replace("'Meta_fullNF4':stage(","'BeeRL_storage_fullNF_capacity':stage(B/'BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121'),\n 'Meta_fullNF4':stage(")
s=s.replace("'continue_beesft','bee8bsft_own'","'continue_beesft','continue_beerl','bee8brl_own','bee-8b-rl','bee8bsft_own'")
a="active=bee_nf or any("
b="""rl_home=B/'BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121'
 rl_nf=(rl_home/'storage_fullNF_stderr.log').exists() and not (rl_home/'complete_storage_fullNF_proof.json').exists() and not (rl_home/'actual_completion.json').exists() and any(x['PID']==stages['BeeRL_storage_fullNF_capacity'].get('state.json',{}).get('value',{}).get('PID') for x in owners)
 active=bee_nf or rl_nf or any("""
assert a in s;s=s.replace(a,b)
s=s.replace("'verify_bee8bsft_' in a", "('verify_bee8bsft_' in a or 'verify_bee8brl_' in a)")
s=s.replace("'verified184_Ovis_delivered_Meta117_retention_complete_Bee_v4_actual_capacity_wait_no_formal_NN_or_NF'", "'verified184_Ovis_delivered_Meta117_retention_complete_BeeSFT_capacity_wait_BeeRL_storage_prepare_or_fullNF_see_actual'")
s=s.replace("'Bee full storage NF already passed; complete200 capacity blocked since 20261121 20:35:24 BJT; v4 capacity-only wait; see actual snapshot'", "'BeeSFT full storage NF passed but capacity wait; BeeRL complete lossless copies and own fullNF/capacity in same serial controller; exact live NF flag in snapshot'")
s=s.replace("current_stop_earliest_proven_Beijing='2026-11-21T20:35:24.837093+08:00'", "current_stop_earliest_proven_Beijing=None if active else '2026-11-21T20:35:24.837093+08:00'")
compile(s,rec,'exec');(V/PurePosixPath(rec).name).write_text(s,encoding='utf8',newline='\n')
extra=['read_BeeRL_existing_completed3_own_source_contract_20261121_remote.py','read_BeeRL_exact_original_completed3_protocol_and_file_SHA_20261121_remote.py','read_BeeRL_exact_original_reference_backup_20261121_remote.py','read_BeeRL_serial_lossless_storage_fullNF_actual_20261121_remote.py','prepare_BeeRL_completed3_full_LZMA_NF_capacity_serial_20261121.py',PurePosixPath(rec).name,Path(__file__).name]
records=[]
for n in extra:
 b=(V/n).read_bytes();ast.parse(b);records.append(dict(path='scripts/maintenance/'+n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
primary=json.loads((V/'verified184_Meta117_Bee_v4_all_new_primary_v70_20261121.json').read_bytes())+json.loads((V/'BeeRL_completed3_full_LZMA_primary_paths_20261121.json').read_bytes())+[x['path'] for x in records]
full=(V/'serial_fullNF_current_full_handoff_template_20261121.txt').read_text('utf8')
full=full.replace(full.splitlines()[0],'完整当前交接：Ovis真200/fullNF2已双表交付184；Meta真200/fullNF4超差并117正确非例证精确清；BeeSFT新完整存储全验过但容量不足，v4无NN等候；立即串行推进BeeRL自己的新完整存储/fullNF/容量，不重跑原3。')
def replace_section(start,end,new):
 global full
 i=full.index(start);j=full.index(end,i);full=full[:i]+new.strip()+'\n\n'+full[j:]
replace_section('【基线与资格】','【Ovis真实200',r'''
【基线与资格，verified184唯一当前】184已交/169数值内/15超差/206完整权重待交/310确缺/原完整资格70；169=70完整+99仅数值，390完整权重仍320需补适配/资格/修复。双书0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1/stateb001dfe5c178d8ba1191ba8ec894fbd8e7ab81b62a50193262e6d8a7fe4cd450/reg42347681b4fe082b1172b6de023a2cc1d1e2f343a54fdf8d656fa2b7fd40614c/bindingf66b7500bbb8c5f5a99a4c35b248f8203b5fd9bde4bbaa15c7f32f21af58526c。Hpub=B/OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121；export adf185dd40b3bd364df1c425ea790d64673ca6a51224203a67ea8f56843c36b8，active184_full_report_stat_baseline1549f33251985bdd1c1c35bb3752e4dc992ae9f87be700ae3076c8ee899147ee，fulltargetaudit033e43ca89d3f6ba87a8924490c98b696e8a96565014dd1d498e2dd8cc3a06a2。P/OvisU1_verified184_incremental_real_results_export_20261121.json原字节同export；本地freshSHA/stat已核。新185先真实计数、freshlocalSHA/stat、29合法精确round9/310缺空G/旧全部C-B-D-G和完整五件stat，changes0不写，严禁旧183/更旧导出覆盖184。
''')
replace_section('【Ovis真实200','【Meta真实fullNF4',r'''
【Ovis已完整通过，禁止重跑】AIDC-AI/Ovis-U1-3B row144/三D.8750/.8750/.875，J=B/OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121 job11f0c684e4dfe6f39bff484c2386dca3e7371f778230faa6ca2a65ca64affc7d。formal20261121_093507/NPU4真实200/102400前向、每条原512，11月21日17:03:34 model0。Accuracy.895=179/200，对三D均2.285714285714%；21错含198/199 invalid，error/empty0，全200原截断标志保持。自己的1060物理487活动573unusedYak/Qwen3_28/AIMv2_24/151936全头/outer4内serial1/BF16/原F32RoPE/EOSNone512保持，不注入早停。真3/1536 proof221e4c...、完整LZMA3 fullNF9c98.../c938...、批准NEWfresh3 a3422cc1cf1b45002e910374e655d832941702741a9cfe7ec9064c6d987f7443全过，不重放。
旧NF1仅198元数据错，18:09:33原job/NF/runner1保持：e1395963e8ef3861025886313cf18e0dc44d3f3a772806e1a215ff37609f3c4d。原Greedy短字母提取失败就保存raw，198/199完整<B>句子</B>仍错/invalid/分母不变。NF2只期望prediction=(choice or raw)，计分choice和全部数值阈值数学不变；scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py 60c0d21fd51f9ad8e712890330e9b7c887d72e8154ace6a14b11ab5f6fc20052。
HNF2=B/OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121于20:15:59.056909正式全200独验0，proofb39872184e780424b2bb4b7e156fce31b3be0391fcb6efe4d893e6c5fb705818/acceptance36244380d0c754936d3255e2869cd02e26633360c70c6680e0e599eeb1e895b1；该验收NEW模型前向0，准确引用同genuine200/102400，不翻旧job0或假称重复NN。所有原200完整矩阵因原截断保护全留。
''')
replace_section('【Meta真实fullNF4','【条件Ovis184',r'''
【Meta fullNF4已过但超差，117精确适清已完成】meta-llama/Llama-3.2-11B-Vision-Instruct row413/三D.775，原同模型NPU4+5真200/2823，.705=141/200，59错、6invalid[14,95,97,100,105,153]，三D9.0322580645%>5，G413空、不增15或资格。旧NN1仅完整baselinegap，旧NF3元数据失败1均保持。新NF4只invalid原空字符串schema，19:09:41.966014正式fullNF0；proofc54902717b16dd42b1b02f851c02652ea77d9ad31884eae238d38dcc3c08cb13/acceptanceadb80b13cd349a23932cc18c608d342b6225ff51a8aed6fa0f3d5fde74179e3b/actual4f70ca9c6ba998daa0050022f39419c349d1e4a49d3afbddb48eff204eb4d07f。906源/fullhead/40KV/全部输入与外部SHA过，验收NEW模型前向0。
按用户最新明确清理要求，RET=B/verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121，plan a5a2f5e6564db2d3d1ead96867cf0b17ff5b2e0e760ce45c57469aa503d10c1f，完整targetaudit e6cccf5b7dab484145de8b199d6885be36ae589b4bec66fcd8420072edc590d8，22:24:33.142416实际清理0。真原EX/quiet/逐SHA/foreign-FD无引用/10790保护前后SHA全同下只删117个正确非例证NPZ，15312491122逻辑bytes/15312990208allocated≈14.26GiB；free121238806528→136551583744。保留83条完整矩阵，含全部59错/6invalid及原outer10完整0:9、90:109、190:199；所有小样本/fresh3/失败/未验、完整五件、逐条raw/ref/error/token/统计分母、源码YAML快照日志全留。receiptc2ed3e30c9f71b4695ea3dfa35630834f4b9752fc58e13f84c1d64a900c63d3c/manifest363e58f7b5e5463c9761df7b11c6780aa9ff0774f89a0f650e440c53c4facc59；ledger6f4dcf0b...→b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd合法精确追加。117例外现在不能再全数组复验，原proof字节留并必须披露。两书/state/reg/G/分数/绑定不变。
v1错ledger pin、v2误用sample扁平布局均在删除前拒；v3用实际groups/0001..0200/actual_visual_positions_cache_full_vocab.npz及signed组映射完成。原SSH返回Connection reset但plan已在原服务器完整完成，先核真实进程/planSHA后只apply一次；全部错误原件留，不重启或扩大范围。
''')
replace_section('【条件Ovis184','【Bee下一',r'''
【Ovis184双书已真实交付，非预览】20:17:09.785711完成全备份/原唯一runnerEX跨server+local精确CAS，只G144空→.895；11隔离实表、20拒绝元数据测试、865distinctSHA/alltargetaudit0、ArtifactTool实际author和实际发布PNG实看均过。publication receipte4a577bb964a5232e60d48d7dcc89a926ab5c4f8b67a67eaea7ac6bb2028fafb/localeef65cc140d23186dd9faaf70a2adfc5e5bd8908bfdfa830220ad2dd4fc864ac。旧183全部C/B/D/G、五件stat、非G/ZIPmeta保持，原窄列General不改。privatewriterbc75450f... raw写函数与冻结83134逐字节同，publisher21aab12e.../conditional7ffd4a5e...真实0，原NNjob仍失败；本地driver已完成0，不重启。初inlineStrings读取错误及一次渲染参数错误均在canonical外，修复后的真实回执全留。
''')
replace_section('【Bee下一','【保护、代码',r'''
【BeeSFT完整新存储fullNF已过，200容量堵塞；v4只等资源】自己的Open-Bee/Bee-8B-SFT原真3/6/proof122118431066100d6dd17a26b7d7d3aa37958de5c117cb3814f2a316a892e803保持，不重跑。827原FP16源按自己已存BF16部署、Siglip26/384-anyres6/Qwen3_36/GQA8x128/152000头/36nativeKV/4096/outer10内serial1/EOS151645及首token边界、原thinkingFalse全保持。新模型9b5db79b...只save_npz完整FortranNPY/ZIP-LZMA；所有非存储函数/整类AST同原自己源。新NF5198f3bd...只原raw回退元数据，不改评分数学。
H2=B/BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121，20:17:21双表锁释放后即接；所有原3完整240部件逐name/shape/dtype/raw位同，20:35:24新的完整存储fullNF通过，proof30927e8537358e03b28aa9bee0fa8679de2c073d84f43847685ecee80261161a。但容量calculator读空tensor_metadata['pixel_values'] KeyError，原actual1/所有值和日志留。原200 frontend的BatchFeature flatten元数据全空如实披露；原独立像素和Rusttoken/patch记录实际已过，不冒称空字段绑定SHA。
H3=B/BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121，只用原保存patches对应真实[1,P,3,384,384]和原3全ZIP/全部200形状修容量轴，未重压/重做fullNF/NN。20:54:39实际容量拒1：192555984252bytes=179.33GiB，原factor4+6GiB/硬3/日志1/预检.5/2GiB×outer10均不降，经验非4096最坏保证。117清后仍约127GiB，缺约52GiB，未首次200批准/NEWfresh3/200/G。
新的无NN容量等候器 scripts/continue_BeeSFT_completed_storage_NF_wait_actual_capacity_v4_serial_20261121.py SHAd4053dc6ff3708fe2046cb80d9e4bf6a2aa8e2dfa5ac66620e4b551aea800359，消费configs/queues/BeeSFT_completed_storage_NF_capacity_v4_wait_actual_capacity_serial_20261121.json SHA36467df7541d12ab886852984d2f44059e80d14b6b329ff74a0b444667abdc2e。Hwait=B/BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121，每20秒实际原盘容量/所有NN-NF-EX/source/canonical检查，自身不持原EX，不当NN/NF。另修仅在旧v3容量真过后才会触发的未定义a审批变量，准确绑定original已过proof；容量数学AST与旧v3完全同，模型数学/参数不变。真过才新私有H=B/BeeSFT_completed_storage_NF_capacity_v4_actual_recovery_approved_serial_20261121与J=B/BeeSFT_completed_storage_NF_capacity_v4_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261121→内部测量批准→NEWfresh3/fixed200。旧v1错误反斜杠相对路径等候器确证无子NN/无EX且前置永不存在，原源和进程保持不信号/不伪造完成；旧v2/v3不重启。资源问题已再次问原盘至少+100GiB，未答不是批准，不能擅扩容换盘或降guard。

【立即继续BeeRL自己的尚未完成阶段】Open-Bee/Bee-8B-RL不同自己827源全BF16，原J3=B/Bee8BRL_own_original_BF16_4096_true3_v1_real_FLOCK_20261116，job8ba4ff0ef3a093a1140872fbe6c88658c83a0cf7c6be2433f5eadd8e9670c566/completiondd5249192cf697cf4658b075e8b75d095510ee6817cb8ab5b66feb0a80a8f37d/proofbad8d9d4ba3d126459edb2ca921be1aaea829f0c17003301b0f3029268299996真实3/11正式独验0，原200原格式242.77GiB容量拒。三D.9350/.9350/.935；小score1.0差6.95187%不是200成绩/额外5%门槛。
只读核自己的完整协议/source/front200/原函数与停止后，新adapter universal_eval/model/adapters/bee8brl_own_native_bf16_4096_complete_LZMA_20261121.py 7de3a373ea2bae78a88f47a8662e1c11b7497453a52dedd40d861cecc97eccce仅完整保存，所有非存储函数/整类AST与自己原330189d...逐项相同。futureNF8032414b...仅原raw期望回退，原独立数学阈值不改。自己的原前端200文件9c41c4f7...不重做；不能把本款全BF16误说SFT原FP16源。自己原参考书364cf16d7fca909bafa4c3fef0c042c595a9be7f6bdcea236d66c1278fb4e87b用GLMOCR_pair_common_baseline_verified169_G_publication_20261116/before_verified_write精确原字节备份，全部700 C/B/row/完整D和旧G与当前184一致后只check重定向，不借其他参考分。
新 scripts/continue_BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.py 3a65b4879688b61bcb82398df80f99cacdfff938911eaf70cfcf52e2c544fb9d，configs/queues/BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.json 3a28ebbb4727f88234113cd0de1de78db643b8c064ecc21dc63b782f8b5f83cb；H=B/BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121。22:48:23实际原EX开始完整副本准备（仅压缩不当正式NF或新NN）；依次全3原值/原shape/dtype/raw位后运行自己的完整独立NF并核全200自己形状容量。原真3和front200不重放。true fullNF已开始/完成以storage_fullNF_stderr.log/真实FD/io/原EX/proof/actual退出为准，别把复制状态叫NN、别按旧state重启。新未来J=B/BeeRL_completed_true3_full_LZMA_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261121仅容量真过后创建，原4096/math/停止/outer10不变；不足保原所有值/五件/source/log，转其他可推进项。
''')
full=full.replace('本轮无删除。retpolicy','本轮仅新增上述117精确清理；未删任何临时/日志或未验/失败/小样本。retpolicy')
full=full.replace('ledger6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c','ledgerb3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd')
full=full.replace('当前Ovis/Meta200及所有小样本/失败/未验/完整必要例证/完整五件','当前Ovis全200、Meta83完整值、所有小样本/失败/未验/完整必要例证/完整五件')
full=full.replace('23及祖先271/154','117及23、祖先271/154')
full=full.replace('原Ovis失败18:09:33后Meta18:09:35接锁，Meta19:09:41完成后Ovis19:09:46接锁，不能跨恢复乱算停摆。','本轮BeeSFT新存储fullNF20:35:24结束后正式200容量拒，停摆最早20:35:24.837093；22:48:23 BeeRL仅压缩准备仍不算恢复。真正开始完整NF后的首个可证实查时点应当轮报恢复，不跨恢复累计旧停摆。22:26冻结monitor0/source_errors空/newlog空；原Ovis旧job return1 inspection仅对应已由新正式NF2及184全审闭合的历史元数据失败，原旧job不翻0，未知新错误不能套解释。')
full=full.replace('此次只读候选Meta base','本轮三Gemma根21:25:47仍暂不可读，原selector只读资源门槛核查，无dispatch、不加310。目录检查空间主要是模型数值矩阵（model_runs约597GiB），非RAM/算力不足；所有临时suffix候选为空，未删日志/JSON。此次只读候选Meta base')
full=full.replace('当前完整主源CURRENT_SOURCE_COUNT','本轮全部新增有效主源和真实配置、Windows控制及失败修复完整保留；当前完整主源CURRENT_SOURCE_COUNT')
(V/'verified184_Ovis_dual_Meta117_Bee_serial_full_handoff_template_v70_20261121.txt').write_text(full,encoding='utf8',newline='\n')
names=[
 'R_OvisU1_verified184_exact_local_CAS_receipt_upload_20261121.json','R_verified184_export_and_report_stat_exact_20261121.json','R_verified184_post_Meta117_frozen_monitor_storage_and_process_20261121.json',
 'R_Meta117_v3_plan_actual_process_status_20261121.json','R_Meta117_v3_exact_reviewed_plan_apply_actual_20261121.json','Meta117_v3_reviewed_plan_full.json',
 'R_verified184_current_accepted_cleanup_discovery_20261121.json','R_BeeSFT_capacity_v3_actual_saved_patch_axis_serial_launch_20261121.json','R_BeeSFT_completed_storage_capacity_v4_immediate_serial_launch_20261121.json','R_Bee_v4_wait_actual_capacity_and_existing_dedup_sources_v2_20261121.json',
 'R_BeeRL_existing_completed3_own_source_contract_20261121.json','R_BeeRL_exact_original_completed3_protocol_and_file_SHA_20261121.json','R_BeeRL_exact_original_reference_backup_20261121.json','R_BeeRL_completed3_full_LZMA_NF_capacity_serial_launch_20261121.json','R_BeeRL_serial_lossless_storage_fullNF_actual_20261121.json',
 'R_LlavaMed_and_LlavaMore_own_original_resources_metadata_20261121.json','R_original_resource_gate_and_project_storage_usage_20261121.json','local_verified184_post_Meta117_fresh_SHA_stat_20261121.json','verified184_export_complete_metadata.json','active184_full_report_stat_baseline.json','full_target_dual_publication_audit.json','official_910b_before_verified184_Meta117_Bee_v4_20261121.toml','official_910b_OvisNF2_MetaNF4_Bee_serial_actual_readback_20261121.toml']
local=[]
for n in names:
 p=V/n
 if p.is_file():b=p.read_bytes();local.append(dict(name=n,SHA256=sha(b),base64=base64.b64encode(b).decode()))
q=dict(expected_pointer_SHA256='5946bf075e842cfa07c0fd5b8cd2e1f43f168ddbc50b51868b7c851c2d155202',expected_pending_SHA256='6ee21343730ea8adefaa5de7b39e390f00e6c3636da6e9dcacaa07cd04f66ac6',round_name='verified184_Ovis_delivered_Meta117_exact_retention_BeeSFT_v4_wait_BeeRL_own_storage_serial_20261121',pending_name='pending_after_daily_20261121_verified184_Ovis_Meta117_BeeSFT_BeeRL_serial_all_primary_v70_20261121.json',full_note_name='full_actual_verified184_Ovis_dual_Meta117_BeeSFT_capacity_wait_BeeRL_own_storage_serial_20261121.txt',new_primary_paths=primary,local_original_records=local,full_handoff_template=full)
plan=V/'verified184_Ovis_dual_Meta117_Bee_serial_handoff_plan_v70_20261121.json';plan.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
payload=[dict(path=x['path'],SHA256=x['SHA256'],base64=x['base64']) for x in records]
data=plan.read_bytes();pb=dict(SHA256=sha(data),base64=base64.b64encode(data).decode())
remote='DATA='+repr(payload)+'\nPLAN='+repr(pb)+r'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,importlib.util
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256']
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as f:f.write(b)
b=base64.b64decode(PLAN['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==PLAN['SHA256']
p=D/'heartbeat_handoffs/verified184_Ovis_Meta117_Bee_serial_current_v70_plan_20261121.json';assert not p.exists()
with p.open('xb') as f:f.write(b)
sys.argv=[str(R/'RECORDER'),str(p),PLAN['SHA256']]
sp=importlib.util.spec_from_file_location('record_verified184_Meta117_BeeRL_v70',R/'RECORDER');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.main()
'''.replace('RECORDER',rec)
(V/'record_verified184_Ovis_dual_Meta117_Bee_serial_handoff_v70_20261121_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(prepared=True,new_primary=len(primary),new_recorder_SHA256=sha(s.encode()),local_original_records=len(local),full_template_bytes=len(full.encode()),plan_SHA256=sha(data))))

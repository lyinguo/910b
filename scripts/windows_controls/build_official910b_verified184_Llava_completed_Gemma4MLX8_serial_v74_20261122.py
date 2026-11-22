"""Prepare the original official heartbeat update from separate exact evidence.
No direct TOML mutation, scheduler workaround, new automation or model change.
"""
from pathlib import Path
import hashlib,json,tomllib
V=Path(__file__).resolve().parent
before=(V/'official_910b_before_Llava_Gemma_serial_v74_20261122.toml').read_bytes();old=tomllib.loads(before.decode('utf-8'))
assert old['id']=='910b' and old['kind']=='heartbeat' and old['status']=='ACTIVE' and old['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300' and old['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
receipt=json.loads((V/'R_verified184_LlavaMini_Gemma4MLX8_full_handoff_v74_actual_20261122.json').read_bytes());assert receipt['passed'] and receipt['source_count']==2715
plan=json.loads((V/'verified184_LlavaMini_Gemma4MLX8_incremental_handoff_plan_v74_20261122.json').read_bytes());base=plan['full_handoff_template'];h='/root/projects/comperror/workspace/diagnostics/batch_910b/heartbeat_handoffs/A100_verified183_AutoSingle_dual_GLM41Thinking_own8192_serial_current_20261120_031429_792677'
intro=old['prompt'].split('\n\n',1)[0]
mandatory='必须先完整读'+h+'/handoff.txt SHA19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1，再同目录facts.json SHA1c12f3f79a204be0c244a866df9fe7b94711053e25e1fd3259872650755c0cd7、closing_addendum.json、late_actual_notes和D/A100_ALIGNMENT_HANDOFF_CURRENT.json。还必须完整读本轮全文'+receipt['full_actual_note']+' SHA'+receipt['full_actual_note_SHA256']+'，并读同目录current_serial_real_process_acceptance_capacity_actual.json及更晚closing/pointer指定全文。原正文/facts是旧初快照，最新真实proc/内核FLOCK/actual退出/closing和用户决定优先。直接父完整008d90cc0b81d4563616fc8411392d95d474fcf0ae396cdd1da707a06a1100c7，全部祖先/65471字符archive的强制规范、家族失败、用户决定和精确清理例外完整继承，短引用不取消约束，不无限拼接、不存密码。当前2715原主源/config及原回执/本地原件逐SHA原字节归档，最新pending '+receipt['pending']+' SHA'+receipt['pending_SHA256']+'；更晚合法交接优先，不漏有用代码与后续提交文件。'
where=base[base.index('R=/root/projects/comperror'):base.index('【LLaVA-Mini真3')]
llava='LLaVA-Mini原真3/6/732源和完整数值NF1已过，原runner1仅stdout四JSON封装错误保持。01:56:23同genuine3正式接收0/proofbf232cfab3c8743b22a5a142716c1b4cd9637e7a774872f16330c27f4a08e375，NEW模型0/数值重放0；其后容量NameError原件留，新v4只补zipfile和未做容量，01:59:04全4050 ZIP与200形状经验3413199714710bytes约3.104TiB>实盘119.5GiB，未批准/NEWfresh3/200/G，不重跑已过3/NF或重压。原math/F16/plainRoPE/2048/原停止和所有失败/小样本/五件全留。详细源码、接收和容量SHA见本全文。'
gemma=base[base.index('【当前下一款自己的Gemma4'):base.index('【Ovis已完整通过')]
protection=base[base.index('【保护、代码与规范全部保持】'):base.index('【本轮更晚实际优先】')]
runtime='本轮正文和actual是快照，不拿旧live/state report_dir/modelrc1/NF0当正式200结束。只读helper已改为R/scripts/maintenance/read_Gemma4MLX8_formal_NF2_serial_recovery_v4_committed_trace_20261122_remote.py，按真实PID的已提交trace和完整原记录逐SHA读，遇正在追加的末行下轮增量补读，不把未完成文件误报失败。原真EX/model1692381/runner1691471/controller1691077都仅历史快照；确认实际退出再接下一款。全数值200/原协议资格/三D5%全过且全部NN/NF/worker/controller/EX退出后已授权只G664/state/reg/receipt备份CAS双书，不逐款再问；ArtifactTool必须真实author和实际PNG查看，原冻结writer11实表和完整targetaudit均过。超差保留空G及全部错例、invalid和分母，不能凑分。新完整200/G/资格尚未完成时四项仍169/15/206/310，完整资格70；三D小3差不是正式200结论。全保存后锁释放立即继续下一可推进项，单款卡住不全队等。'
prompt='\n\n'.join([intro,mandatory,where,llava,gemma,protection,runtime])
assert 'CURRENT_ACTUAL_JSON' not in prompt and len(prompt)<24000
args=dict(mode='update',id='910b',kind='heartbeat',name=old['name'],prompt=prompt,rrule=old['rrule'],status=old['status'],targetThreadId=old['target_thread_id'],notificationPolicy=None)
out=V/'official_910b_verified184_Llava_completed_Gemma4MLX8_serial_v74_requested_args_20261122.json';out.write_text(json.dumps(args,ensure_ascii=False,indent=2),encoding='utf-8')
meta=dict(args_path=str(out),args_SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),before_TOML_SHA256=hashlib.sha256(before).hexdigest(),prompt_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),prompt_characters=len(prompt),original_name_thread_ACTIVE_300minute_preserved=True,full_handoff_SHA256=receipt['full_actual_note_SHA256'])
(V/'official_910b_verified184_Gemma4MLX8_v74_requested_metadata_20261122.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False))

"""Preserve the existing heartbeat and record the user's optional-cache pause."""
from pathlib import Path
import hashlib,json,tomllib
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
raw=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes();old=tomllib.loads(raw.decode())
assert sha(raw)=='b46dda48ba321117ff99121301727d807c0b3d126c0c5177d3406aedee6fc509'
assert old['id']=='910b' and old['status']=='ACTIVE'
cleanup=json.loads((P/'R_user_optional_HTTP_cache_cleanup_actual_1412_20261119.json').read_bytes());assert cleanup['passed'] and cleanup['model_evidence_files_deleted']==0
first,tail=old['prompt'].split('\n\n',1)
note='''【最新用户要求：暂停不必要缓存，必须保存文件继续保护】用户原话“请问除了代码，和必须保存的文件，哪里的内存占用这么大呢？？为什么会运行运行就这样呢，对不必要的缓存暂时停止哇”。先区分RAM、NPU显存、磁盘，不能把Linux可回收文件页缓存误称RAM不足。本轮11月19日13:48实际RAM总2164370493440bytes、available2127357935616bytes约1.93TiB，没有RAM容量阻塞；项目盘free约92.45GiB，瓶颈是磁盘。完整只读路径/扩展名/实际分配stat证明workspace/diagnostics约692.64GiB，其中model_runs约633.71GiB，整诊断目录NPZ约567.86GiB，JSON约83.52GiB；这些是数值与跟踪/验收元数据，不等于普通缓存。每样本/层/KV/全头的全值捕获会落盘留存，模型退出仅释放RAM，不会自动删除这些历史证据。
实际已消费configs/retention/auxiliary_cache_pause_user_20261119.json，拥有的维护命令PYTHONDONTWRITEBYTECODE=1/PIP_NO_CACHE_DIR=1/UV_NO_CACHE=1，继续python -B。暂停重复CPU/NPU加载、重复压缩/存储试验等不能推进真实新模型所需门槛的额外工作；源实现与必要全模型适配继续，必要真3/fresh3/200/完整独验照原协议。不能关闭模型原use_cache/KV或数学/原独立数值捕获来冒充“暂停普通缓存”，不能清主机全局pagecache影响别人服务。
仅/root/.cache/pip/http-v2与selfcheck中312个已完整stat/SHA且无活FD的HTTP下载缓存，按planSHAd1350a1a3bb18eaec44824f81569e6eb31d15c60a365e14e7eb856a5f75750d1/真原EX精确删除，释放198504448bytes约189.3MiB；receiptSHAb05a1da28385fb1d4d9a40120862eb9a1b4bf220be9c5d8074dd7618f1b3d458，R D/user_optional_HTTP_cache_pause_exact_cleanup_1410_20261119/actual_cache_cleanup_receipt.json。所有模型报告/源/配置/日志/小样本/失败/未验/必要例证未删，numeric retention ledger及canonical SHA全同。其余~5GiB普通cache含UV环境包后备/模型资产/他人服务，不可当垃圾整删。该0.185GiB不足补串行Huihui仍约60GiB缺口，不能称正式200已恢复。保留现有两个无NN等候器，不重复、不热改，串行优先不变。原扩容/别盘/降guard/减少数值证据仍没有授权。
本轮详细空间证据和最新用户决定要完整归档在当前交接late_actual_notes，追加所有有用R源码/真实配置消费者与Windows读取/清理/官方维护实现到完整pending；只代码配置进入Day20审核，全部报告/数组/日志/回执留证据，不漏后续提交文件。11月19日已有唯一commit，不第二提交。新清理仅安装包HTTP缓存，不扩大旧数值清理例外，不改原policy/ledger/表/state/reg/活源。更晚真实状态优先，新增正式200/G/资格仍0，180交付/165数值内/15超差/210完整权重待交付/310确缺/66完整资格不变。'''
prompt=first+'\n\n'+note+'\n\n'+tail
args=dict(mode='update',id='910b',kind='heartbeat',name=old['name'],status=old['status'],rrule=old['rrule'],targetThreadId=old['target_thread_id'],prompt=prompt)
if 'notification_policy' in old:args['notificationPolicy']=old['notification_policy']
(P/'automation910b_before_optional_cache_pause_20261119.toml').write_bytes(raw)
(P/'automation910b_before_optional_cache_pause_prompt_20261119.txt').write_text(old['prompt'],encoding='utf8',newline='\n')
(P/'official910b_optional_cache_pause_arguments_1415_20261119.json').write_text(json.dumps(args,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print(json.dumps(dict(previous_TOML_SHA256=sha(raw),requested_prompt_SHA256=sha(prompt.encode()),prompt_chars=len(prompt),only_prompt_changes=True),ensure_ascii=False))

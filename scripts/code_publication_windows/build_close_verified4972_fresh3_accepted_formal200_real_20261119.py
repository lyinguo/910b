"""Archive accepted fresh3 and genuine live serial200 facts without touching a live job."""
from pathlib import Path
import ast, base64, hashlib, json

P = Path(__file__).resolve().parent
read = lambda n: json.loads((P / n).read_bytes())
sha = lambda b: hashlib.sha256(b).hexdigest()
prior = read('verified4972_complete_late_handoff_plan_20261119.json')
closed = read('R_close_verified4972_complete_sources_current_20261119.json')
live = read('R_user_Huihui_lossless_serial_v4_formal200_actual_samples_20261119.json')
accepted = live['stages']['fresh3']['accepted.json']
assert accepted['sample_count'] == 3 and accepted['new_real_NPU_inference'] is True
assert accepted['proof_sha256'] == '54dd893d73271c5d65d2cadef23e833426dabda367c4006b28a5a09d66943022'
samples = live['actual_samples']
assert len(samples) == 1 and samples[0]['completed_samples'] >= 8
assert samples[0]['committed_forwards'] >= 72 and not samples[0]['failed_records']
assert samples[0]['all_own_source_actual_NPU_exists']
assert live['original_job_and_live_sources_not_modified']
assert any(v['PID'] == 1565755 for v in live['current_original_FLOCK'])
R = '/root/projects/comperror'
D = R + '/workspace/diagnostics/batch_910b'
C = D + '/code_publication_source_layout_20261114'
B = D + '/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H4 = B + '/HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119'
J4 = B + '/HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
note = '''2026-11-19 15:37:37北京时间实际接续：用户授权重新审查已验收矩阵和缓存、保留完整五件/全部逐条记录/源码配置日志/必要数值例证，空间足够立即串行。
4972个精确非例证NPZ已清，实际释放47173816320bytes（43.935GiB）；15399保护文件SHA全部相同。另支持的离线UV无用缓存清理实际释放2969726976bytes，精确派生字节码/临时wheel清理df实际增加331231232bytes。旧完整proof和全部五件不改，精确pruned数组现在不可再验；以manifest f955add85f6012e7b1f8f4ad5fc0b78af043b80d2d0797315c21d59b35478382限定，不扩清。
ledger仅合法4ae138...→5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32，旧waiters因旧ledger自失败的原件保留。新私有v4精确绑定该单一合法transition，未清G/state/保护。v3批准schema在任何NN/锁前KeyError原件留，v4只在新私有批准补已验真实3依据所需字段，原冻结runner及独验不改。
Huihui自己的704BF16、2048、outer4内serial1、原thinkingFalse部署、EOS/全部数学/cache/head/位置/评分保持。标准Fortran NPY+ZIP-LZMA只无损存储，原3全部333数组每个raw位相等，模型非save_npz全部方法AST相同；预算146869927228bytes（136.78GiB），原factor4+6GiB及全部hard/log/precheck/sample guard不降。
新v4同runner NEWfresh3报告20261119_151046真实3/3、27前向，全部独立数值验收0，proof54dd893d73271c5d65d2cadef23e833426dabda367c4006b28a5a09d66943022。小score1.0对全D.945相对5.8201%仅小样本，不新增首次200门槛、不当正式200资格。
同一原runner1565755持真原EX，自动进入formal200报告20261119_152641，model1567344/PPid1565755，卡4。15:33:23正式200推理循环开始；15:37:37实查8/200完整提交、72前向、失败记录0、own source actualNPU证明已成，当前没有正式200完成/完整NF200/资格/G。尚在写第9个archive不是已提交第9条；提交样本数以原sample trace为准。活NN/NF/runner/源/job/YAML不可热改/信号/重启。
controller1565741同J全流程继续200后全数值独验，不等五小时。不得把state旧fresh3 report/modelrc1/verifier0当200退出；fresh3 model1仅恰完整五件baselinegap且NF0，原失败/警告都留。200/全NF/原资格/三D5%通过且所有NN/NF/协调器worker/真EX均释放后，仍授权原备份/CAS精确旧G/C唯一B一致的双书本款交付；未满足不得写G。
仍180交付、165数值内/15超差/210完整权重待正式交付/310确缺、原完整资格66（165=66完整+99仅数值）。本轮新完整200/G/资格0，原因新正式200仍在运行。最近完整200仍Prithiv11月17日16:57:59 .880，对三D.87均1.1494252873563%。当前停摆已由11月19日15:17:34真实fresh3NN恢复，15:33:23正式200真实NN继续，不能延用父停摆。
NPU0他人服务不动。当前用户优先串行效率max1，不为第二模型等总预算、不扩盘换盘/减guard。当前空余约137.89GiB，已足原门槛下当前单款；所有原家族失败/资源暂不可读跳过而不加310/历史决定/保留例外和完整强制规范继承原完整交接与closing。
双书/state/reg/PAUSE/POLICY/retention_policy/原runner/Gwriter等保持原SHA，仅上述合法ledger变更。Day19已push e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d，本地clean；今天不第二提交。v39完整pending和最新新增有用源码/真实配置依赖/全部原件用于Day20逐SHA用途消费者/diff/outgoing/检查/plan_SHA审核，不漏后续提交文件，不传权重数据数组报告日志凭据归档。
原910b ACTIVE/原名/原线程/每300分钟不变；当前official prompt是阶段快照，更晚本actual/proc/退出优先。完整五件/源码/YAML/日志/全部small/failed/unverified/必要首中末完整例证永久保留。全部资源款原协议真200/三D5%、确缺跳过、无pending与资格复审/双书全量复查前不关闭。
'''
(P / 'handoff_verified4972_fresh3_accepted_formal200_real_20261119.txt').write_text(note, encoding='utf8', newline='\n')
names = [
    'R_user_Huihui_lossless_serial_v4_fresh3_verifier_current2_20261119.json',
    'R_user_Huihui_lossless_serial_v4_formal200_live_current_20261119.json',
    'R_user_Huihui_lossless_serial_v4_formal200_actual_samples_20261119.json',
    'verified4972_local_workbook_and_codeDay19_fresh_readonly_20261119.json',
    'R_close_verified4972_complete_sources_current_20261119.json',
    'handoff_verified4972_fresh3_accepted_formal200_real_20261119.txt',
    Path(__file__).name,
]
packets = []
for n in names:
    data = (P / n).read_bytes()
    packets.append(dict(name=n, bytes=len(data), SHA256=sha(data), base64=base64.b64encode(data).decode()))
source_names = [Path(__file__).name]
source_data = []
for n in source_names:
    data = (P / n).read_bytes(); ast.parse(data)
    source_data.append(dict(relative='scripts/code_publication_windows/' + n, SHA256=sha(data), base64=base64.b64encode(data).decode(), role='Actual final closure consumer archiving accepted genuine fresh3 and real serial200 progress; immutable live model source and retention boundaries'))
stage_files = dict(prior['current_stage_files'])
stage_files['current_v4/fresh3_verification_stdout.json'] = J4 + '/fresh3/verification_stdout.json'
stage_files['current_v4/fresh3_accepted.json'] = J4 + '/fresh3/accepted.json'
stage_files['current_v4/formal200_current_run.log'] = R + '/workspace/reports/huihui-ai_Huihui-GLM-4.6V-Flash-abliterated/image-text-to-text/20261119_152641/run.log'
plan = dict(
    handoff_directory=closed['handoff_directory'], expected_pointer_SHA256=closed['pointer_SHA256'],
    canonical_SHA256=prior['canonical_SHA256'], note_name='verified4972_fresh3_NF0_formal200_real_serial_samples_current_20261119',
    local_original_packets=packets, official_readback=prior['official_readback'],
    additional_existing_primary_sources=[{k:v for k,v in x.items() if k!='base64'} for x in source_data],
    process_keyword='Huihui', current_stage_files=stage_files,
    new_pending=C + '/pending_after_daily_20261119_verified180_verified4972_fresh3_accepted_formal200_real_serial_all_primary_v39.json')
raw = (json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode()
(P / 'verified4972_fresh3_accepted_formal200_actual_late_plan_20261119.json').write_bytes(raw)
transport = dict(primary_sources=source_data, plan_base64=base64.b64encode(raw).decode(), plan_SHA256=sha(raw))
remote = 'DATA=' + repr(transport) + '\n' + '''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib
R=Path('/root/projects/comperror')
for row in DATA['primary_sources']:
 p=R/row['relative'];b=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==row['SHA256'];assert p.resolve().is_relative_to(R);ast.parse(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
raw=base64.b64decode(DATA['plan_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==DATA['plan_SHA256'];plan=json.loads(raw)
H=Path(plan['handoff_directory']);p=H/'verified4972_fresh3_accepted_formal200_actual_late_plan_20261119.json';assert not p.exists();p.write_bytes(raw)
sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.close_alignment_handoff_official_readback_current_owned_stage_v4_20261118')
print(json.dumps(m.execute(p,DATA['plan_SHA256']),ensure_ascii=False))
'''
ast.parse(remote)
(P / 'execute_close_verified4972_fresh3_accepted_formal200_actual_20261119_remote.py').write_text(remote, encoding='utf8', newline='\n')
print(json.dumps(dict(plan_SHA256=sha(raw), original_packets=len(packets), new_source_count=len(source_data), plan_bytes=len(raw)), ensure_ascii=False))

"""Create exact finite preservation inputs for the next official task continuation."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
prior=json.loads((V/'Day23_verified185_actual_commit_blocked_resource_full_handoff_plan_v90_20261123.json').read_bytes())
template=prior['full_handoff_template'];cut=template.index('【必须先读的完整继承】')
prefix=template[:cut].replace('完整当前交接v90：','完整当前交接v91：',1)
prefix=prefix.replace('本轮截至本snapshot只读TOML，后续如更新只能official并核实际readback归档，不能直接写TOML。',
    '本轮后来已通过official完成原910b仅prompt更新和实际readback，ACTIVE/名称/原线程/300分钟全同，完整before/args/toolresult/after/readback按原字节归档，不能直接写活跃TOML。')
extra='''【v91更晚实际：MedraN自己的原FP16失败只读源码核查已成，不是模型修复】
v90全文62226a97a8e3a86e9e9b17c5f0916c5aa23d14ef3c3919d9727a92488ca25e21/facts b2b5a43e08e59a855f0ad23a05ee4c56e616becac1ff61c2c372099d055d6990和pending0c4331b829b3f64534a16adf75e15217afbba9edc0a0899b7fcb89aec0502c4b完整继承；当前晚于v90，以本段/末actual/closing和真实proc/原EX/退出为准，不重复完成款。
新R/scripts/diagnostics/review_medran_original_FP16_graph_and_loader_source_contract_20261123.py SHAe0bc21686108b589d3edea7acb4e22036da39b782522437edeb241b9984c0a34，只读旧completion/review、自己的原保存YAML/graph绑定/adapter/vision/实际已装Transformers source，精确源快照与SHA保存。2026-11-23 02:22:43北京时间source-audit0，Hdiag=B/MedraN_completed_FP16_failure_native_source_contract_readonly_no_model_20261123/review.json SHA7c994a500f38445ebc1734ccb114d9678fc14b40881108210eddc78582d5a8a0，未加载任何权重数组，learnedCPUforward0/NPUmodel0/新真3/fullNF/200/G/资格0。自己的原FP16失败NPU与CPU参数化graph在vision_blocks.2非有限保持；这次明确旧CPU诊断重放的是同一参数化graph，不证明真实原HF runtime等价或原A100也失败。当前实际已装Transformers Gemma3n source SHA1b2d906b4de587328e7fcb8681b90edef7282d2a7e333ede73323fe848c43dcb，TimmWrapper source800c11e601fec0f261dd6c981c034ce5679ed479c2da5c456101a671d22b3588明确requires_backends vision+timm；真实timm包不存在，原graph注明的两个精确原fast_norm/mobilenetv5 source路径也不存在，不能装包或stub，不能据此猜keepFP32/eps根因、改BF16/eps/math/阈值/预算/提取或翻旧失败job。未实际构造/加载原HF，不把源码审计0充ready或恢复。后续只在原可证source基础上闭合，不借其它checkpoint/数据/图精度或盲重跑失败3。
本轮02:22:40冻结monitor完整读且source_errors/inspection_errors/newlog全空，原盘22341648384bytes约20.81GiB，全项目NN/NF/原EX皆无；原Bee v5等候器1696145和旧waiter1668102/cleanupwatcher1629242真实/proc+PPid/startticks/FD核在，均非模型NN/NF，不持原EX，不热改/信号/复制。最终actual更晚读数优先，最早当前停摆仍11月22日17:40:49.156504，不把源码审计或Git/单元测试充恢复。没有新可适清正确非例证或tmp，0删除/0新释放；185两书/state/reg/ledger等canonical均未写。
【原910b已official更新并核实际readback】2026-11-23 02:25:53北京时间通过唯一官方automation_update更新原910b，回执ACTIVE；实际TOML只prompt/model_prompt_sha256/updated_at变，原名跟进910B模型评测与规范检查/原聊天01a18bbd-8040-7d02-9004-beef491d836a/每300分钟完全同。actualTOML_SHA590801a94c1493e88f304f20f35bf5f31fd3a77e5548f346c735179ca0c52929，prompt_SHA4eb16f2db5bf80eb7004f0f5dd131e0336cf5da901655a64843509febada98a9；本地工作簿fresh SHA2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878、bytes6099931/mtime1795290387531571700/ctime1795290387529570400全同。提示引用有限完整v90再读更晚pointer，本v91是更晚完整交接，不需第二次提示更新形成循环，不写活TOML。全部本地before/args/officialresult/after/readback以及新useful R源码/维护builders逐字节SHA入CURRENT_SOURCE_COUNT完整pending；今日Day23已5370d46唯一commit，后续源码只Day24审核，不第二commit。

'''
full=prefix+extra+template[cut:]
assert full.count('CURRENT_ACTUAL_JSON')==1
mainname='preserve_verified185_Day23_MedraN_source_official_full_handoff_v91_20261123.py'
mapping={
 'scripts/diagnostics/review_medran_original_FP16_graph_and_loader_source_contract_20261123.py':'review_medran_original_FP16_graph_and_loader_source_contract_20261123.py',
 'scripts/code_publication_windows/build_official910b_verified185_Day23_commit_MedraN_source_audit_v91_requested_args_20261123.py':'build_official910b_verified185_Day23_commit_MedraN_source_audit_v91_requested_args_20261123.py',
 'scripts/code_publication_windows/verify_official910b_verified185_Day23_MedraN_source_v91_actual_readback_20261123.py':'verify_official910b_verified185_Day23_MedraN_source_v91_actual_readback_20261123.py',
 'scripts/maintenance/'+mainname:mainname,
 'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).name,
}
stage={n:(V/p).read_bytes() for n,p in mapping.items()}
for b in stage.values():ast.parse(b)
names=['official910b_before_verified185_Day23_MedraN_source_v91_20261123.toml',
 'official910b_verified185_Day23_MedraN_source_v91_requested_args_20261123.json',
 'official910b_verified185_Day23_MedraN_source_v91_official_tool_result_20261123.json',
 'official910b_after_verified185_Day23_MedraN_source_v91_20261123.toml',
 'official910b_verified185_Day23_MedraN_source_v91_actual_readback_receipt_20261123.json',
 'R_MedraN_completed_FP16_native_source_contract_no_model_20261123.json',
 'R_MedraN_completed_FP16_native_source_contract_no_model_20261123.json.stderr',
 'R_Day23_current_serial_monitor_actual_20261123_1824.json',
 'R_Day23_current_serial_monitor_actual_20261123_1824.json.stderr']
originals=[]
for name in names:
    b=(V/name).read_bytes();originals.append(dict(name=name,SHA256=sha(b),base64=base64.b64encode(b).decode()))
q=dict(full_handoff_template=full,new_primary_expected_SHA256={n:sha(b) for n,b in stage.items()},
       local_original_records=originals,
       official_actual_readback=json.loads((V/'official910b_verified185_Day23_MedraN_source_v91_actual_readback_receipt_20261123.json').read_bytes()),
       MedraN_source_audit=json.loads((V/'R_MedraN_completed_FP16_native_source_contract_no_model_20261123.json').read_bytes()))
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();plan='verified185_Day23_MedraN_source_official_full_handoff_plan_v91_20261123.json';(V/plan).write_bytes(qb)
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,os,subprocess\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n;p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(plan)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+mainname+"'),str(p),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(s);(V/'stage_verified185_Day23_MedraN_source_official_full_handoff_v91_20261123_remote.py').write_text(s,encoding='utf8')
print(json.dumps(dict(plan_SHA256=sha(qb),new_primary_count=len(stage),local_originals=len(originals),finite_full_template_chars=len(full))))

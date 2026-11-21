"""Incremental declared code/evidence archive and accurate current-stage handoff."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan)
 pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';old=pointer.read_bytes();assert hashlib.sha256(old).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(old);H=Path(ptr['handoff_directory'])
 assert sha(H/'handoff.txt')=='19ab73bfea3f560e9324a67667b046ea4d42200e4879ce53f1a11ed6141a87c1'
 canon=ptr['canonical_SHA256'];assert all(sha(p)==h for p,h in canon.items())
 parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']
 rows={x['path']:x for x in js(parent)['all_current_primary_source_and_true_consumer_configs']}
 assert all(sha(p)==v['SHA256'] for p,v in rows.items())
 note=H/'late_actual_notes'/q['late_note_name'];note.mkdir(exist_ok=False)
 (note/'previous_pointer.json').write_bytes(old);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 additions={x['relative']:x for x in q['new_primary_sources']}
 selfrel=str(Path(__file__).resolve().relative_to(R));additions[selfrel]=dict(relative=selfrel,SHA256=sha(__file__),role='actual incremental source archive and current GLM handoff closure consumer')
 for rel,v in additions.items():
  p=R/rel;assert p.resolve().is_relative_to(R) and '..' not in Path(rel).parts and rel.startswith(('scripts/','configs/','universal_eval/')) and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.md'] and sha(p)==v['SHA256']
  snap=note/'primary_source_snapshots'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes())
  rows[str(p)]=dict(path=str(p),relative=rel,SHA256=v['SHA256'],bytes=p.stat().st_size,snapshot=str(snap),role=v['role'])
 for v in q['original_records']:
  p=Path(v['path']);assert sha(p)==v['SHA256'];dest=note/'local_original_records'/v['name'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 reader=R/'scripts/maintenance/read_GLM41_reviewed_publication_and_next_Meta_actual_v2_exact_runpy_20261120.py'
 run=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr.decode();actual=json.loads(run.stdout)
 put(note/'latest_GLM41_actual.json',actual)
 current=[x for x in actual['GLM41_captures'] if x['current_model_PID_matches_capture']]
 nn=bool(actual['formal_model_NN_processes']);nf=bool(actual['formal_fullNF_processes'])
 stage='GLM41_own_original8192_same_approved_fixed200_fullNF' if nf else 'GLM41_own_original8192_same_approved_fixed200_NN' if nn else 'GLM41_same_approved_fixed200_real_completion_publication_check_required'
 assert sum(v['status']=='done' for v in js(D/'state.json')['models'].values())==183
 full='\n'.join([
  '本轮完整实际交接：GLM4.1 Thinking同一正式200、合格后双表窗口与下一款串行准备。',
  '本文件与真实/proc、内核原FLOCK和实际退出优先。原handoff.txt及全部祖先/65471字符archive的强制规范、家族失败、用户决定、精确清理例外完整继承；短引用不取消约束。',
  '当前仍verified183：183交付/168数值内/15超差/207完整权重待交付/310确缺，完整原协议资格69；本轮尚未新增完整200/G/资格。数值达标不等于完整原协议资格。',
  '同J job756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163，不重复原真3/新fresh3。真实进度快照：'+json.dumps(current,ensure_ascii=False),
  '真实阶段='+stage+'；原锁='+json.dumps(actual['original_FLOCK'],ensure_ascii=False)+'。模型/独验运行与无NN等待器必须区分。',
  '发布排队器v1因错误要求runner参数为job.json在取得锁/写表前拒绝，原源与错误日志全留。真实argv是任务目录。新v2只按真实argv/PPid/startticks绑定，9项隔离检查通过，PID1632189仅快照，排队同原EX；不改活源/任务/期限/信号，不第二NN。',
  '新私有GLM184发布器保留原frozen writer83134...、原runner58de6...和全部原数值/源SHA/资格/三D/备份/CAS/11隔离实表/ZIPmeta门槛，仅自身704源和完整3009部件容量/183→184计数/原锁同PID接续绑定；18项元数据拒绝检查通过。未执行发布，不能把准备/预览当同步。',
  '真实200+fullNF+原资格+server/local/保留原D三份5%全过，原runner/controller及全部NN/NF自然退出，才实际ArtifactTool作者/PNG实看、同一原EX跨双表CAS与fulltargetaudit。超差或失败保留G空并释放，绝不改数学/预算/thinking/提取/筛样本。',
  'Meta下一款原3/11/fullNF3已经通过，不重跑。旧v3等待器PID1620589仅快照，不热改或信号；新v4 PID1632610仅快照，只等GLM合格双表全审、原锁释放和旧v3自然精确0NN保护拒绝后，核唯一G696合法发布的四项canonical变更，保留Meta原D/906 BF16/40KV/2048/outer10/双BOS/3EOS/全数值和原NF，仅新私有参考SHA重绑后批准NEWfresh3/fixed200。当前不是Meta NN、无新J200。',
  '即时磁盘看护PID1629242每20秒，活NN/NF/原EX时不清；空闲且真实容量阻塞时立即触发精确无损回收，不等300分钟。已有75份完全同字节物理副本合并释放65732198400bytes，全部116路径/完整SHA仍在，未裁剪任何数值数组。本轮没有额外删除。',
  '原精确pruned271和全部祖先例外现在不能再全数组复验；原proof保留。完整五件、逐条input/output/raw/ref/error/token/指标统计、适配源码YAML配置日志、全部小样本/失败/未验值及必要完整例证必须保存。',
  '源与配置已逐SHA归档并登记完整pending，Day20已be09d9a8...唯一push，今日不第二commit；Day21全用途/消费者/依赖/freshSHA/完整diff/alloutgoing/相应检查/planSHA后唯一正常push，不漏有用代码。',
  '原910b ACTIVE/原名/原聊天/300分钟不变；用户要求串行max1、锁释放马上接续、停摆报错普通可见中文报告。NPU0他人服务不动。扩容未答不授权，不换盘/减guard/借权重数据/装包。',
  '更晚真实完成/新canonical/closing优先，已完成款不按旧live重启。'])+'\n'
 (note/'full_actual_GLM41_same200_reviewed_publication_and_next_Meta_serial_20261120.txt').write_text(full,encoding='utf-8')
 fullpath=note/'full_actual_GLM41_same200_reviewed_publication_and_next_Meta_serial_20261120.txt'
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=actual['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-20',no_second_same_day_commit=True,next_daily_review='Day21 complete useful source/config/Windows controls/dependencies/diff/outgoing/tests/planSHA review',handoff_directory=str(H)))
 updates=dict(UTC=actual['UTC'],latest_actual_note=str(fullpath),latest_actual_note_SHA256=sha(fullpath),latest_full_actual_summary=str(fullpath),latest_full_actual_summary_SHA256=sha(fullpath),latest_actual_snapshot=str(note/'latest_GLM41_actual.json'),latest_actual_snapshot_SHA256=sha(note/'latest_GLM41_actual.json'),current_source_count=len(rows),current_primary_source_count=len(rows),current_model_NN_or_NF_active=nn or nf,current_actual_model_NN_or_NF_active=nn or nf,current_actual_model_process_or_NF_active=nn or nf,current_NN_or_NF_active=nn or nf,current_actual_stage=stage,current_formal_model_stage=stage,current_owned_candidate='zai-org/GLM-4.1V-9B-Thinking',current_current_owned_candidate='zai-org/GLM-4.1V-9B-Thinking',current_actual_job=str(J),current_actual_controller=str(B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'),current_formal_acceptance_home=str(J/'formal200'),severity_first_line=None if nn or nf else '严重告警｜需核实正式200退出与发布窗口',first_proven_stoppage_BJT=None,first_proven_current_formal_stoppage_BJT=None,new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending))
 close=js(H/'closing_addendum.json');close.update(updates)
 assert pointer.read_bytes()==old and all(sha(p)==h for p,h in canon.items()) and all(sha(p)==v['SHA256'] for p,v in rows.items())
 put(H/'closing_addendum.json',close);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 result=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),source_count=len(rows),pending=str(pending),pending_SHA256=sha(pending),full_actual_note=str(fullpath),full_actual_note_SHA256=sha(fullpath),canonical_and_all_live_model_sources_unchanged=True)
 put(note/'closing_actual_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

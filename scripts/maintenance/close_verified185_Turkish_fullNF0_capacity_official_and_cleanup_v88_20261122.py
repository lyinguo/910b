"""Append exact completed-stage, cleanup and official receipts without rewriting accepted evidence."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, os, subprocess
R = Path('/root/projects/comperror'); D = R/'workspace/diagnostics/batch_910b'
C = D/'code_publication_source_layout_20261114'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def js(p): return json.loads(Path(p).read_bytes())
def put(p, x):
    p = Path(p); t = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf8'); os.replace(t,p)

def main():
    p = Path(sys.argv[1]); assert sha(p)==sys.argv[2]; q=js(p)
    pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'; pb=pointer.read_bytes()
    assert hashlib.sha256(pb).hexdigest()=='d80837dd904a0a0d22c166f2cb662b52ab1ff37705eb6da2be7d7203d5277790'
    ptr=json.loads(pb); closing=Path(ptr['handoff_directory'])/'closing_addendum.json'; cb=closing.read_bytes()
    assert hashlib.sha256(cb).hexdigest()=='68b5bf24388887aed46973f9eedacd5494d067941c36691c753389575aa7c0bd'
    parent=Path(ptr['new_primary_pending']); assert sha(parent)=='35777b97e7c5242dac244c813c6b7cbdc841318230b625bbe8d16aa86b870a48'
    rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows)==2942 and all(sha(n)==r['SHA256'] for n,r in rows.items())
    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
    note=Path(ptr['handoff_directory'])/'late_actual_notes'/'verified185_Turkish_completed_fullNF0_capacity_official_actual_v88_20261122'
    note.mkdir(exist_ok=False); (note/'previous_pointer.json').write_bytes(pb); (note/'previous_closing_addendum.json').write_bytes(cb)
    local=[]
    for z in q['local_original_records']:
        b=base64.b64decode(z['base64'],validate=True); assert hashlib.sha256(b).hexdigest()==z['SHA256']
        dest=note/'local_original_records'/z['name']; assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(b)
        local.append(dict(path=str(dest),SHA256=z['SHA256'],bytes=len(b)))
    for n,h in q['new_primary_expected_SHA256'].items():
        p=R/n; assert n.startswith(('scripts/','configs/','universal_eval/')) and p.resolve().is_relative_to(R) and sha(p)==h,n
        if str(p) in rows: assert rows[str(p)]['SHA256']==h
        else:
            dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='Actual completed Turkish fullNF/capacity/cleanup metadata read-only and official original heartbeat maintenance controls, true consumers and full source preservation')
    reader=R/'scripts/maintenance/read_TurkishLlava_same_true3_fullNF_capacity_actual_v8_20261122_remote.py'
    r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode==0,r.stderr;actual=json.loads(r.stdout);x=actual['all_original_and_current_actual']['same_GLM41Base_only_actual_serial_model']
    assert x['cards_returncode']==0 and not x['frozen_monitor_summary']['source_errors']
    snapshot=note/'completed_Turkish_fullNF_capacity_current_actual.json';put(snapshot,actual)
    records=actual['Turkish_serial_v2_and_Base_CPU_diagnostic_complete_records']
    td=[z['complete_value'] for k,z in records.items() if k.endswith('TurkishLlava_own_original_Transformers_FP16_2048_true3_v2_after_known_Base_visual_failure_real_FLOCK_20261122/completion.json')]
    assert len(td)==1 and td[0]['actual_returncode']==0
    accepted,=td[0]['accepted'];assert accepted['sample_count']==3 and accepted['proof_sha256']=='cf89fde1ab3067eba798cde73d0cd37f2798bc21d283b86c0bcabd4e6086a4a1'
    assert sha(accepted['proof'])==accepted['proof_sha256']
    proof=js(accepted['proof']);assert proof['passed'] and proof['actual_forward_count']==6
    assert not proof['original_protocol_final_qualification_complete']
    capname='TurkishLlava_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_after_known_Base_visual_failure_20261122/own_passed3_complete_all200_shape_capacity.json'
    caps=[(p,z) for p,z in records.items() if p.endswith(capname)];assert len(caps)==1
    capfile,caprec=caps[0];cap=caprec['complete_value']
    assert caprec['SHA256']=='a029a47b242df3e0b411e501a1f79a5cef4f604d77c6633dcce9230f15d22605'
    assert cap['empirical_estimate_bytes']==4228496264073 and not cap['capacity_passed']
    discovery=Path(q['cleanup_discovery']);assert sha(discovery)==q['cleanup_discovery_SHA256'];clean=js(discovery)
    assert clean['candidate_count']==0 and clean['candidate_bytes']==0 and clean['no_deletion'] and not clean['temporary_suffix_files_preserved_pending_actual_scope_verification']
    for z in [Path(accepted['proof']),Path(capfile),discovery]:
        dst=note/'complete_current_original_records'/z.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read_bytes())
    states={p:z['complete_value'] for p,z in records.items() if p.endswith(('completion.json','actual_completion.json','state.json'))}
    procs=x['actual_proc_argv_cwd_PPid_startticks_FD_io']
    models=[p for p in procs if 'run.py' in p['argv']]
    nf=[p for p in procs if any(Path(a).name.startswith('verify_') for a in p['argv'])]
    compact=dict(UTC=actual['UTC'],counts=dict(delivered=185,numeric_within=170,numeric_outside=15,complete_weights_pending=205,confirmed_missing=310,full_original_qualification=71),
        Turkish_completed_genuine3=dict(report_dir=proof['report_dir'],sample_count=3,native_forwards=6,fullNF=0,proof=str(accepted['proof']),proof_SHA256=accepted['proof_sha256'],runner_completion=td[0]),
        Turkish200_capacity=dict(path=capfile,SHA256=caprec['SHA256'],required_bytes=cap['empirical_estimate_bytes'],actual_free_at_review=cap['actual_free_bytes'],capacity_passed=False,no_approval_fresh3_200_G=True),
        cleanup=dict(discovery=str(discovery),SHA256=sha(discovery),candidate_count=0,candidate_bytes=0,temporary_count=0,actual_deletions=0),
        official_actual_readback=q['official_actual_readback'],full_primary_source_count=len(rows),parent_pending=str(parent),parent_pending_SHA256=sha(parent),
        current_model_processes=models,current_NF_processes=nf,current_original_EX=x['actual_original_lock_owners'],actual_free_bytes=x['actual_original_disk_free_bytes'],
        complete_monitor_summary=x['frozen_monitor_summary'],current_states=states,full_actual_snapshot=str(snapshot),full_actual_snapshot_SHA256=sha(snapshot),
        local_original_records=local,canonical_SHA256=ptr['canonical_SHA256'],no_new_complete200_G_or_qualification=True,no_deletion=True,no_second_Day22_commit=True)
    facts=note/'facts.json';put(facts,compact)
    text='''本轮正式已完成阶段及官方维护晚注v88，覆盖v87初正文所有旧live。仍verified185/170数值内15超差205完整权重待310确缺/71完整原资格；本轮新完整200/G/资格0。
Turkish-LLaVA同J2真正3条6前向、完整五件与全数值独立验收已在2026-11-22 17:40:38.937624北京时间完成0，proofcf89fde1ab3067eba798cde73d0cd37f2798bc21d283b86c0bcabd4e6086a4a1。自己的686BF16存储→原请求F16、原head128258/soleEOS128009/2048/无system-date/全部原数学及8034完整数值发生、32物理KV/每步fullhead/source-cast-actualNPU及全部externalSHA过。Accuracy1.0只有3条，三D.705差41.84397%只是小样本披露，不是完整200差或资格。三个12.56GB完整小样本是必要例证不能清。
同controller17:40:49.156504 actual1仅200容量拒，全200自己原形状/全部ZIP完整部件/原factor4+6GiB与所有guard不降，要求4228496264073bytes约3.85TiB，实际22721392640bytes约21.16GiB，无200批准/NEWfresh3/200/G，所有原NN/NF/controller/原EX自然退出，旧state running不得当live或重启。当前正式停摆从此时按当前钟计，真实恢复才截断；原Base视觉NF失败不接受，CPU诊断0不等NF0，所有原值/失败/源码配置日志都保持。
已再次只读复核当前185基线已验收完整200的非例证与临时suffix：新可清候选0/tmp0，没有删除。旧pruned例外不能重复全数组复验，未显式完整原协议声明的旧proof不拿来贸然删；wrong/invalid/trunc/小样本/失败/未验/必要原例证、所有完整报告input-output-raw-ref-error-token/源码/config/log永保。不能声称已释放新空间。现有清理watcher仅全字节等值物理hardlink去重。当前等待容量的BeeSFT预算192555984252bytes约179.33GiB，约21GiB不足；已异步请求原项目盘再增至少200GiB可用空间，尚未答复不算批准或已扩容，不擅换盘/扩容/降guard/流式剪证据。串行max1/0卡外部服务不动，不等AA，不用已过3凑运行。
原910b已仅通过official更新并实际回读ACTIVE/原名称/原thread/300分钟均同，只prompt/model_prompt_sha256/updated_at变化；本地fresh workbook仍canonical185 SHA2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878，无表写。新官方before/args/toolresult/after/readback原字节、只读metadata清理源/Windows消费者及全部有用主实现都入最新明确pending，Day22已唯一43a149bb...不二次提交，Day23逐用途/消费者/依赖/freshSHA/完整diff-alloutgoing/tests/实际publisher计划SHA审核后正常push核remote，不漏源，不上传报告权重数据数组日志凭据/归档。
完整规范继续以v87完整全文及同facts、根handoff/facts、全部祖先/65471archive继承；本晚注只准确覆盖完成时点/容量/官方回执，不取消任何家族失败、用户决定或精确清理例外，不无限拼接，不存密码。继续资源真实可行的原款；所有NN/NF/worker/controller/原EX自然释放和完整200+原资格+三D5%全过才备份/ArtifactTool实际author+PNG实看/原writer11/真原EX跨双书精确CAS/全targetaudit新186。
完整当前facts SHA CURRENT_FACTS_SHA256。真实proc/原内核EX/实际退出及更晚closing优先，停摆首行严重阻塞、关键检查连接失败严重告警，普通中文说明模型阶段/样本/前向/卡/真实原因/最近完整与合格结果北京时间/最早可证持续时长/解除动作；恢复当轮也报，磁盘NOTIFY不充聊天通知。
CURRENT_ACTUAL_JSON
'''
    full=note/'late_actual_Turkish_fullNF0_capacity_official_cleanup_v88_20261122.txt'
    full.write_text(text.replace('CURRENT_FACTS_SHA256',sha(facts)).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False)),encoding='utf8')
    pending=C/'pending_after_daily_20261122_verified185_Turkish_completed_official_cleanup_all_primary_v88_20261122.json'
    assert not pending.exists();put(pending,dict(UTC=actual['UTC'],parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[n] for n in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',no_second_same_day_commit=True,next_daily_review='Day23 explicit useful main sources, true config consumers and dependencies, full diff/outgoing and tests plus real publisher plan_SHA, one ordinary push then verify remote'))
    upd=dict(UTC=actual['UTC'],new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_primary_source_count=len(rows),current_source_count=len(rows),latest_late_actual_note=str(full),latest_late_actual_note_SHA256=sha(full),latest_actual_snapshot=str(snapshot),latest_actual_snapshot_SHA256=sha(snapshot),current_completed_stage_facts=str(facts),current_completed_stage_facts_SHA256=sha(facts),current_model_process_active=bool(models),current_full_numerical_verifier_active=bool(nf),current_model_NN_or_NF_active=bool(models or nf),current_actual_model_NN_or_NF_active=bool(models or nf),current_NN_or_NF_active=bool(models or nf),current_formal_model_task_active=bool(models or nf),current_actual_model_process_or_NF_active=bool(models or nf),current_next_original_model_stage='Turkish same genuine3 fullNF0 completed; fixed200 capacity refused, do not replay. Actual original disk resource required; no active NN/NF unless later real process proves recovery',current_Turkish_fullNF0_proof_SHA256=accepted['proof_sha256'],current_Turkish_capacity=compact['Turkish200_capacity'],current_cleanup_readonly_actual=compact['cleanup'],current_official_actual_readback=q['official_actual_readback'],current_actual_model_processes=models,current_actual_NF_processes=nf,current_original_EX=x['actual_original_lock_owners'],current_resource_stop_earliest_UTC='2026-11-22T09:40:49.156504+00:00')
    assert pointer.read_bytes()==pb and closing.read_bytes()==cb
    assert all(sha(n)==h for n,h in ptr['canonical_SHA256'].items()) and all(sha(n)==r['SHA256'] for n,r in rows.items())
    co=json.loads(cb);co.update(upd);put(closing,co);ptr.update(upd);ptr['closing_addendum_SHA256']=sha(closing);put(pointer,ptr)
    result=dict(passed=True,UTC=actual['UTC'],pointer_SHA256=sha(pointer),closing_SHA256=sha(closing),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),late_actual_note=str(full),late_actual_note_SHA256=sha(full),facts=str(facts),facts_SHA256=sha(facts),actual_snapshot_SHA256=sha(snapshot),official_actual_readback=q['official_actual_readback'],genuine3_fullNF0=True,capacity_passed=False,models=models,full_NF=nf,EX=x['actual_original_lock_owners'],actual_free_bytes=x['actual_original_disk_free_bytes'],monitor_summary=x['frozen_monitor_summary'],no_deletion=True,no_G_or_canonical_mutation=True,no_second_Day22_commit=True)
    put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()

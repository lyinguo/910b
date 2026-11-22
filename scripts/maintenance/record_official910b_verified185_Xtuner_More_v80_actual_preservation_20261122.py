"""Append official original heartbeat readback and all actual useful control sources."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, os, subprocess
R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
C = D / 'code_publication_source_layout_20261114'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def js(p):
    return json.loads(Path(p).read_bytes())

def put(p, value):
    t = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(t, p)

def main():
    plan = Path(sys.argv[1]); assert sha(plan) == sys.argv[2]
    q = js(plan)
    pointer = D / 'A100_ALIGNMENT_HANDOFF_CURRENT.json'
    before = pointer.read_bytes(); assert hashlib.sha256(before).hexdigest() == q['expected_pointer_SHA256']
    ptr = json.loads(before); H = Path(ptr['handoff_directory'])
    cp = H / 'closing_addendum.json'; cb = cp.read_bytes()
    assert hashlib.sha256(cb).hexdigest() == q['expected_closing_SHA256']
    parent = Path(ptr['new_primary_pending'])
    assert sha(parent) == ptr['new_primary_pending_SHA256'] == q['expected_pending_SHA256']
    assert ptr['current_verified_count'] == 185
    assert all(sha(n) == h for n, h in ptr['canonical_SHA256'].items())
    rows = {r['path']: r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']}
    assert len(rows) == 2811 and all(sha(n) == r['SHA256'] for n, r in rows.items())
    note = H / 'late_actual_notes' / q['round_name']; note.mkdir(exist_ok=False)
    (note / 'previous_pointer.json').write_bytes(before)
    (note / 'previous_closing_addendum.json').write_bytes(cb)
    for rec in q['local_original_records']:
        raw = base64.b64decode(rec['base64'], validate=True)
        assert hashlib.sha256(raw).hexdigest() == rec['SHA256']
        dest = note / 'local_original_records' / rec['name']
        assert dest.resolve().is_relative_to(note.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
    actualread = js(note / 'local_original_records/official_910b_verified185_Xtuner_More_v79_actual_readback_20261122.json')
    assert actualread['passed'] and actualread['all_original_nonprompt_fields_unchanged']
    assert actualread['id'] == '910b' and actualread['status'] == 'ACTIVE'
    assert actualread['prompt_exact_requested_bytes']
    assert actualread['prompt_SHA256'] == '284e9111ac6e54ba09d2c6186738e0da37a1c6d36e8103e6965fdbf033ee8fd4'
    assert actualread['actual_TOML_SHA256'] == 'ec8db0c7c8f213a2bcf8535aed75c149a5329b33c1ab2f259e893938f862a4e4'
    for n in sorted(set(q['new_primary_paths'] + [str(Path(__file__).relative_to(R))])):
        p = R / n; assert p.is_file() and p.resolve().is_relative_to(R)
        assert n.startswith(('scripts/', 'configs/', 'universal_eval/', 'tests/'))
        h = sha(p)
        if str(p) in rows:
            assert rows[str(p)]['SHA256'] == h
        else:
            dest = note / 'primary_source_snapshots' / n
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
            rows[str(p)] = dict(path=str(p), relative=n, SHA256=h, bytes=p.stat().st_size, snapshot=str(dest),
                role='Original official 910b exact prompt and nonprompt readback, useful Windows builder, immutable active serial model and source preservation')
    reader = R / 'scripts/maintenance/read_verified185_Xtuner_LlavaMore_serial_ready_frozen_monitor_actual_20261122_remote.py'
    r = subprocess.run(['python3', '-B', str(reader)], cwd=R, capture_output=True, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode == 0, r.stderr
    current = json.loads(r.stdout); a = current['current_Xtuner_same_serial_actual']
    assert a['npu_smi']['returncode'] == 0
    assert current['all_current_canonical_fresh_SHA256'] == ptr['canonical_SHA256']
    mon = next(v for k, v in current['complete_monitor_latest_checkpoint'].items() if k.endswith('/latest.json'))
    assert not mon['source_errors'] and not mon.get('new_log_error_candidates')
    owners = current['actual_original_lock_owner_proc']
    ids = {o['owner']['PID'] for o in owners if 'owner' in o}
    specific = [e for e in mon.get('inspection_errors', []) if e.get('reason') == 'runner_lock_owner_not_in_project_process_snapshot']
    closed = bool(specific and all({v['PID'] for v in e['owners']}.issubset(ids) for e in specific) and all(
        o.get('cwd') == '/root' and len(o.get('argv', [])) == 4 and
        Path(o['argv'][2]).name == 'A100_alignment_evidence_runner_20261103.py' and
        o['argv'][3].endswith('/XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122')
        for o in owners))
    unknown = [e for e in mon.get('inspection_errors', []) if e not in specific] if closed else mon.get('inspection_errors', [])
    assert not unknown, unknown
    model = [p for p in a['actual_processes'] if 'run.py' in p['argv']]
    nf = [p for p in a['actual_processes'] if any(Path(z).name.startswith('verify_') for z in p['argv'])]
    cap = [{k: c.get(k) for k in ['home', 'committed_step_counts', 'total_committed_native_forwards', 'last_committed_step', 'pending_partial_line', 'NPZ']} for c in a['captures']]
    forwards = sum(c['total_committed_native_forwards'] for c in cap)
    last = max((c['last_committed_step']['UTC'] for c in cap if c.get('last_committed_step')), default=None)
    age = (datetime.fromisoformat(a['UTC']) - datetime.fromisoformat(last)).total_seconds() if last else None
    submitting = bool(model and age is not None and age < 10)
    fds = [{ 'PID': p['PID'], 'fd': {k: v for k, v in p.get('fd', {}).items() if v.endswith('.npz')}} for p in model]
    fds = [v for v in fds if v['fd']]
    writing = bool(model and fds and not submitting)
    current['actual_official_original_910b_readback'] = actualread
    current['current_exact_root_cwd_owner_check_closed'] = closed
    current['current_unknown_inspection_errors'] = unknown
    current['current_recent_NPU_forward_submission_active'] = submitting
    current['current_full_numeric_persistence_not_new_NN_or_NF'] = writing
    current['current_numeric_persistence_open_FDs'] = fds
    snapshot = note / 'official_original910b_readback_Xtuner_same_v2_More_waiter_full_actual.json'
    put(snapshot, current)
    compact = dict(UTC=current['UTC'], current_captures=cap, current_committed_forwards=forwards,
        actual_free_bytes=a['free_bytes'], actual_original_lock_owners=owners,
        current_model_task_active=bool(model), current_fullNF_active=bool(nf),
        recent_forward_submission_active=submitting, numeric_persistence_not_new_NN_or_NF=writing,
        persistence_open_FDs=fds, last_step_age_seconds=age,
        More_actual={k: v for k, v in current['LlavaMore_new_source_ready_and_serial_waiter'].items() if k.endswith(('state.json', 'actual_completion.json', 'actual_failure.json'))},
        current_specific_root_cwd_owner_check_closed=closed, unknown_inspection_errors=unknown,
        source_errors=mon['source_errors'], new_log_error_candidates=mon.get('new_log_error_candidates', []),
        official_actual_readback=actualread, snapshot=str(snapshot), snapshot_SHA256=sha(snapshot))
    text = '''更晚实际v80：必须先完整读父v79全文5e1dacec7b6921278fcefb761e38df7f15416d54bdcf18bb65dc2aaa00890e01；2811原源及全部祖先规范/家族失败/用户决定/65471字符archive和精确pruned例外完整继承，本短晚注不取消全文。不拼接无限历史、不存密码。
原910b已通过official工具实际更新，原名称/ACTIVE/原thread/300分钟/通知字段保持，未建新自动化或写活TOML。实际prompt SHA284e9111ac6e54ba09d2c6186738e0da37a1c6d36e8103e6965fdbf033ee8fd4/11342字符与请求完全同字节，本次无末尾LF差异。actualTOML SHAec8db0c7c8f213a2bcf8535aed75c149a5329b33c1ab2f259e893938f862a4e4，before3506df...原字节/requested args/实际tool回执/只读after/readback源完整保留。全部非prompt设置和本地185工作簿SHA/stat同，不第二Day22commit。
同Xtuner v2原J3 f422.../NPU4正在按原要求保存完整数值；真实committed步、所有NPZ stat、实际proc/PPid/startticks/FD/io/原内核EX/currentattempt/实际退出及More接续器见下完整快照。数组压缩保存不是新NN/NF，仍是同一模型任务，不能把过去2048步当当前正在新前向或把等待器当模型。第1条完整NPZ14518613628bytes，后续值仍增长，不热改原writer或活source/job/YAML/deadline/signal，不重复旧v1/v2或已过front/CPUconstructor。若已转第三条/fullNF，以当前实际优先。
More准备真实0、自己的原概率一次前向协议/full686源/无cache/全128257头/独立NF/单YAML/标准完整NPZ CPU并行7测试与own3字节同均已原字节归档。More接续器1700499仅snapshot，等前Xtuner actual完成和全部NN-NF-controller-原EX释放自动接自己的NEW3；如果前款完整200合格则先等双表发布，合法canonical变化需新私有精确重绑定不能清保护。不要重复More控制器、也不要把ready或每20秒poll说成已NN。原Bee资源等待器不占原EX，179.33GiB预算及全部guard保持，不擅扩容/换盘/降guard。
当前仍185交付/170数值内/15超差/205完整权重待正式交付/310确缺/71完整原资格。本次新完整200/G/资格0，没有写G/state/reg/双表或删除任何值。Gemma .840对三D.8350/.8350/.835相对.5988023952%为本日更早已交付，不再计新增；最近完整200全NF03:34:24.274295/双表审03:47:22.926705北京时间。全报告/全部逐样本明细/源配置快照日志/错invalidtruncerror/小样本失败未验/必要首中末完整值保持，137和祖先pruned不可全数组复验但原proof永久留。
本轮所有新有用实现和Windows参数构建/实际只读校验/维护源精确pending Day23，不漏自动YAML/protocol/smoke、真实consumer/依赖/测试；opaque transport载荷只证据不当Git源。原冻结monitor rootcwd owner过滤只本次实证闭合，未知错误不套旧解释，原warning不抹不改monitor。严重停摆或关键检查异常及时可见中文，写真实阶段/样本前向/卡/最近完整结果/最早可证时长和动作，恢复也报，不拿CPU压缩当新NN恢复。
完整当前摘要：
''' + json.dumps(compact, ensure_ascii=False)
    late = note / 'late_actual_official_readback_and_same_Xtuner_More_serial_v80_20261122.txt'
    late.write_text(text, encoding='utf8')
    pending = C / q['pending_name']; assert not pending.exists()
    put(pending, dict(UTC=datetime.now(timezone.utc).isoformat(), parent_pending=str(parent), parent_pending_SHA256=sha(parent),
        source_count=len(rows), all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],
        Code_Day_already_pushed='2026-11-22', Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',
        no_second_same_day_commit=True, next_daily_review='Day23 all explicit useful sources, real consumers/dependencies/tests/Windows controls; fresh SHA/full diff/all outgoing/checks and reviewed publisher plan before one normal push'))
    updates = dict(UTC=datetime.now(timezone.utc).isoformat(), new_primary_pending=str(pending),
        new_primary_pending_SHA256=sha(pending), latest_pending=str(pending), latest_pending_SHA256=sha(pending),
        current_primary_source_count=len(rows), current_source_count=len(rows), latest_actual_snapshot=str(snapshot),
        latest_actual_snapshot_SHA256=sha(snapshot), latest_late_actual_note=str(late), latest_late_actual_note_SHA256=sha(late),
        official_910b_actual_readback=str(note / 'local_original_records/official_910b_verified185_Xtuner_More_v79_actual_readback_20261122.json'),
        official_910b_actual_prompt_SHA256=actualread['prompt_SHA256'], official_910b_actual_TOML_SHA256=actualread['actual_TOML_SHA256'],
        official_910b_active_name_thread_cadence_preserved=True, current_genuine_committed_native_forwards=forwards,
        current_model_process_active=bool(model), current_full_numerical_verifier_active=bool(nf),
        current_formal_model_task_active=bool(model or nf), current_recent_NPU_forward_submission_active=submitting,
        current_complete_numeric_persistence_active_not_new_NN_or_NF=writing,
        current_model_NN_or_NF_active=bool(submitting or nf), current_actual_model_NN_or_NF_active=bool(submitting or nf),
        current_NN_or_NF_active=bool(submitting or nf), current_actual_model_process_or_NF_active=bool(model or nf),
        current_inspection_owner_specific_root_cwd_filter_closed=closed,
        current_actual_stage='Xtuner_same_private_v2_serial_actual_steps_complete_numeric_persistence_More_source_ready_waiter_official_readback')
    assert pointer.read_bytes() == before and cp.read_bytes() == cb
    assert all(sha(n) == h for n, h in ptr['canonical_SHA256'].items())
    assert all(sha(n) == row['SHA256'] for n, row in rows.items())
    closing = json.loads(cb); closing.update(updates); put(cp, closing)
    ptr.update(updates); ptr['closing_addendum_SHA256'] = sha(cp); put(pointer, ptr)
    receipt = dict(passed=True, UTC=datetime.now(timezone.utc).isoformat(), pointer_SHA256=sha(pointer),
        closing_SHA256=sha(cp), pending=str(pending), pending_SHA256=sha(pending), source_count=len(rows),
        late_actual_note=str(late), late_actual_note_SHA256=sha(late), actual_snapshot=str(snapshot), actual_snapshot_SHA256=sha(snapshot),
        official_actual_prompt_SHA256=actualread['prompt_SHA256'], official_actual_TOML_SHA256=actualread['actual_TOML_SHA256'],
        current_committed_native_forwards=forwards, current_captures=cap, current_model_task_active=bool(model),
        current_fullNF_active=bool(nf), numeric_persistence_not_new_NN_or_NF=writing, unknown_inspection_errors=unknown,
        no_active_model_source_job_config_deadline_signal_mutation=True, no_canonical_mutation=True,
        no_deletion=True, no_second_Day22_commit=True)
    put(note / 'official_readback_and_current_serial_preservation_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()

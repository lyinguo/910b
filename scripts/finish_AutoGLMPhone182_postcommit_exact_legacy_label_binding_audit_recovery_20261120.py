import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timezone
from collections import Counter
import os,json,hashlib,base64,zipfile,re,xml.etree.ElementTree as ET,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120';I=B/'AutoGLMPhone9B_verified182_reviewed_single_lock_dual_publication_20261120';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import atomic_json,workbook_rows
from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
cache={}
def read(p):
 p=Path(p);v=json.loads(p.read_bytes())
 if p==H/'publication_receipt.json':
  assert v['old180_state_registry_and_G_preserved'] and v['canonical_counts']=={'done':182,'skipped_no_complete_weights':310,'needs_adaptation':208}
  old=json.loads((H/'before_verified_write/state.json').read_bytes());assert sum(x['status']=='done' for x in old['models'].values())==181
  v=dict(v,old181_state_registry_and_G_preserved=True)
 return v
def sha(p):
    p = Path(p)
    s = p.stat()
    k = (str(p), s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if k not in cache:
        with p.open('rb') as f:
            cache[k] = hashlib.file_digest(f, 'sha256').hexdigest()
    return cache[k]

def now():
    return datetime.now(timezone.utc).isoformat()

def glm46_capacity_review_passed(cap):
    bound = cap.get('all_complete_original_values_still_available_and_SHA_bound', {})
    return cap.get('passed') is True and cap.get('capacity_passed') is True and (cap.get('model_id') == 'zai-org/AutoGLM-Phone-9B-Multilingual') and (cap.get('empirical_estimate_bytes') == 83391788264) and (cap.get('actual_free_bytes', 0) >= 83391788264) and (cap.get('original_factor') == 4) and (cap.get('original_extra_reserve_bytes') == 6 * 1024 ** 3) and (cap.get('all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved') is True) and (cap.get('all200_original_shapes_used_not_only_200_div3') is True) and (cap.get('no_tensor_value_or_occurrence_or_model_parameter_changed') is True) and (cap.get('all200_inputs_are_this_models_own_original_frontend') is True) and (cap.get('no_NN_or_compression_replay_in_capacity_review') is True) and isinstance(bound, dict) and (len(bound) == 7) and (len([n for n in bound if n.endswith('.npz')]) == 3) and (bound.get('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/AutoGLMPhone9B_own_original_BF16_2048_true3_v3_actual_published_GLM_fixed_input_set_real_FLOCK_20261119/fresh3/verification_stdout.json') == '9676feeab2999d51ddac07d4900ec1947342795fba8db322b694de589efd89c3') and (len(cap.get('all_complete_ZIP_parts_individually_classified', [])) == 261) and (len(cap.get('all_original200_prompt_lengths', [])) == len(cap.get('all_original200_pixel_elements', [])) == 200) and (cap.get('original_true3_proof_SHA256') == '9676feeab2999d51ddac07d4900ec1947342795fba8db322b694de589efd89c3')

def finish_local_and_full_target_audit(Q):
    kernel_lock_owned(os.getpid(), D / 'runner.lock')
    core_unchanged()
    alias = read(H / 'own_original_real3_full_source_and_current_reference_review.json')
    assert alias['passed'] and alias['model_id'] == 'zai-org/AutoGLM-Phone-9B-Multilingual' and (alias['original_true3_proof_SHA256'] == '9676feeab2999d51ddac07d4900ec1947342795fba8db322b694de589efd89c3') and alias['all_original_true3_external_SHA_exact_or_exact_authorized_retention_transition'] and (alias['original_current_reference_SHA256'] == '4e740da3981c17fd7b599875ab95e60c5f709804f08c22c9fb64c3c8a5454ed8')
    local = base64.b64decode(Q['local_receipt_base64'], validate=True)
    assert hashlib.sha256(local).hexdigest() == Q['local_receipt_SHA256']
    lp = json.loads(local)
    pub = read(H / 'publication_receipt.json')
    assert pub['completed'] and pub['old181_state_registry_and_G_preserved'] and (lp['changed_G_cells'] == ['G693'])
    book = Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
    assert sha(book) == lp['after_sha256'] == Q['after_book_SHA256']
    p = D / 'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
    before = p.read_bytes()
    assert hashlib.sha256(before).hexdigest() == '0ae99984e7e1f37f31f6c17d586d2f41c38abe838760900dba49ac5c5abbb931'
    binding = json.loads(before)
    assert binding['local_workbook_sha256'] == lp['before_sha256'] and binding['real_record_count'] == binding['real_measured_record_count'] == 181
    assert binding['local_A100_reference_snapshot_sha256'] == '700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba' and binding['current_user_workbook_D_reference_snapshot_sha256'] == '14ab07ef221d0d9deb6da157bcb3e7cadd9577b6a7d0110eb333235c4728c49b'
    (H / 'before_local_binding_verbatim.json').write_bytes(before)
    (H / 'user_local_publication_receipt.json').write_bytes(local)
    binding.update(local_workbook_sha256=lp['after_sha256'], real_record_count=182, real_measured_record_count=182, numeric_within5pct=167, numeric_outside5pct=15, blank_count=527, original_reference_numeric_within5pct=167, original_reference_numeric_outside5pct=15, last_model_update='zai-org/AutoGLM-Phone-9B-Multilingual', last_local_publication_receipt_SHA256=Q['local_receipt_SHA256'], this_publication_preserved_all_non_G_content=True)
    assert p.read_bytes() == before
    atomic_json(p, binding)
    dual = dict(completed=True, UTC=datetime.now(timezone.utc).isoformat(), models=[x['model_id'] for x in pub['targets']], server_publication_receipt_sha256=sha(H / 'publication_receipt.json'), local_publication_receipt_sha256=Q['local_receipt_SHA256'], local_and_server_workbook_sha256=sha(book), local_binding_sha256=sha(p), old181_exactly_preserved=True, only_one_G_value_changed=True, current_D_counts=dict(within=167, outside=15), preserved_original_D_counts=dict(within=167, outside=15))
    atomic_json(H / 'dual_publication_completed.json', dual)
    transitions = []
    maps = []
    after = dict(pub['canonical_after_sha256'])
    after[str(p)] = sha(p)

    def checkmap(label, m):
        for path, h in m.items():
            assert re.fullmatch('[0-9a-f]{64}', h)
            if path in pub['canonical_before_sha256']:
                assert h == pub['canonical_before_sha256'][path] and sha(path) == after[path]
                transitions.append(dict(path=path, before=h, after=sha(path), reason='Authorized own zai GLM4.6V original BF16 publication; original inference and proof bytes remain unchanged'))
            else:
                assert sha(path) == h, (label, path)
        maps.append(dict(label=label, entries=len(m)))
    state = read(D / 'state.json')
    registry = read(D / 'registry.json')
    counts = Counter((v['status'] for v in state['models'].values()))
    assert dict(counts) == {'done': 182, 'skipped_no_complete_weights': 310, 'needs_adaptation': 208}
    rows, sheet, bh = workbook_rows(book)
    assert len(rows) == len({r['model_id'] for r in rows}) == 700 and state['current'] is None
    oldstate = read(H / 'before_verified_write/state.json')
    oldreg = read(H / 'before_verified_write/registry.json')
    targets = {x['model_id'] for x in pub['targets']}
    assert all((v == state['models'][m] for m, v in oldstate['models'].items() if m not in targets)) and all((v == registry[m] for m, v in oldreg.items())) and (set(registry) == set(oldreg) | targets)
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(book) as z:
        values = {c.get('r'): c.findtext('s:v', namespaces=ns) for c in ET.fromstring(z.read(sheet)).findall('.//s:c', ns)}
    within = sum((bool(Decimal(values[f"D{r['row']}"]) and abs(Decimal(values[f"D{r['row']}"]) - Decimal(r['score'])) / abs(Decimal(values[f"D{r['row']}"])) <= Decimal('.05')) for r in rows if r['score'] is not None))
    assert within == 167 and sum((r['score'] is not None for r in rows)) == 182
    for mid, e in state['models'].items():
        if e['status'] == 'skipped_no_complete_weights':
            assert next((r for r in rows if r['model_id'] == mid))['score'] is None
    targetaudits = []
    for t in pub['targets']:
        mid = t['model_id']
        spec = registry[mid]
        entry = state['models'][mid]
        proof = read(t['formal_proof'])
        assert sha(t['formal_proof']) == t['formal_proof_sha256'] == spec['accepted_formal200_proof_sha256'] and entry == read(D / 'receipts' / (mid.replace('/', '_') + '.json'))
        assert proof['passed'] and proof['sample_count'] == 200 and proof['actual_NPU_inference'] and proof['formal_alignment_within_5pct'] and proof['original_protocol_final_qualification_complete'] and (not proof['historical_record_reuse'])
        row, = [r for r in rows if r['model_id'] == mid]
        assert Decimal(row['score']) == Decimal(str(t['new_G'])) == Decimal(str(entry['score']))
        folder = Path(t['new_report'])
        assert {f.name for f in folder.iterdir()} == set(proof['report_sha256']) == {'eval_report.json', 'sample_inputs.jsonl', 'sample_outputs.jsonl', 'run.log', 'operator_lineage.json'}
        checkmap(mid + ' complete five', {str(folder / n): h for n, h in proof['report_sha256'].items()})
        checkmap(mid + ' external', proof['external_evidence_sha256'])
        for name, h in spec['source_sha256'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts and (sha(R / name) == h)
        approval = read(spec['alignment_approval_review'])
        assert sha(spec['alignment_approval_review']) == spec['alignment_approval_review_sha256'] and approval['approved']
        for name, key in [('true3_proof', 'true3_proof_SHA256'), ('capacity_review', 'capacity_review_SHA256')]:
            assert sha(approval[name]) == approval[key] and (glm46_capacity_review_passed(read(approval[name])) if name == 'capacity_review' else read(approval[name])['passed'])
        tests = read(H / mid.replace('/', '--') / 'isolated_11_writer_tests.json')
        assert tests['passed'] and tests['test_count'] == 11 and tests['production_workbook_unchanged']
        for k, m in proof.items():
            if k != 'external_evidence_sha256' and k.endswith('evidence_sha256') and isinstance(m, dict):
                checkmap(mid + ' ' + k, m)
        targetaudits.append(dict(model_id=mid, cell=f"G{row['row']}", G=row['score'], D=values[f"D{row['row']}"], proof_SHA256=t['formal_proof_sha256'], three_D_errors=t['exact_three_D_errors'], forward_count=t['actual_forward_count']))
    with zipfile.ZipFile(next((x for x in (H / 'before_verified_write').iterdir() if x.suffix == '.xlsx'))) as a, zipfile.ZipFile(book) as b:
        assert a.namelist() == b.namelist() and a.comment == b.comment
        for info in a.infolist():
            j = b.getinfo(info.filename)
            for attr in ('date_time', 'compress_type', 'external_attr', 'create_system', 'extra', 'comment'):
                assert getattr(info, attr) == getattr(j, attr)
            if info.filename != sheet:
                assert a.read(info.filename) == b.read(info.filename)
        ac = {x.get('r'): ET.tostring(x) for x in ET.fromstring(a.read(sheet)).findall('.//s:c', ns)}
        bc = {x.get('r'): ET.tostring(x) for x in ET.fromstring(b.read(sheet)).findall('.//s:c', ns)}
        assert {k for k in set(ac) | set(bc) if ac.get(k) != bc.get(k)} == {'G693'}
    baseline = read(B / 'ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119/verified181_export_complete_metadata.json')
    assert baseline['measured_count'] == len(baseline['records']) == 181
    for rec in baseline['records']:
        row, = [r for r in rows if r['model_id'] == rec['model_id']]
        assert row['task'] == rec['task'] and Decimal(row['score']) == Decimal(rec['current_G']) and (Decimal(values[f"D{row['row']}"]) == Decimal(rec['current_D']))
    for t in pub['targets']:
        row, = [r for r in rows if r['model_id'] == t['model_id']]
        e = state['models'][t['model_id']]
        s = registry[t['model_id']]
        baseline['records'].append(dict(model_id=t['model_id'], task=row['task'], current_row=row['row'], current_D=values[f"D{row['row']}"], current_G=row['score'], actual_canonical_metric=e['score'], accepted_report=e['report_dir'], active_report_files=e['active_report_files'], accepted_formal_proof=s['accepted_formal200_proof'], accepted_formal_proof_SHA256=s['accepted_formal200_proof_sha256']))
    baseline.update(measured_count=182, numeric_within5pct=167, numeric_outside5pct=15, complete_weights_pending_formal=208, confirmed_missing=310, original_protocol_qualified=68, workbook_SHA256=bh, UTC=datetime.now(timezone.utc).isoformat(), source='Incremental verified182 after own zai GLM4.6V HF publication; old181 proof reuse')
    for k in list(baseline):
        if k.startswith('next_new') or k.startswith('prior_') or k in ['target_full_audit', 'report_stat_baseline']:
            baseline.pop(k)
    baseline.update(current_original_A100_protocol_qualified=68, blank_count=518, server_book_SHA256=bh, source_canonical_SHA256={str(x): sha(x) for x in [book, D / 'state.json', D / 'registry.json', p]}, prior_181_record_evidence_not_rerun=True, next_new182_count_branch_review_mandatory=True)
    oldstats_path = B / 'ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119/active181_full_report_stat_baseline.json'
    assert sha(oldstats_path) == '46915b35ef9b942783ed09f8664eaf7bd6351d9d90c1eef62dd793bdb0642243'
    oldstats = read(oldstats_path)
    stats = {}
    for rec in baseline['records']:
        mid = rec['model_id']
        stats[mid] = {}
        files = rec.get('active_report_files')
        if files is None:
            assert mid not in targets and mid in oldstats['reports'] and isinstance(rec['accepted_report'], str)
            files = {name: {'path': str(Path(rec['accepted_report']) / name)} for name in oldstats['reports'][mid]}
        assert set(files) == {'eval_report.json', 'sample_inputs.jsonl', 'sample_outputs.jsonl', 'run.log', 'operator_lineage.json'}
        for name, info in files.items():
            q = Path(info['path'])
            st = q.stat()
            value = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
            stats[mid][name] = value
            if mid not in targets:
                assert value == oldstats['reports'][mid][name], (mid, name, 'Old accepted report stat changed: targeted deep review required before acceptance')
    assert len(stats) == 182 and len(oldstats['reports']) == 181
    stat_path = H / 'active182_full_report_stat_baseline.json'
    atomic_json(stat_path, dict(reports=stats, prior181_report_stats_exactly_unchanged=True, prior181_stat_file_SHA256=sha(oldstats_path), stat_only_cache_invalidation_not_new_acceptance=True))
    baseline['report_stat_baseline'] = str(stat_path)
    export = H / 'verified182_export_complete_metadata.json'
    atomic_json(export, baseline)
    statmeta = read(stat_path)
    statmeta['export_SHA256'] = sha(export)
    atomic_json(stat_path, statmeta)
    audit = dict(passed=True, UTC=datetime.now(timezone.utc).isoformat(), local_and_server_book_SHA256=bh, canonical_counts=dict(counts), numeric_within5pct=167, numeric_outside5pct=15, complete_weights_pending_formal=208, confirmed_missing=310, qualified_delivered=68, old181_state_registry_G_C_B_D_preserved=True, all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved=True, confirmed310_missing_G_all_blank=True, target_audits=targetaudits, all_target_external_maps_checked=maps, distinct_SHA256_files_checked=len(cache), authorized_canonical_pre_to_post_transitions=transitions, only_new_targets_deep_audited_no_old_model_CPU200_rerun=True, verified181_export=str(export), verified181_export_SHA256=sha(export), canonical_SHA256={str(x): sha(x) for x in [book, D / 'state.json', D / 'registry.json', p]})
    atomic_json(H / 'full_target_dual_publication_audit.json', audit)
    return dict(dual=dual, audit=audit, audit_path=str(H / 'full_target_dual_publication_audit.json'), audit_SHA256=sha(H / 'full_target_dual_publication_audit.json'), export=baseline)

def full_numeric_head_verified(proof):
    checks = proof.get('numeric_sample_checks', [])
    return len(checks) == proof['sample_count'] and {x['index'] for x in checks} == set(range(proof['sample_count'])) and (sum((x['full_head_rows'] for x in checks)) == proof['actual_forward_count']) and all((x['full_head_rows'] > 0 and len(x['head_reference_checks']) == x['full_head_rows'] and (x['all40_native_cache_values_reconstructed'] is True) and (x['original_input_pixels_token_ids_checked'] is True) and (x['all_masks_3D_positions_and_own_visual_injection_checked'] is True) for x in checks))
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]
original=R/'scripts/publish_AutoGLMPhone9B_verified182_original_single_lock_dual_transaction_v1_20261120.py';assert sha(original)=='fab4bab195788d36959c6d5ac4ad39f8e170d2456772cd1727690b52928ea6e8'
failed=read(I/'publication_actual_completion.json');assert failed['actual_returncode']==1 and failed['source_SHA256']==sha(original)
stderr=(I/'publication_stderr.log').read_text();assert "KeyError: 'old181_state_registry_and_G_preserved'" in stderr and 'finish_local_and_full_target_audit' in stderr
assert sha(H/'publication_receipt.json')=='bf766c39dee1928d7ed96e783d15039d8d7169b1111a0efd705351b7d76fa696'
assert sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')=='0b2fcf21fe5a56e5b33111c4e65c2ec0ddf53dae43d451dec756a342df810340'
assert sha(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')=='0ae99984e7e1f37f31f6c17d586d2f41c38abe838760900dba49ac5c5abbb931'
assert not (H/'dual_publication_completed.json').exists() and not (H/'full_target_dual_publication_audit.json').exists()
q=read(H/'reviewed_local_receipt_ready.json');assert q['local_receipt_SHA256']=='d6addb964b36b3649f38307d18fe510bddde149f7489969b281d61a327620bbf'
receipt_before=sha(H/'publication_receipt.json');before={str(p):sha(p) for p in [Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),D/'state.json',D/'registry.json']}
with (D/'runner.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
 result=finish_local_and_full_target_audit(q)
 for p,h in before.items():assert sha(p)==h
 assert sha(H/'publication_receipt.json')==receipt_before and read(I/'publication_actual_completion.json')['actual_returncode']==1
 value=dict(UTC=now(),actual_returncode=0,source=str(Path(__file__).resolve()),source_SHA256=sha(Path(__file__)),prior_real_publisher_returncode_preserved=1,only_legacy_receipt_schema_read_alias_and_exact_binding_final_audit_recovered=True,no_new_model_forward_or_G_state_registry_write=True,original_server_receipt_bytes_unchanged=True,local_and_server_book_SHA256=before['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'],full_target_audit=result['audit_path'],full_target_audit_SHA256=result['audit_SHA256'],counts=result['audit']['canonical_counts'],qualified_delivered=result['audit']['qualified_delivered'])
 atomic_json(H/'postcommit_exact_legacy_label_binding_audit_recovery_actual.json',value);print(json.dumps(value,ensure_ascii=False))

import sys
sys.dont_write_bytecode = True
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from collections import Counter
import os, copy, json, hashlib, fcntl, shutil, base64, subprocess, time, zipfile, re, xml.etree.ElementTree as ET
R = Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
B = D / 'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B / 'MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117'
Q = json.loads(Path(sys.argv[1]).read_bytes())
assert hashlib.sha256((R / Q['test_template']).read_bytes()).hexdigest() == Q['test_template_SHA256']
TEST_SOURCE_BASE64 = base64.b64encode((R / Q['test_template']).read_bytes()).decode()
sys.path[:0] = [str(R), str(R / 'scripts')]
from batch_910b_support import atomic_json, validate_report, workbook_rows
from A100_alignment_evidence_runner_20261103 import unchanged
from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
from eager_corrective_control_safety_2203 import core_unchanged, kernel_lock_owned
from local_A100_reference_gate_20261103 import check_local_A100_reference
from verified_Merve_actual_formal_acceptance_G_cell_writer_schema_v2_20261117 import write_verified_A100_G
import batch_910b as batch

def read(p):
    return json.loads(Path(p).read_bytes())
cache = {}

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
PROFILES = {'merve/paligemma_vqav2': {'total': 603, 'source_guard': 'all603_own_sources_actual_native_NPU_packed_mapping_verified', 'unused_guard': 'all603_own_sources_actual_native_NPU_packed_mapping_verified'}}
roles = {'model', 'alignment_runner', 'CPU_verifier_or_audit', 'CPU_verifier', 'verifier', 'independent_verifier', 'legacy_controller'}

def merve_capacity_review_passed(cap):
    assert cap['passed'] and cap['no_completed_NPU_stage_replayed'] and cap['no_model_YAML_or_protocol_or_weight_or_math_or_dtype_or_sampling_or_budget_changed'] and cap['all_original_reserves_and_per_batch_guards_unchanged']
    raw = read(cap['capacity_review'])
    assert sha(cap['capacity_review']) == cap['capacity_review_SHA256'] and raw['factor'] == 4 and (raw['extra_budget_bytes'] == 6 * 1024 ** 3) and raw['all_semantic_occurrences_and_values_retained'] and raw['metadata_overhead_included'] and (raw['empirical_estimate_bytes'] == cap['required_capacity_bytes'] < cap['actual_free_bytes'])
    return True

def merve_approval(raw):
    a = copy.deepcopy(raw)
    assert a['approved'] and a['real_NPU3_independent_proof_verified'] and a['numerical_review_verified'] and a['complete_original_values_occurrences_and_all_numeric_checks_retained']
    a['true3_proof'] = a['actual_completed_original_true3_proof']
    a['true3_proof_SHA256'] = a['actual_completed_original_true3_proof_SHA256']
    a['capacity_review'] = str(B / 'Merve_lossless_v2_completed_approved_fresh3_resumed_unstarted200_after_retention_v3_real_FLOCK_20261117/same_approved_fresh3_unstarted200_source_and_capacity_review.json')
    a['capacity_review_SHA256'] = sha(a['capacity_review'])
    return a

def finish_local_and_full_target_audit(Q):
    kernel_lock_owned(os.getpid(), D / 'runner.lock')
    core_unchanged()
    alias = read(H / 'own_original_real3_full_source_and_current_reference_review.json')
    assert alias['passed'] and alias['model_id'] == 'merve/paligemma_vqav2' and (alias['original_true3_proof_SHA256'] == 'd8b7425d08dfbc19077f41c83b79dd296671802f9f56f84bc8555e2f8481cdd3') and alias['all_original_true3_external_SHA_match_fresh_files'] and (alias['original_current_reference_SHA256'] == 'd2f69f9805a26da93b050e3bac71ed7d2bdb1f0823b57082cd068792c20fd54a')
    local = base64.b64decode(Q['local_receipt_base64'], validate=True)
    assert hashlib.sha256(local).hexdigest() == Q['local_receipt_SHA256']
    lp = json.loads(local)
    pub = read(H / 'publication_receipt.json')
    assert pub['completed'] and pub['old178_state_registry_and_G_preserved'] and (lp['changed_G_cells'] == ['G411'])
    book = Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
    assert sha(book) == lp['after_sha256'] == Q['after_book_SHA256']
    p = D / 'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
    assert sha(p)=='3534a1dc1762fcb276d9c074d52bc47b04cf82459bb58424c2893804dfd301d1'
    binding=read(p)
    assert binding['local_workbook_sha256']==lp['after_sha256'] and binding['real_record_count']==binding['real_measured_record_count']==179
    assert sha(H/'before_local_binding_verbatim.json')=='ed4eead6ccd1802f5731266f8f9ba8b180cf0a17ad39b3e1076abd2fc267e38d'
    assert (H/'user_local_publication_receipt.json').read_bytes()==local and binding['last_local_publication_receipt_SHA256']==Q['local_receipt_SHA256']
    dual=read(H/'dual_publication_completed.json')
    assert dual['completed'] and dual['local_and_server_workbook_sha256']==sha(book) and dual['local_publication_receipt_sha256']==Q['local_receipt_SHA256']
    transitions = []
    maps = []
    after = dict(pub['canonical_after_sha256'])
    after[str(p)] = sha(p)

    def checkmap(label, m):
        for path, h in m.items():
            assert re.fullmatch('[0-9a-f]{64}', h)
            if path in pub['canonical_before_sha256']:
                assert h == pub['canonical_before_sha256'][path] and sha(path) == after[path]
                transitions.append(dict(path=path, before=h, after=sha(path), reason='Authorized own Merve native PaliGemma original BF16 publication; original inference and proof bytes remain unchanged'))
            else:
                assert sha(path) == h, (label, path)
        maps.append(dict(label=label, entries=len(m)))
    state = read(D / 'state.json')
    registry = read(D / 'registry.json')
    counts = Counter((v['status'] for v in state['models'].values()))
    assert dict(counts) == {'done': 179, 'skipped_no_complete_weights': 310, 'needs_adaptation': 211}
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
    assert within == 164 and sum((r['score'] is not None for r in rows)) == 179
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
        approval = merve_approval(read(spec['alignment_approval_review']))
        assert sha(spec['alignment_approval_review']) == spec['alignment_approval_review_sha256'] and approval['approved']
        for name, key in [('true3_proof', 'true3_proof_SHA256'), ('capacity_review', 'capacity_review_SHA256')]:
            assert sha(approval[name]) == approval[key] and (merve_capacity_review_passed(read(approval[name])) if name == 'capacity_review' else read(approval[name])['passed'])
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
        assert {k for k in set(ac) | set(bc) if ac.get(k) != bc.get(k)} == {'G411'}
    baseline = read(B / 'Granite41_original_HF_BF16_verified178_blank_G_publication_20261117/verified178_export_complete_metadata.json')
    assert baseline['measured_count'] == len(baseline['records']) == 178
    for rec in baseline['records']:
        row, = [r for r in rows if r['model_id'] == rec['model_id']]
        assert row['task'] == rec['task'] and Decimal(row['score']) == Decimal(rec['current_G']) and (Decimal(values[f"D{row['row']}"]) == Decimal(rec['current_D']))
    for t in pub['targets']:
        row, = [r for r in rows if r['model_id'] == t['model_id']]
        e = state['models'][t['model_id']]
        s = registry[t['model_id']]
        baseline['records'].append(dict(model_id=t['model_id'], task=row['task'], current_row=row['row'], current_D=values[f"D{row['row']}"], current_G=row['score'], actual_canonical_metric=e['score'], accepted_report=e['report_dir'], active_report_files=e['active_report_files'], accepted_formal_proof=s['accepted_formal200_proof'], accepted_formal_proof_SHA256=s['accepted_formal200_proof_sha256']))
    baseline.update(measured_count=179, numeric_within5pct=164, numeric_outside5pct=15, complete_weights_pending_formal=211, confirmed_missing=310, original_protocol_qualified=65, workbook_SHA256=bh, UTC=datetime.now(timezone.utc).isoformat(), source='Incremental verified179 after own Merve native PaliGemma HF publication; old178 proof reuse')
    for k in list(baseline):
        if k.startswith('next_new') or k.startswith('prior_') or k in ['target_full_audit', 'report_stat_baseline']:
            baseline.pop(k)
    baseline.update(current_original_A100_protocol_qualified=65, blank_count=521, server_book_SHA256=bh, source_canonical_SHA256={str(x): sha(x) for x in [book, D / 'state.json', D / 'registry.json', p]}, prior_178_record_evidence_not_rerun=True, next_new180_count_branch_review_mandatory=True)
    oldstats_path = B / 'Granite41_original_HF_BF16_verified178_blank_G_publication_20261117/active178_full_report_stat_baseline.json'
    assert sha(oldstats_path) == 'cdd6073ed2ec93766f9c65616b5912fbb3de18d594e1e887926d940c37a631cf'
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
    assert len(stats) == 179 and len(oldstats['reports']) == 178
    stat_path = H / 'active179_full_report_stat_baseline.json'
    atomic_json(stat_path, dict(reports=stats, prior178_report_stats_exactly_unchanged=True, prior178_stat_file_SHA256=sha(oldstats_path), stat_only_cache_invalidation_not_new_acceptance=True))
    baseline['report_stat_baseline'] = str(stat_path)
    export = H / 'verified179_export_complete_metadata.json'
    atomic_json(export, baseline)
    statmeta = read(stat_path)
    statmeta['export_SHA256'] = sha(export)
    atomic_json(stat_path, statmeta)
    audit = dict(passed=True, UTC=datetime.now(timezone.utc).isoformat(), local_and_server_book_SHA256=bh, canonical_counts=dict(counts), numeric_within5pct=164, numeric_outside5pct=15, complete_weights_pending_formal=211, confirmed_missing=310, qualified_delivered=65, old178_state_registry_G_C_B_D_preserved=True, all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved=True, confirmed310_missing_G_all_blank=True, target_audits=targetaudits, all_target_external_maps_checked=maps, distinct_SHA256_files_checked=len(cache), authorized_canonical_pre_to_post_transitions=transitions, only_new_targets_deep_audited_no_old_model_CPU200_rerun=True, verified179_export=str(export), verified179_export_SHA256=sha(export), canonical_SHA256={str(x): sha(x) for x in [book, D / 'state.json', D / 'registry.json', p]})
    atomic_json(H / 'full_target_dual_publication_audit.json', audit)
    return dict(dual=dual, audit=audit, audit_path=str(H / 'full_target_dual_publication_audit.json'), audit_SHA256=sha(H / 'full_target_dual_publication_audit.json'), export=baseline)

REC=B/'Merve_verified179_committed_dual_result_readonly_final_audit_field_schema_recovery_v4_20261117'
assert not REC.exists();REC.mkdir()
started=now();canonical=[Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]+[D/n for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json']]
before={str(p):sha(p) for p in canonical}
try:
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in roles]
 I=B/'MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117'
 fail=read(I/'publication_actual_completion.json');assert fail['actual_returncode']==1 and fail['source_SHA256']=='96b8f91ecbd645f8a4e920e4209c9f840a90c6829c28436796fe2a6265ef4317'
 assert "KeyError: 'actual_native_model_forward_calls'" in (I/'publication_stderr.log').read_text()
 request=read(H/'reviewed_local_receipt_ready.json')
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
  assert not [p for p in project_processes() if p['role'] in roles]
  result=finish_local_and_full_target_audit(request)
  assert {str(p):sha(p) for p in canonical}==before
  receipt=dict(passed=True,started_UTC=started,completed_UTC=now(),actual_returncode=0,original_failed_publication_completion_preserved=True,canonical_G_state_registry_binding_not_rewritten=True,all_original_full_numeric_and_dual_publication_checks_retained=True,only_final_audit_forward_count_key_corrected_to_actual_existing_receipt_schema=True,original_model_and_five_reports_and_proofs_and_sources_unchanged=True,canonical_before_and_after_SHA256=before,full_target_audit_SHA256=result['audit_SHA256'],verified179_export=result['audit']['verified179_export'],verified179_export_SHA256=result['audit']['verified179_export_SHA256'])
  atomic_json(REC/'actual_completion.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False))
except BaseException as exc:
 atomic_json(REC/'actual_completion.json',dict(passed=False,started_UTC=started,completed_UTC=now(),actual_returncode=1,error=repr(exc),canonical_before_SHA256=before));raise

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
H = B / 'HuihuiGLM46Flash_original_BF16_2048_lossless_verified181_blank_G_publication_20261119'
Q = json.loads(Path(sys.argv[1]).read_bytes())
assert hashlib.sha256((R / Q['test_template']).read_bytes()).hexdigest() == Q['test_template_SHA256']
TEST_SOURCE_BASE64 = base64.b64encode((R / Q['test_template']).read_bytes()).decode()
sys.path[:0] = [str(R), str(R / 'scripts')]
from batch_910b_support import atomic_json, validate_report, workbook_rows
from A100_alignment_evidence_runner_20261103 import unchanged
from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
from eager_corrective_control_safety_2203 import core_unchanged, kernel_lock_owned
from local_A100_reference_gate_20261103 import check_local_A100_reference
from verified_A100_G_cell_writer_20261103 import write_verified_A100_G
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
PROFILES = {'huihui-ai/Huihui-GLM-4.6V-Flash-abliterated': {'total': 704, 'source_guard': 'all_own_source_actual_NPU_SHA256_verified', 'unused_guard': 'all_original_active_and_unused_source_bytes_verified'}}
roles = {'model', 'alignment_runner', 'CPU_verifier_or_audit', 'CPU_verifier', 'verifier', 'independent_verifier', 'legacy_controller'}

def huihui_capacity_review_passed(cap):
    return cap.get('passed') is True and cap.get('empirical_estimate_bytes') == 146869927228 and (cap.get('all333_names_shapes_dtypes_every_raw_bit_equal') is True) and (cap.get('only_standard_lossless_NPY_order_ZIP_codec_change') is True) and (cap.get('original_factor4_plus6GiB_and_all_sample_guards_unchanged') is True) and (cap.get('original_factor4_plus6GiB_and_hard3_log1_precheckHalf_sample2GiB_preserved') is True) and (len(cap['all200_original_prompt_lengths']) == 200) and (len(cap['all200_original_pixel_elements']) == 200)

def finish_local_and_full_target_audit(Q):
    kernel_lock_owned(os.getpid(), D / 'runner.lock')
    core_unchanged()
    alias = read(H / 'own_original_real3_full_source_and_current_reference_review.json')
    assert alias['passed'] and alias['model_id'] == 'huihui-ai/Huihui-GLM-4.6V-Flash-abliterated' and (alias['original_true3_proof_SHA256'] == '7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59') and alias['all_original_true3_external_SHA_match_fresh_files'] and (alias['original_current_reference_SHA256'] == '689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34')
    local = base64.b64decode(Q['local_receipt_base64'], validate=True)
    assert hashlib.sha256(local).hexdigest() == Q['local_receipt_SHA256']
    lp = json.loads(local)
    pub = read(H / 'publication_receipt.json')
    assert pub['completed'] and pub['old180_state_registry_and_G_preserved'] and (lp['changed_G_cells'] == ['G355'])
    book = Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
    assert sha(book) == lp['after_sha256'] == Q['after_book_SHA256']
    p = D / 'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
    before = p.read_bytes()
    assert hashlib.sha256(before).hexdigest() == 'e3a02a66da363e3ca15649f1cdab2d553f693297dc1ccbad25ea8495d12ca88b'
    binding = json.loads(before)
    assert binding['local_workbook_sha256'] == lp['before_sha256'] and binding['real_record_count'] == binding['real_measured_record_count'] == 180
    assert binding['local_A100_reference_snapshot_sha256'] == '700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba' and binding['current_user_workbook_D_reference_snapshot_sha256'] == '14ab07ef221d0d9deb6da157bcb3e7cadd9577b6a7d0110eb333235c4728c49b'
    (H / 'before_local_binding_verbatim.json').write_bytes(before)
    (H / 'user_local_publication_receipt.json').write_bytes(local)
    binding.update(local_workbook_sha256=lp['after_sha256'], real_record_count=181, real_measured_record_count=181, numeric_within5pct=166, numeric_outside5pct=15, blank_count=528, original_reference_numeric_within5pct=166, original_reference_numeric_outside5pct=15, last_model_update='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated', last_local_publication_receipt_SHA256=Q['local_receipt_SHA256'], this_publication_preserved_all_non_G_content=True)
    assert p.read_bytes() == before
    atomic_json(p, binding)
    dual = dict(completed=True, UTC=datetime.now(timezone.utc).isoformat(), models=[x['model_id'] for x in pub['targets']], server_publication_receipt_sha256=sha(H / 'publication_receipt.json'), local_publication_receipt_sha256=Q['local_receipt_SHA256'], local_and_server_workbook_sha256=sha(book), local_binding_sha256=sha(p), old180_exactly_preserved=True, only_one_G_value_changed=True, current_D_counts=dict(within=166, outside=15), preserved_original_D_counts=dict(within=166, outside=15))
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
                transitions.append(dict(path=path, before=h, after=sha(path), reason='Authorized own Huihui GLM4.6V original BF16 publication; original inference and proof bytes remain unchanged'))
            else:
                assert sha(path) == h, (label, path)
        maps.append(dict(label=label, entries=len(m)))
    state = read(D / 'state.json')
    registry = read(D / 'registry.json')
    counts = Counter((v['status'] for v in state['models'].values()))
    assert dict(counts) == {'done': 181, 'skipped_no_complete_weights': 310, 'needs_adaptation': 209}
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
    assert within == 166 and sum((r['score'] is not None for r in rows)) == 181
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
            assert sha(approval[name]) == approval[key] and (huihui_capacity_review_passed(read(approval[name])) if name == 'capacity_review' else read(approval[name])['passed'])
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
        assert {k for k in set(ac) | set(bc) if ac.get(k) != bc.get(k)} == {'G355'}
    baseline = read(B / 'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117/verified180_export_complete_metadata.json')
    assert baseline['measured_count'] == len(baseline['records']) == 180
    for rec in baseline['records']:
        row, = [r for r in rows if r['model_id'] == rec['model_id']]
        assert row['task'] == rec['task'] and Decimal(row['score']) == Decimal(rec['current_G']) and (Decimal(values[f"D{row['row']}"]) == Decimal(rec['current_D']))
    for t in pub['targets']:
        row, = [r for r in rows if r['model_id'] == t['model_id']]
        e = state['models'][t['model_id']]
        s = registry[t['model_id']]
        baseline['records'].append(dict(model_id=t['model_id'], task=row['task'], current_row=row['row'], current_D=values[f"D{row['row']}"], current_G=row['score'], actual_canonical_metric=e['score'], accepted_report=e['report_dir'], active_report_files=e['active_report_files'], accepted_formal_proof=s['accepted_formal200_proof'], accepted_formal_proof_SHA256=s['accepted_formal200_proof_sha256']))
    baseline.update(measured_count=181, numeric_within5pct=166, numeric_outside5pct=15, complete_weights_pending_formal=209, confirmed_missing=310, original_protocol_qualified=67, workbook_SHA256=bh, UTC=datetime.now(timezone.utc).isoformat(), source='Incremental verified181 after own Huihui GLM4.6V HF publication; old180 proof reuse')
    for k in list(baseline):
        if k.startswith('next_new') or k.startswith('prior_') or k in ['target_full_audit', 'report_stat_baseline']:
            baseline.pop(k)
    baseline.update(current_original_A100_protocol_qualified=67, blank_count=519, server_book_SHA256=bh, source_canonical_SHA256={str(x): sha(x) for x in [book, D / 'state.json', D / 'registry.json', p]}, prior_180_record_evidence_not_rerun=True, next_new181_count_branch_review_mandatory=True)
    oldstats_path = B / 'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117/active180_full_report_stat_baseline.json'
    assert sha(oldstats_path) == '2ad3869d9152b05f7235a649525fdd54328c6ac157ae938cae23f8f62254dc37'
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
    assert len(stats) == 181 and len(oldstats['reports']) == 180
    stat_path = H / 'active181_full_report_stat_baseline.json'
    atomic_json(stat_path, dict(reports=stats, prior180_report_stats_exactly_unchanged=True, prior180_stat_file_SHA256=sha(oldstats_path), stat_only_cache_invalidation_not_new_acceptance=True))
    baseline['report_stat_baseline'] = str(stat_path)
    export = H / 'verified181_export_complete_metadata.json'
    atomic_json(export, baseline)
    statmeta = read(stat_path)
    statmeta['export_SHA256'] = sha(export)
    atomic_json(stat_path, statmeta)
    audit = dict(passed=True, UTC=datetime.now(timezone.utc).isoformat(), local_and_server_book_SHA256=bh, canonical_counts=dict(counts), numeric_within5pct=166, numeric_outside5pct=15, complete_weights_pending_formal=209, confirmed_missing=310, qualified_delivered=67, old180_state_registry_G_C_B_D_preserved=True, all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved=True, confirmed310_missing_G_all_blank=True, target_audits=targetaudits, all_target_external_maps_checked=maps, distinct_SHA256_files_checked=len(cache), authorized_canonical_pre_to_post_transitions=transitions, only_new_targets_deep_audited_no_old_model_CPU200_rerun=True, verified181_export=str(export), verified181_export_SHA256=sha(export), canonical_SHA256={str(x): sha(x) for x in [book, D / 'state.json', D / 'registry.json', p]})
    atomic_json(H / 'full_target_dual_publication_audit.json', audit)
    return dict(dual=dual, audit=audit, audit_path=str(H / 'full_target_dual_publication_audit.json'), audit_SHA256=sha(H / 'full_target_dual_publication_audit.json'), export=baseline)

def full_numeric_head_verified(proof):
 checks=proof.get('numeric_sample_checks',[])
 return len(checks)==proof['sample_count'] and {x['index'] for x in checks}==set(range(proof['sample_count'])) and sum(x['full_head_rows'] for x in checks)==proof['actual_forward_count'] and all(x['full_head_rows']>0 and len(x['head_reference_checks'])==x['full_head_rows'] and x['all40_native_cache_values_reconstructed'] is True and x['original_input_pixels_token_ids_checked'] is True and x['all_masks_3D_positions_and_own_visual_injection_checked'] is True for x in checks)

def check_original_external_hash(path, expected):
 actual=sha(path)
 if actual==expected:return
 ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
 assert str(path)==str(ledger) and expected=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5' and actual=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32', (path,'External evidence changed outside exact user-authorized cleanup transition')
 receipt=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json'
 assert sha(receipt)=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
 value=read(receipt);assert value['actual_returncode']==0 and value['all_protected_SHA256_unchanged'] is True and value['removed_files']==4972 and value['manifest_SHA256']=='f955add85f6012e7b1f8f4ad5fc0b78af043b80d2d0797315c21d59b35478382'
 assert read(ledger)['events'][-1]['receipt_SHA256']==sha(receipt)

assert not H.exists() and (not runner_lock_owners()) and (not [p for p in project_processes() if p['role'] in roles])
policy = read(D / 'A100_ALIGNMENT_PUBLICATION_POLICY.json')
decision = Path(policy['authorization'])
assert sha(decision) == policy['authorization_sha256'] == 'b47ae739b279138134f87978bf4a71b3bbcecff149dcdbfc0b26e49ab04234f1'
assert sha(R / 'scripts/verified_A100_G_cell_writer_20261103.py') == '83134eec524c27bc4e9fd4569193e72038a7b248b379c12cf6999e1dcff798b8'
book = Path(read(D / 'POLICY.json')['workbook'])
sf = D / 'state.json'
rf = D / 'registry.json'
binding = D / 'A100_ALIGNMENT_LOCAL_WORKBOOK.json'
with (D / 'runner.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    kernel_lock_owned(os.getpid(), D / 'runner.lock')
    core_unchanged()
    assert not [p for p in project_processes() if p['role'] in roles]
    assert sha(book) == Q['expected_before_sha256'] == '689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
    state = read(sf)
    registry = read(rf)
    old_state = copy.deepcopy(state)
    old_reg = copy.deepcopy(registry)
    assert state['current'] is None and read(D / 'PAUSE')['owner'] == 'A100_relative5pct_alignment_20261103_explicit_user_resume'
    rows, _, before = workbook_rows(book)
    assert len(rows) == 700 and sum((r['score'] is not None for r in rows)) == 180
    candidates = []
    all_external = {}
    reference_alias_review = None
    assert len(Q['changes']) == 1 and {x['model_id'] for x in Q['changes']} == set(PROFILES)
    assert {x['cell'] for x in Q['changes']} == {'G355'}
    for c in Q['changes']:
        mid = c['model_id']
        profile = PROFILES[mid]
        expected = Q['models'][mid]
        J = Path(expected['job'])
        job = read(J / 'job.json')
        spec = copy.deepcopy(job['spec'])
        assert sha(J / 'job.json') == expected['job_SHA256'] and job['stages'] == [3, 200] and spec['approved'] and (not job['write_workbook'])
        unchanged(job['protected_canonical_hashes'])
        batch.check_sources(spec)
        a = read(spec['alignment_approval_review'])
        assert sha(spec['alignment_approval_review']) == spec['alignment_approval_review_sha256']
        assert a['approved'] and a['real_NPU3_independent_proof_verified'] and a['numerical_review_verified'] and a['first200_measurement_only_not_preclaimed_qualification']
        assert sha(a['true3_proof']) == a['true3_proof_SHA256'] and sha(a['capacity_review']) == a['capacity_review_SHA256'] and huihui_capacity_review_passed(read(a['capacity_review']))
        t = read(a['true3_proof'])
        assert t['passed'] and t['sample_count'] == 3 and t['actual_NPU_inference'] and (not t['historical_record_reuse'])
        assert t['all_own_source_actual_NPU_SHA256_verified'] and t[profile['source_guard']] and t[profile['unused_guard']] and t['all_native_KV_values_losslessly_reconstructed'] and full_numeric_head_verified(t) and (validate_report(R, t['report_dir'], mid, c['task'], 3, t) == expected['true3_score']) and (sha(a['true3_proof']) == expected['true3_proof_SHA256'])
        done = read(J / 'completion.json')
        sup = read(Path(expected['controller']) / 'actual_completion.json')
        assert done['actual_returncode'] == done['model_returncode'] == done['verifier_returncode'] == sup['actual_returncode'] == 0
        assert [x['sample_count'] for x in done['accepted']] == [3, 200]
        run = done['accepted'][-1]
        assert sha(run['proof']) == run['proof_sha256'] == c['formal_proof_sha256']
        proof = read(run['proof'])
        assert proof['independent_verifier_SHA256']=='22d2919d3bca51c68a6f3de45141ed55999d781aefccb9b22c18affc79e61a1d' and proof['source_tensor_count']==704 and proof['original_unused_MTP_source_tensor_count']==0 and proof['own_original_BF16_preserved'] and full_numeric_head_verified(proof)
        assert proof['passed'] and proof['actual_NPU_inference'] and (not proof['historical_record_reuse']) and proof['formal_alignment_within_5pct']
        assert proof['original_protocol_final_qualification_complete'] and (not proof['A100_historical_engine_or_tensor_capture_claimed']) and (not proof['error_indices'])
        assert proof['all_own_source_actual_NPU_SHA256_verified'] and proof[profile['source_guard']] and proof[profile['unused_guard']] and full_numeric_head_verified(proof)
        assert proof['actual_forward_count'] == expected['expected_forward_count'] and len(proof['numeric_sample_checks']) == 200 and (proof['NPU_profiler_kernel_events'] == expected['expected_kernels'])
        assert proof['all_native_KV_values_losslessly_reconstructed'] and proof['all_full_inputs_tokens_raw_references_errors_preserved']
        assert proof['truncated_indices'] == expected['expected_truncated'] and proof['empty_indices'] == expected['expected_empty']
        assert proof['invalid_indices'] == expected['expected_invalid']
        score = validate_report(R, run['report_dir'], mid, c['task'], 200, proof)
        assert Decimal(str(score)) == Decimal(str(c['new_value']))
        assert all((Decimal(v) <= Decimal('.05') for v in proof['relative_error_to_current_A100_D'].values()))
        gate = check_local_A100_reference(mid, c['task'], score)
        err = abs(Decimal(c['A100_full_precision_cached_D']) - Decimal(str(score))) / abs(Decimal(c['A100_full_precision_cached_D']))
        assert err <= Decimal('.05')
        gate['current_user_workbook_D_additional_gate'] = {'D': c['A100_full_precision_cached_D'], 'relative_error': str(err), 'passed': True, 'original_full_precision_reference_gate_also_retained': True}
        for p in [t] + [read(x['proof']) for x in done['accepted']]:
            validate_report(R, p['report_dir'], mid, c['task'], p['sample_count'], p)
            for path, h in p['external_evidence_sha256'].items():
                check_original_external_hash(path,h)
                if path in all_external and all_external[path]!=h:
                    check_original_external_hash(path,all_external[path])
                all_external[path] = sha(path)
        row, = [r for r in rows if r['model_id'] == mid]
        assert row['row'] == c['row'] and row['task'] == c['task'] and (row['score'] is None)
        assert state['models'][mid]['status'] == 'needs_adaptation' and state['models'][mid]['score'] is None and (mid not in registry) and (not (D / 'receipts' / (mid.replace('/', '_') + '.json')).exists())
        candidates.append(dict(change=c, J=J, job=job, spec=spec, done=done, run=run, proof=proof, score=score, gate=gate))
    for p, h in all_external.items():
        assert sha(p) == h, (p, 'External evidence changed before batch publication')
    reference_alias_review = dict(passed=True, model_id=mid, original_true3_proof=a['true3_proof'], original_true3_proof_SHA256=a['true3_proof_SHA256'], original_current_reference_SHA256=Q['expected_before_sha256'], all_original_true3_external_SHA_match_fresh_files=True, no_reference_alias_used_or_needed=True, all_other_external_SHA_mandatory=True)
    assert reference_alias_review['passed']
    H.mkdir()
    atomic_json(H / 'own_original_real3_full_source_and_current_reference_review.json', reference_alias_review)
    backup = H / 'before_verified_write'
    backup.mkdir()
    saved = {}
    for path in [book, sf, rf, binding]:
        dest = backup / path.name
        shutil.copy2(path, dest)
        assert sha(dest) == sha(path)
        saved[str(path)] = {'backup': str(dest), 'sha256': sha(path)}
    atomic_json(H / 'backup_manifest.json', saved)
    for v in candidates:
        mid = v['change']['model_id']
        home = H / mid.replace('/', '--')
        home.mkdir()
        v['home'] = home
        manifest = home / 'verified_candidate.json'
        v['manifest'] = manifest
        atomic_json(manifest, dict(model_id=mid, task=v['change']['task'], new_score=v['score'], expected_old_score=None, expected_book_sha256=before, job_completion=str(v['J'] / 'completion.json'), report_dir=v['run']['report_dir'], proof=v['run']['proof'], proof_sha256=v['run']['proof_sha256'], new_real_NPU_inference=True, historical_record_reuse=False, authorization=str(decision), authorization_sha256=sha(decision), local_full_precision_A100_gate=v['gate'], artifact_tool_authored_target_numeric_value_verified=v['change']))
        src = base64.b64decode(TEST_SOURCE_BASE64, validate=True).decode().replace('InternVL2_2B_A100_existing_D_verified_blank_G_publication_v2_1805_20261115', str(home.relative_to(B)))
        tp = R / ('scripts/tests/test_' + mid.replace('/', '_').replace('-', '_') + '_verified181_original_writer_isolated_20261117.py')
        assert not tp.exists()
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(src, encoding='utf-8')
        (home / 'isolated_writer_test_source_snapshot.py.txt').write_bytes(tp.read_bytes())
        with (home / 'isolated_11_writer_tests.json').open('xb') as out, (home / 'isolated_11_writer_tests_stderr.log').open('xb') as err:
            result = subprocess.run(['python3', '-B', str(tp)], cwd=R, stdout=out, stderr=err, env=dict(os.environ, PYTHONPATH=str(R / 'scripts'), PYTHONDONTWRITEBYTECODE='1'))
        assert result.returncode == 0, (home / 'isolated_11_writer_tests_stderr.log').read_text()
        tests = read(home / 'isolated_11_writer_tests.json')
        assert tests['passed'] and tests['test_count'] == 11 and tests['production_workbook_unchanged']
    fixture = H / 'isolated_sequential_pair_fixture'
    fixture.mkdir()
    tb = fixture / 'book.xlsx'
    shutil.copy2(book, tb)
    auth = fixture / 'authorization.json'
    atomic_json(auth, dict(test_fixture_only=True, origin='isolated_test_fixture', test_root=str(fixture)))
    predicted = []
    for v in candidates:
        predicted.append(write_verified_A100_G(tb, v['change']['row'], v['change']['model_id'], v['score'], fixture / v['change']['model_id'].replace('/', '--'), expected_task=v['change']['task'], decision_path=auth, expected_book_hash=sha(tb), manifest_path=v['manifest']))
    assert sha(book) == before
    unchanged({p: v['sha256'] for p, v in saved.items()})
    atomic_json(H / 'transaction.json', dict(phase='single_full_evidence_and_11_tests_passed_same_prepublication_baseline', before=saved, expected_final_book=sha(tb)))
    for n, v in enumerate(candidates):
        c = v['change']
        mid = c['model_id']
        entry = state['models'][mid]
        spec = v['spec']
        done = v['done']
        proof = v['proof']
        run = v['run']
        folder = Path(run['report_dir'])
        write = write_verified_A100_G(book, c['row'], mid, v['score'], v['home'] / 'writer_backup', expected_task=c['task'], decision_path=decision, expected_book_hash=before if n == 0 else candidates[n - 1]['write']['after_sha256'], manifest_path=v['manifest'])
        assert write['after_sha256'] == predicted[n]['after_sha256']
        v['write'] = write
        atomic_json(H / 'transaction.json', dict(phase='authorized_G_batch_write_in_progress', completed_targets=[x['change']['model_id'] for x in candidates[:n + 1]], before=saved, latest_write=write))
        for item in done['accepted']:
            stage = 'fresh3' if item['sample_count'] == 3 else 'formal200'
            entry['attempts'].append(dict(count=item['sample_count'], diagnostics=str(v['J'] / stage), report_dir=item['report_dir'], independent_verification=item['proof'], proof_sha256=item['proof_sha256'], score=item['score'], new_real_NPU_inference=True))
        entry.update(status='done', row=write['row'], score=v['score'], score_in_workbook=repr(v['score']), published_score_in_workbook=v['score'], workbook_score_representation='exact', report_dir=str(folder), completed_at=done['completed_at'], active_config=str(R / spec['config']), active_attempt=str(v['J'] / 'formal200'), original_protocol_final_qualification_complete=True, qualification_basis='User-confirmed A100 workbook D; own original Huihui GLM4.6V BF16/card4/2048/existing thinkingFalse, complete original math and full independent NPU200 audit; only lossless numeric storage changed', A100_historical_engine_or_tensor_capture_claimed=False, reason=None, last_error=None, workbook_write=write, active_report_files={name: {'path': str(folder / name), 'sha256': h} for name, h in proof['report_sha256'].items()}, A100_relative_alignment=dict(metric='Accuracy', A100_D=float(c['A100_full_precision_cached_D']), relative_error=proof['relative_error_to_A100_table_D'], passed=True, sample_count=200, authorization=str(decision), authorization_sha256=sha(decision)))
        spec.update(accepted_formal200_report=str(folder), accepted_formal200_proof=run['proof'], accepted_formal200_proof_sha256=run['proof_sha256'], active_report_files=entry['active_report_files'])
        registry[mid] = spec
    assert sha(book) == sha(tb)
    state['updated_at'] = now()
    targets = {v['change']['model_id'] for v in candidates}
    assert all((state['models'][k] == val and registry.get(k) == old_reg.get(k) for k, val in old_state['models'].items() if k not in targets))
    counts = Counter((v['status'] for v in state['models'].values()))
    assert counts['done'] == 181 and counts['needs_adaptation'] == 209 and (counts['skipped_no_complete_weights'] == 310)
    unchanged({str(p): saved[str(p)]['sha256'] for p in [sf, rf, binding]})
    atomic_json(rf, registry)
    atomic_json(sf, state)
    for v in candidates:
        atomic_json(D / 'receipts' / (v['change']['model_id'].replace('/', '_') + '.json'), state['models'][v['change']['model_id']])
    receipt = dict(completed=True, UTC=now(), authorized_single_verified_target=True, targets=[dict(model_id=v['change']['model_id'], new_G=v['score'], A100_D=v['change']['A100_full_precision_cached_D'], exact_three_D_errors=v['proof']['relative_error_to_current_A100_D'], workbook_write=v['write'], new_report=v['run']['report_dir'], formal_proof=v['run']['proof'], formal_proof_sha256=v['run']['proof_sha256'], actual_forward_count=v['proof']['actual_forward_count'], source_tensor_count=PROFILES[v['change']['model_id']]['total'], all704_own_stored_source_and_actual_NPU_SHA256_verified=True, active_source_tensor_count=704, original_unused_MTP_source_tensor_count=0, loaded_source_count_is_not_a_claim_that_all_source_modules_executed=True, invalid_indices=v['proof']['invalid_indices'], empty_indices=v['proof']['empty_indices'], truncated_indices=v['proof']['truncated_indices'], actual_NPU_card=4) for v in candidates], canonical_counts=dict(counts), old180_state_registry_and_G_preserved=True, canonical_before_sha256={p: v['sha256'] for p, v in saved.items()}, canonical_after_sha256={str(p): sha(p) for p in [book, sf, rf]}, all_new_target_external_SHA_freshly_checked=len(all_external), distinct_files_hashed=len(cache), original_protocol_final_qualification_complete=True, local_CAS_commit_pending=True)
    atomic_json(H / 'publication_receipt.json', receipt)
    atomic_json(H / 'transaction.json', dict(phase='single_authorized_G_state_registry_receipts_committed', receipt=str(H / 'publication_receipt.json')))
    atomic_json(H / 'server_phase_waiting_for_local_receipt.json', dict(UTC=now(), PID=os.getpid(), same_original_FLOCK_held=True, server_receipt_SHA256=sha(H / 'publication_receipt.json'), local_receipt_not_yet_claimed=True))
    request = H / 'reviewed_local_receipt_ready.json'
    deadline = time.monotonic() + 1800
    while not request.exists():
        if time.monotonic() > deadline:
            atomic_json(H / 'NOTIFY_local_sync_pending.json', dict(UTC=now(), server_publication_complete=True, local_sync_not_complete=True, chat_notification_not_claimed=True))
            raise TimeoutError('Server publication completed but local CAS receipt did not arrive within1800s; keep server evidence and resume the separately bound finish audit, never rerun the model')
        time.sleep(2)
    final = finish_local_and_full_target_audit(read(request))
    atomic_json(H / 'transaction.json', dict(phase='dual_workbook_state_registry_binding_full_target_audit_committed', same_original_FLOCK_held_across_both_books=True, dual_receipt=str(H / 'dual_publication_completed.json'), full_audit=str(H / 'full_target_dual_publication_audit.json')))
print(json.dumps(dict(server_receipt=receipt, dual=final['dual'], audit=final['audit']), ensure_ascii=False))

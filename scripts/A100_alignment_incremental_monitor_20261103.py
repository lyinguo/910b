"""Incremental routine checks only; never accepts a model, writes G, or controls jobs."""
import argparse, collections, datetime, fcntl, hashlib, json, os, pathlib, re, subprocess, tempfile

R = pathlib.Path('/root/projects/comperror')
D = R / 'workspace/diagnostics/batch_910b'
M = D / 'incremental_monitor'
FIVE = ('eval_report.json', 'sample_inputs.jsonl', 'sample_outputs.jsonl', 'run.log', 'operator_lineage.json')
LIMIT = 1024 * 1024
ERROR = re.compile(r'Traceback \(most recent|\bERROR\b|\bFATAL\b|\bException\b|\bAssertionError\b|\bRuntimeError\b|out of memory|\bOOM\b|\bACL_ERROR\b', re.I)


def is_error_candidate(line):
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return bool(ERROR.search(line))
    if not isinstance(value, dict):
        return False
    # Successful traces contain error:null, and model raw outputs may themselves say ERROR.
    if any(value.get(k) for k in ('error', 'last_error', 'traceback', 'exception')):
        return True
    if str(value.get('level', value.get('severity', ''))).upper() in ('ERROR', 'FATAL', 'CRITICAL'):
        return True
    event = str(value.get('event', value.get('type', value.get('phase', '')))).lower()
    if 'failed' in event or event in ('error', 'fatal'):
        return True
    return bool(ERROR.search(str(value.get('message', ''))))

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def read_json(p, fallback=None):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except FileNotFoundError:
        return fallback

def stat(p):
    try:
        s = pathlib.Path(p).stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    except FileNotFoundError:
        return None

def atomic(p, obj):
    q = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(q, p)

def scan_log(p, old=None, seed_at_end=False, limit=LIMIT):
    """Only advance through complete lines. Anchor catches in-place rewrites/rotation."""
    p = pathlib.Path(p)
    s = stat(p)
    if s is None:
        return {'stat': None, 'offset': 0}, {'bytes_read': 0, 'lines': [], 'pending_bytes': 0}
    old = old or {}
    if old.get('stat') == s and old.get('offset') == s[2]:
        return old, {'bytes_read': 0, 'lines': [], 'pending_bytes': 0, 'rotation_or_rewrite': False}
    offset = old.get('offset', 0)
    reset = bool(old) and (old.get('stat', [])[:2] != s[:2] or s[2] < offset)
    with p.open('rb') as f:
        if not reset and offset and old.get('anchor'):
            f.seek(max(0, offset - 4096))
            reset = hashlib.sha256(f.read(min(offset, 4096))).hexdigest() != old['anchor']
        if seed_at_end and not old:
            offset = s[2]
            raw = b''
        else:
            if reset:
                offset = 0
            f.seek(offset)
            raw = f.read(limit)
            # A sample's complete JSON may exceed the normal 1 MiB scan chunk.
            # Finish that first line without advancing through a partial record.
            # Bound the extension so a corrupt/unclosed record remains visible.
            if raw and b'\n' not in raw:
                chunks = [raw]
                total = len(raw)
                maximum_line_bytes = 64 * 1024 * 1024
                while total < maximum_line_bytes:
                    more = f.readline(min(limit, maximum_line_bytes - total))
                    if not more:
                        break
                    chunks.append(more)
                    total += len(more)
                    if more.endswith(b'\n'):
                        break
                raw = b''.join(chunks)
        cut = raw.rfind(b'\n') + 1
        complete = raw[:cut]
        offset += cut
        f.seek(max(0, offset - 4096))
        anchor = hashlib.sha256(f.read(min(offset, 4096))).hexdigest()
    return {'stat': s, 'offset': offset, 'anchor': anchor}, {
        'bytes_read': len(raw), 'lines': complete.decode('utf-8', errors='replace').splitlines(),
        'pending_bytes': max(0, s[2] - offset), 'rotation_or_rewrite': reset,
        'seeded_historical_bytes_without_rereading': s[2] if seed_at_end and not old else 0,
        'oversized_or_incomplete_line': bool(raw) and not cut,
    }

def file_digest(p, old, counters):
    s = stat(p)
    if old and old.get('stat') == s:
        return old
    counters['new_or_changed_files_hashed'] += int(s is not None)
    return {'stat': s, 'sha256': sha(p) if s is not None else None}

def project_processes():
    found = []
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = [a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            if not args or not pathlib.Path(args[0]).name.startswith('python'):
                continue
            cwd = os.readlink(p/'cwd')
            if cwd != str(R) and not cwd.startswith(str(R) + '/'):
                continue
            role = 'other_project_python'
            if any(pathlib.Path(a).name == 'run.py' for a in args[1:]):
                role = 'model'
            elif any(pathlib.Path(a).name.startswith(('verify_', 'audit_')) for a in args[1:]):
                role = 'CPU_verifier_or_audit'
            elif any(pathlib.Path(a).name == 'A100_alignment_evidence_runner_20261103.py' for a in args[1:]):
                role = 'alignment_runner'
            elif any(pathlib.Path(a).name == 'batch_910b.py' for a in args[1:]) and '--run' in args:
                role = 'legacy_controller'
            pp = re.search(r'^PPid:\s+(\d+)', (p/'status').read_text(), re.M)
            found.append({'PID': int(p.name), 'PPid': int(pp[1]) if pp else None, 'role': role, 'argv': args, 'cwd': cwd})
        except (OSError, ValueError):
            pass
    return found

def runner_lock_owners():
    s = (D/'runner.lock').stat()
    result = []
    for line in pathlib.Path('/proc/locks').read_text().splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[1] != 'FLOCK':
            continue
        device = parts[5].split(':')
        if len(device) == 3 and (int(device[0], 16), int(device[1], 16), int(device[2])) == (os.major(s.st_dev), os.minor(s.st_dev), s.st_ino):
            result.append({'PID': int(parts[4]), 'kernel_line': line})
    return result

def path_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from path_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from path_strings(v)

def main():
    M.mkdir(parents=True, exist_ok=True)
    with (M/'monitor.lock').open('a+') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(M/'checkpoint.json', {})
        initial = not old
        state = read_json(D/'state.json')
        registry = read_json(D/'registry.json')
        alignment = read_json(D/'A100_ALIGNMENT_STATUS.json', {})
        local = read_json(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json', {})
        processes = project_processes()
        locks = runner_lock_owners()
        live = [p for p in processes if p['role'] in ('model', 'CPU_verifier_or_audit', 'alignment_runner', 'legacy_controller')]
        counters = collections.Counter()
        out = {'UTC': now(), 'initial_baseline': initial, 'scope': 'routine_incremental_monitor_only_not_model_acceptance',
               'canonical_counts': dict(collections.Counter(x['status'] for x in state['models'].values())),
               'local_numeric_progress': {k: local.get(k) for k in ('real_record_count', 'numeric_within5pct', 'numeric_outside5pct', 'blank_count', 'local_workbook_sha256')},
               'numeric_match_is_not_A100_protocol_qualification': True, 'live_processes': processes,
               'runner_kernel_FLOCK_owners': locks, 'inference_or_verification_active': bool(live),
               'alignment_phase': alignment.get('phase'), 'last_alignment_model': alignment.get('current'),
               'changed_models': [], 'affected_models_requiring_targeted_review': [], 'source_errors': [],
               'new_log_error_candidates': [], 'inspection_errors': [], 'log_deltas': [],
               'unchanged_accepted_models_reusing_previous_evidence': 0}
        nxt = {'UTC': out['UTC'], 'files': {}, 'reports': {}, 'models': {}, 'logs': {}, 'alignment_job': alignment.get('job')}
        model_changes = set()
        expected = collections.defaultdict(dict)
        for mid, e in state['models'].items():
            spec = registry.get(mid, {})
            signature = {k: e.get(k) for k in ('status', 'score', 'report_dir', 'active_config', 'last_error')}
            signature['review_revision'] = spec.get('review_revision')
            nxt['models'][mid] = signature
            if not initial and old.get('models', {}).get(mid) != signature:
                out['changed_models'].append({'model_id': mid, 'before': old.get('models', {}).get(mid), 'after': signature})
                model_changes.add(mid)
            for rel, digest in spec.get('source_sha256', {}).items():
                expected[str(R/rel)][mid] = digest
            if e.get('status') == 'done' and e.get('report_dir'):
                report = pathlib.Path(e['report_dir'])
                manifest = {name: stat(report/name) for name in FIVE}
                nxt['reports'][mid] = manifest
                if any(v is None for v in manifest.values()):
                    out['inspection_errors'].append({'model_id': mid, 'missing_required_report_file': True})
                    model_changes.add(mid)
                elif not initial and old.get('reports', {}).get(mid) != manifest:
                    model_changes.add(mid)
                elif not initial:
                    out['unchanged_accepted_models_reusing_previous_evidence'] += 1
        job = read_json(alignment.get('job'), {}) if alignment.get('job') else {}
        for rel, digest in job.get('spec', {}).get('source_sha256', {}).items():
            expected[str(R/rel)][job.get('model_id', 'current_alignment_job')] = digest
        for path, owners in expected.items():
            if not pathlib.Path(path).is_relative_to(R) or '..' in pathlib.Path(path).parts:
                out['source_errors'].append({'path': path, 'reason': 'invalid_registered_source_path'})
                continue
            before = old.get('files', {}).get(path)
            current = file_digest(path, before, counters)
            nxt['files'][path] = current
            for mid, digest in owners.items():
                if current['sha256'] != digest:
                    out['source_errors'].append({'model_id': mid, 'path': path, 'expected': digest, 'actual': current['sha256']})
                    model_changes.add(mid)
            if not initial and (before or {}).get('sha256') != current['sha256']:
                model_changes.update(owners)
        for p in [D/'state.json', D/'registry.json', D/'POLICY.json', D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',
                  D/'PAUSE', pathlib.Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]:
            nxt['files'][str(p)] = file_digest(p, old.get('files', {}).get(str(p)), counters)
        out['changed_metadata_files'] = [p for p, v in nxt['files'].items() if not initial and old.get('files', {}).get(p, {}).get('sha256') != v['sha256']]
        out['PAUSE_owner'] = (read_json(D/'PAUSE', {}) or {}).get('owner')
        out['affected_models_requiring_targeted_review'] = sorted(model_changes)
        watched = {}
        for root, patterns in [(R/'configs', ('*.yaml', '*.yml')), (R/'universal_eval/model/adapters', ('*.py',)), (R/'scripts', ('*.py',))]:
            for pattern in patterns:
                for p in root.rglob(pattern):
                    if '__pycache__' not in p.parts:
                        watched[str(p)] = stat(p)
        nxt['source_tree_stats'] = watched
        out['new_or_modified_source_config_paths'] = [] if initial else [p for p, s in watched.items() if old.get('source_tree_stats', {}).get(p) != s]
        out['removed_source_config_paths'] = [] if initial else sorted(set(old.get('source_tree_stats', {})) - set(watched))
        new_job = not initial and old.get('alignment_job') != alignment.get('job')
        logs = {D/'events.jsonl', D/'controller.log'}
        attempt = pathlib.Path(alignment['attempt']) if alignment.get('attempt') else None
        if attempt:
            logs.update([attempt/'console.log', attempt/'verification_stderr.log'])
        progress_paths = set()
        report = pathlib.Path(alignment['report_dir']) if alignment.get('report_dir') else None
        if live and alignment.get('current') in state['models']:
            entry = state['models'][alignment['current']]
            parent = R/'workspace/reports'/alignment['current'].replace('/', '_')/entry['task']
            candidates = sorted(p for p in parent.iterdir() if p.is_dir()) if parent.exists() else []
            if candidates:
                report = candidates[-1]
        if report:
            logs.add(report/'run.log')
            lineage = read_json(report/'operator_lineage.json', {})
            for value in path_strings(lineage.get('actual_execution', {})):
                p = pathlib.Path(value)
                if p.is_absolute() and p.is_relative_to(R/'workspace/diagnostics/model_runs'):
                    for name in ('sample_execution.jsonl', 'generation_progress.jsonl'):
                        q = (p if p.name == name else p.parent/name if p.suffix else p/name)
                        if q.is_file():
                            progress_paths.add(q)
        logs.update(progress_paths)
        for p in sorted(logs):
            if not p.is_file():
                continue
            seed = initial and (not live or p in {D/'events.jsonl', D/'controller.log'})
            cursor, delta = scan_log(p, old.get('logs', {}).get(str(p)), seed_at_end=seed)
            nxt['logs'][str(p)] = cursor
            lines = delta.pop('lines')
            counters['log_bytes_read'] += delta['bytes_read']
            counters['historical_log_bytes_skipped'] += delta.get('seeded_historical_bytes_without_rereading', 0)
            out['log_deltas'].append(dict(path=str(p), complete_new_lines=len(lines), last_new_lines=lines[-3:], **delta))
            for line in lines:
                if is_error_candidate(line):
                    out['new_log_error_candidates'].append({'path': str(p), 'line': line[:2000], 'requires_context_review_not_automatic_model_failure': True})
            if delta['pending_bytes']:
                out['inspection_errors'].append({'path': str(p), 'unread_or_incomplete_bytes': delta['pending_bytes'], 'must_finish_before_claiming_no_new_errors': True})
        # Keep old cursors so a temporarily absent log is not scanned from zero later.
        nxt['logs'] = dict(old.get('logs', {}), **nxt['logs'])
        if alignment.get('job'):
            completion = read_json(pathlib.Path(alignment['job']).parent/'completion.json', {})
            out['actual_job_returncode'] = completion.get('actual_returncode')
            if completion.get('actual_returncode') not in (None, 0):
                out['inspection_errors'].append({'job': alignment['job'], 'actual_returncode': completion['actual_returncode']})
        if locks and not any(p['PID'] in {x['PID'] for x in locks} for p in processes):
            out['inspection_errors'].append({'reason': 'runner_lock_owner_not_in_project_process_snapshot', 'owners': locks})
        try:
            out['status_cli'] = subprocess.run(['python3', 'scripts/batch_910b.py', '--status'], cwd=R, capture_output=True, text=True, timeout=40).stdout
            q = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30)
            out['npu_smi'] = {'returncode': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr}
        except subprocess.TimeoutExpired as exc:
            out['inspection_errors'].append({'health_command_timeout': str(exc)})
        out['work_counters'] = dict(counters)
        out['not_checked_this_cycle'] = 'Unchanged historical sample records and external NPZ/probe arrays are not reread; previous full acceptance remains provenance, not a new acceptance.'
        out['publication_full_independent_verification_and_final_full_audit_still_required'] = True
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        history = M/('check_' + tag + '.json')
        atomic(history, out)
        atomic(M/'checkpoint.json', nxt)
        atomic(M/'latest.json', out)
        print(json.dumps({'report': str(history), **{k: out[k] for k in ('UTC', 'initial_baseline', 'canonical_counts', 'local_numeric_progress', 'inference_or_verification_active', 'alignment_phase', 'changed_models', 'source_errors', 'new_log_error_candidates', 'inspection_errors', 'unchanged_accepted_models_reusing_previous_evidence', 'work_counters')}}, ensure_ascii=False, indent=2))

def selftest():
    with tempfile.TemporaryDirectory(prefix='incremental_monitor_') as folder:
        p = pathlib.Path(folder)/'run.log'
        p.write_bytes(b'old ERROR historical\n')
        c, d = scan_log(p, seed_at_end=True)
        assert not d['lines'] and d['bytes_read'] == 0
        with p.open('ab') as f:
            f.write(b'new ERROR first\npartial Runtime')
        c, d = scan_log(p, c)
        assert d['lines'] == ['new ERROR first'] and d['pending_bytes'] > 0
        with p.open('ab') as f:
            f.write(b'Error last\n')
        c, d = scan_log(p, c)
        assert d['lines'] == ['partial RuntimeError last']
        c, d = scan_log(p, c)
        assert not d['lines'] and d['bytes_read'] == 0
        p.write_bytes(b'rewritten ERROR new file\n')
        c, d = scan_log(p, c)
        assert d['rotation_or_rewrite'] and d['lines'] == ['rewritten ERROR new file']
        p.rename(p.with_suffix('.old'))
        p.write_bytes(b'rotation ERROR new inode\n')
        c, d = scan_log(p, c)
        assert d['rotation_or_rewrite'] and d['lines'] == ['rotation ERROR new inode']
        count = collections.Counter()
        a = file_digest(p, None, count)
        b = file_digest(p, a, count)
        assert a == b and count['new_or_changed_files_hashed'] == 1
        previous_stat = p.stat()
        p.write_bytes(b'changed same size ??????\n')
        # The fixture changes within one FS clock tick; model the later check's timestamp explicitly.
        os.utime(p, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000))
        b = file_digest(p, b, count)
        assert a['sha256'] != b['sha256'] and count['new_or_changed_files_hashed'] == 2
    assert not is_error_candidate('{"error":null,"raw_output":"ERROR printed on image"}')
    assert not is_error_candidate('{"error_count":0,"raw_output":"RuntimeError"}')
    assert is_error_candidate('{"error":"RuntimeError actual failed kernel"}')
    assert is_error_candidate('Traceback (most recent call last):')
    print(json.dumps({'passed': True, 'structured_error_null_and_model_raw_not_false_alarms': True, 'tests': ['historical_log_seed_no_realert', 'append_only_new_lines', 'partial_line_preserved', 'unchanged_no_reread', 'in_place_rewrite', 'rotation_inode', 'unchanged_digest_cache', 'changed_digest_recheck'], 'no_production_files_or_runner_lock_touched': True}))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    selftest() if args.self_test else main()

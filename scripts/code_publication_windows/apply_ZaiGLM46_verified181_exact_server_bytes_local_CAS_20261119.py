"""Copy the verified server transaction bytes after ArtifactTool authoring and visual QA.

No spreadsheet is authored here. This CAS preserves the frozen writer's exact
ZIP members and checks that only the previously blank target G value changed.
"""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import base64, hashlib, json, os, re, shutil, zipfile, xml.etree.ElementTree as E
P = Path(__file__).resolve().parent
H = P / 'ZaiGLM46Flash_local_result_sync_verified181_20261119'
q = json.loads((H / 'plan.json').read_bytes())
bundle = json.loads((P / 'R_ZaiGLM46_verified181_server_book_for_local_CAS_20261119.json').read_bytes())
assert bundle['ready_for_local_CAS'] and bundle['same_original_FLOCK_held'] and bundle['publication_actual_model_and_verifier_inactive']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
target = Path(q['target']).resolve()
assert target.parent == Path('D:/code-lv').resolve() and target.name == 'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'
if (H / 'local_publication_receipt.json').exists():
    old = json.loads((H / 'local_publication_receipt.json').read_bytes())
    assert old['completed'] and old['after_sha256'] == sha(target) == bundle['workbook_SHA256']
    print(json.dumps(dict(already_completed_no_rewrite=True, receipt=old), ensure_ascii=False))
    raise SystemExit(0)
before = target.read_bytes()
assert hashlib.sha256(before).hexdigest() == q['expected_before_sha256'] == sha(q['source_backup'])
data = base64.b64decode(bundle['workbook_base64'], validate=True)
assert hashlib.sha256(data).hexdigest() == bundle['workbook_SHA256']
source = H / 'verified_server_exact_bytes.xlsx'
if source.exists():
    assert source.read_bytes() == data
else:
    with source.open('xb') as f:
        f.write(data)
author = json.loads((H / 'artifact_authored_G_values.json').read_bytes())
assert author[695][0] == q['changes'][0]['new_value']
review = json.loads((H / 'actual_PNG_visual_review.json').read_bytes())
assert review['before_and_author_after_actually_viewed'] and review['before_SHA256'] == sha(H / 'before_697.png') and (review['after_SHA256'] == sha(H / 'after_697.png'))
assert bundle['publication_receipt']['targets'][0]['model_id'] == q['changes'][0]['model_id'] and Decimal(str(bundle['publication_receipt']['targets'][0]['new_G'])) == Decimal(str(q['changes'][0]['new_value']))
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(q['source_backup']) as a, zipfile.ZipFile(source) as b:
    assert a.namelist() == b.namelist() and a.comment == b.comment and (b.testzip() is None)
    for info in a.infolist():
        other = b.getinfo(info.filename)
        for attr in ['date_time', 'compress_type', 'external_attr', 'create_system', 'extra', 'comment']:
            assert getattr(info, attr) == getattr(other, attr)
        old = a.read(info.filename)
        new = b.read(info.filename)
        if info.filename != q['sheet_xml']:
            assert old == new
            continue
        cells = lambda raw: {c.get('r'): c for c in E.fromstring(raw).findall('.//s:c', ns)}
        ac, bc = (cells(old), cells(new))
        assert {k for k in set(ac) | set(bc) if (E.tostring(ac[k]) if k in ac else None) != (E.tostring(bc[k]) if k in bc else None)} == {'G697'}
        assert 'G697' in bc and (ac.get('G697') is None or ac['G697'].find('s:f', ns) is None)
        assert Decimal(bc['G697'].findtext('s:v', namespaces=ns)) == Decimal(str(q['changes'][0]['new_value']))
        if 'G697' in ac:
            assert ac['G697'].get('s') == bc['G697'].get('s')
        pattern = b'<c\\b(?=[^>]*\\br="G697")[^>]*(?:/>|>.*?</c>)'
        assert len(re.findall(pattern, old, flags=re.S)) <= 1 and len(re.findall(pattern, new, flags=re.S)) == 1
        assert re.sub(pattern, b'', old, flags=re.S) == re.sub(pattern, b'', new, flags=re.S)
st = target.stat()
fresh = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
backup = H / 'local_CAS_immediate_backup.xlsx'
if not backup.exists():
    shutil.copy2(target, backup)
assert sha(backup) == q['expected_before_sha256']
temp = target.with_name(target.name + '.owned_ZaiGLM46_verified181_' + str(os.getpid()) + '.tmp')
assert not temp.exists()
with temp.open('xb') as f:
    f.write(data)
assert sha(temp) == bundle['workbook_SHA256']
st2 = target.stat()
assert dict(size=st2.st_size, mtime_ns=st2.st_mtime_ns, ctime_ns=st2.st_ctime_ns) == fresh and sha(target) == q['expected_before_sha256']
os.replace(temp, target)
assert sha(target) == bundle['workbook_SHA256']
after = target.stat()
receipt = dict(completed=True, UTC=datetime.now(timezone.utc).isoformat(), target=str(target), backup=str(backup), before_sha256=q['expected_before_sha256'], after_sha256=sha(target), changed_G_cells=['G697'], only_target_G_changed=True, all_non_G_formulas_styles_rows_ZIP_metadata_and_members_preserved=True, all_prior180_C_B_D_G_preserved=True, artifact_tool_authored_target_numeric_values={'G697': q['changes'][0]['new_value']}, source_server_receipt_SHA256=bundle['publication_receipt_SHA256'], fresh_stat_before=fresh, fresh_stat_after=dict(size=after.st_size, mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns), actual_PNG_visual_review_SHA256=sha(H / 'actual_PNG_visual_review.json'))
with (H / 'local_publication_receipt.json').open('x', encoding='utf8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
print(json.dumps(receipt, ensure_ascii=False))

import sys
sys.dont_write_bytecode = True
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
import io
import json
import zipfile
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
book = Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
export_path = P / 'PrithivGemma4E4B_verified180_incremental_real_results_export_20261117.json'
audit_packet = json.loads((P / 'R_heartbeat1916_Prithiv180_final_dual_target_audit_actual_20261117.json').read_bytes())
audit = next(v['value'] for k, v in audit_packet['files'].items() if k.endswith('/full_target_dual_publication_audit.json'))
raw = book.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
export_raw = export_path.read_bytes()
export = json.loads(export_raw)
assert digest == audit['local_and_server_book_SHA256'] == export['workbook_SHA256']
assert hashlib.sha256(export_raw).hexdigest() == audit['verified180_export_SHA256']
assert (len(export['records']), export['measured_count']) == (180, 180)
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        shared = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', ns)]
    root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    cells = {}
    for cell in root.findall('.//s:c', ns):
        value = cell.findtext('s:v', namespaces=ns)
        if cell.get('t') == 's' and value is not None:
            value = shared[int(value)]
        elif cell.get('t') == 'inlineStr':
            node = cell.find('s:is', ns)
            value = ''.join(node.itertext()) if node is not None else ''
        cells[cell.get('r')] = value
models = {}
for address, value in cells.items():
    if address.startswith('C') and address[1:].isdigit() and value and '/' in value:
        assert value not in models, ('duplicate C', value)
        models[value] = int(address[1:])
assert len(models) == 700
round9 = set(export['round9_legal_exact_models'])
assert len(round9) == 29
changes = []
within = outside = 0
for record in export['records']:
    model = record['model_id']
    row = models[model]
    assert row == record['current_row']
    assert cells[f'B{row}'] == record['task']
    assert Decimal(cells[f'D{row}']) == Decimal(record['current_D'])
    actual = Decimal(cells[f'G{row}'])
    exact = Decimal(str(record['actual_canonical_metric']))
    legal = {exact}
    if model in round9:
        legal.add(Decimal(str(round(float(record['actual_canonical_metric']), 9))))
    if actual not in legal:
        changes.append({'model': model, 'cell': f'G{row}', 'actual': str(actual), 'legal': sorted(map(str, legal))})
    assert actual == Decimal(record['current_G'])
    reference = Decimal(cells[f'D{row}'])
    if reference != 0:
        if abs(reference - actual) / abs(reference) <= Decimal('.05'):
            within += 1
        else:
            outside += 1
assert changes == []
assert (within, outside) == (165, 15)
blank = sum(cells.get(f'G{row}') in (None, '') for row in models.values())
assert blank == export['blank_count'] == 520
assert audit['confirmed310_missing_G_all_blank'] and export['confirmed_missing'] == 310
assert export['current_original_A100_protocol_qualified'] == 66
assert book.read_bytes() == raw
st = book.stat()
review = {
    'UTC': datetime.now(timezone.utc).isoformat(), 'passed': True,
    'workbook': str(book), 'SHA256': digest,
    'fresh_stat': {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns},
    'export': str(export_path), 'export_SHA256': hashlib.sha256(export_raw).hexdigest(),
    'C_unique_models': 700, 'all_180_C_B_D_G_exact_or_authorized_round9': True,
    'round9_allowed_models': 29, 'numeric_within5pct': within, 'numeric_outside5pct': outside,
    'complete_weights_pending': 210, 'confirmed_missing': 310, 'full_qualification': 66,
    'model_G_blank': blank, 'missing_310_blank_proof': 'Exact same workbook SHA as full server target audit with explicit 310-missing check',
    'repeat_publication_plan_changes': changes, 'changes_count': len(changes),
    'no_workbook_write': True,
}
out = P / 'PrithivGemma4E4B_verified180_local_complete_records_fresh_stat_review_20261117.json'
out.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(review, ensure_ascii=False))

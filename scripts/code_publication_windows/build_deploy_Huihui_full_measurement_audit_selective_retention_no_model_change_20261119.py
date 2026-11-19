import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,zipfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
names={
 'audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119.py':'scripts/maintenance/audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119.py',
 'retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py':'scripts/maintenance/retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py',
 'test_Huihui_completed200_baselinegap_selective_retention_wrong_controls_20261119.py':'scripts/tests/test_Huihui_completed200_baselinegap_selective_retention_wrong_controls_20261119.py',
}
payload={}
for n,target in names.items():
 raw=(P/n).read_bytes();ast.parse(raw);payload[target]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes_BASE64=base64.b64encode(raw).decode())
book=P.parent.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'
assert book==Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
raw=book.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 cells=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));shared=[]
 if 'xl/sharedStrings.xml' in z.namelist():shared=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
 def cell(n):
  e=cells.find('.//s:c[@r="'+n+'"]',ns)
  if e is None:return None
  v=e.findtext('s:v',namespaces=ns)
  return shared[int(v)] if e.get('t')=='s' and v is not None else ''.join(e.find('s:is',ns).itertext()) if e.get('t')=='inlineStr' else v
 snap=dict(UTC=datetime.now(timezone.utc).isoformat(),local_book_path=str(book),book_SHA256=h,book_stat=dict(size=book.stat().st_size,mtime_ns=book.stat().st_mtime_ns),model_id=cell('C355'),task=cell('B355'),D=cell('D355'),G=cell('G355'),whole_local_current_book_byte_identical_to_verified180_server_baseline=True,no_workbook_write=True)
assert snap['model_id']=='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated' and snap['task']=='image-text-to-text' and snap['D']=='0.9450' and snap['G'] in ['',None]
snapraw=(json.dumps(snap,ensure_ascii=False,indent=2)+'\n').encode();(P/'Huihui_completed200_baselinegap_local_current_snapshot_20261119.json').write_bytes(snapraw)
payload['workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Huihui_completed200_full_numerical_baselinegap_target_audit_and_correct_nonexample_retention_20261119/local_current_snapshot.json']=dict(SHA256=hashlib.sha256(snapraw).hexdigest(),bytes_BASE64=base64.b64encode(snapraw).decode())
encoded=base64.b64encode(json.dumps(payload,ensure_ascii=False).encode()).decode()
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
R=Path('/root/projects/comperror')
payload=json.loads(base64.b64decode(PAYLOAD))
for n,v in payload.items():
 p=R/n;raw=base64.b64decode(v['bytes_BASE64']);assert p.resolve().is_relative_to(R) and hashlib.sha256(raw).hexdigest()==v['SHA256']
 if p.suffix=='.py':ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw,('New immutable destination exists with different bytes',n)
for n,v in payload.items():
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_bytes(base64.b64decode(v['bytes_BASE64']))
print(json.dumps(dict(passed=True,new_files={n:v['SHA256'] for n,v in payload.items()},no_live_source_job_config_or_model_mutation=True),ensure_ascii=False))
'''.replace('PAYLOAD',repr(encoded))
ast.parse(remote);(P/'deploy_Huihui_full_measurement_audit_selective_retention_no_model_change_20261119_remote.py').write_text(remote,encoding='utf8')
print(json.dumps(dict(source_SHA256={n:v['SHA256'] for n,v in payload.items()},local_snapshot_SHA256=hashlib.sha256(snapraw).hexdigest(),ready=True),ensure_ascii=False))

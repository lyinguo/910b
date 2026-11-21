"""Fresh current183 workbook check and narrow ArtifactTool plan; no canonical write."""
from pathlib import Path
from decimal import Decimal
import json,hashlib,zipfile,xml.etree.ElementTree as E,shutil,re
V=Path(__file__).resolve().parent;H=V/'OvisU1_local_result_sync_verified184_20261121';H.mkdir(exist_ok=True)
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
before='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(book)==before
baseline=json.loads((V/'AutoGLMPhone9BSingle_verified183_incremental_real_results_export_20261120.json').read_bytes());assert baseline['measured_count']==len(baseline['records'])==183 and baseline['workbook_SHA256']==before
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 strings=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
 sheet=E.fromstring(z.read('xl/workbook.xml')).find('s:sheets/s:sheet',ns);sheet_name=sheet.get('name');sheet_xml='xl/worksheets/sheet1.xml';root=E.fromstring(z.read(sheet_xml));cells={c.get('r'):c for c in root.findall('.//s:c',ns)}
 def val(cell):
  c=cells.get(cell)
  if c is None:return None
  v=c.findtext('s:v',namespaces=ns)
  if c.get('t')=='s':return strings[int(v)] if v is not None else None
  if c.get('t')=='inlineStr':return ''.join(c.find('s:is',ns).itertext())
  return v
 rows={int(k[1:]):val(k) for k in cells if re.fullmatch('C[0-9]+',k) and val(k)}
 assert len(rows)==701 # header plus700 unique models
 assert len(set(rows.values()))==701
 measured=[r for r in rows if r>1 and val('G'+str(r)) not in [None,'']];assert len(measured)==183
 for rec in baseline['records']:
  r=next(r for r,mid in rows.items() if mid==rec['model_id']);assert val('B'+str(r))==rec['task'] and Decimal(val('D'+str(r)))==Decimal(rec['current_D']) and Decimal(val('G'+str(r)))==Decimal(rec['current_G'])
 row,=[r for r,mid in rows.items() if mid=='AIDC-AI/Ovis-U1-3B'];assert row==144 and val('B144')=='image-text-to-text' and val('G144') in [None,''] and Decimal(val('D144'))==Decimal('.875') and cells.get('G144',E.Element('c')).find('s:f',ns) is None
last=max(rows);backup=H/'original_user_workbook.xlsx'
if backup.exists():assert sha(backup)==before
else:shutil.copy2(book,backup)
st=book.stat();change=dict(model_id='AIDC-AI/Ovis-U1-3B',task='image-text-to-text',row=row,cell='G144',old_value=None,new_value=.895,A100_full_precision_cached_D=val('D144'),relative_error=str(abs(Decimal('.895')-Decimal('.875'))/Decimal('.875')),new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir='/root/projects/comperror/workspace/reports/AIDC-AI_Ovis-U1-3B/image-text-to-text/20261121_093507')
q=dict(target=str(book),source_backup=str(backup),expected_before_sha256=before,sheet_name=sheet_name,sheet_xml=sheet_xml,last_model_row=last,changes=[change],current_measured_count=183,real_measured_count_if_qualified=184,numeric_within5pct_if_qualified=169,numeric_outside5pct=15,complete_weights_pending_formal_if_qualified=206,confirmed_missing=310,original_protocol_qualified_if_qualified=70,model_blank_count_if_qualified=516,complete_numeric_verifier_passed=False,all_prior183_C_B_D_G_checked=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),qualification_and_G_write_not_yet_claimed=True)
p=H/'plan.json';assert not p.exists();p.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(plan=str(p),fresh_book_SHA256=before,current183_records_exact=True,row=row,last_model_row=last,provisional_score=.895,no_canonical_write=True),ensure_ascii=False))

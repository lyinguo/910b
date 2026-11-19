"""Preserve the exact current local180 workbook and preview the one empty G cell."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,shutil,zipfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent;book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(book)=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 wb=ET.fromstring(z.read('xl/workbook.xml'));first=wb.find('s:sheets/s:sheet',ns);sheet_name=first.get('name');rid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id');rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(x.get('Target') for x in rels if x.get('Id')==rid);sheet_xml=target.lstrip('/') if target.startswith('/') else 'xl/'+target
 strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else [];values={}
 for cell in ET.fromstring(z.read(sheet_xml)).findall('.//s:c',ns):
  v=cell.findtext('s:v',namespaces=ns);values[cell.get('r')]=strings[int(v)] if cell.get('t')=='s' else ''.join(cell.find('s:is',ns).itertext()) if cell.get('t')=='inlineStr' else v
ids={v:int(k[1:]) for k,v in values.items() if k.startswith('C') and k!='C1' and v};MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated';assert len(ids)==700 and ids[MID]==355 and values['B355']=='image-text-to-text' and Decimal(values['D355'])==Decimal('.945') and values.get('G355') in [None,'']
baseline=json.loads((P/'PrithivGemma4E4B_verified180_incremental_real_results_export_20261117.json').read_bytes());assert len(baseline['records'])==baseline['measured_count']==180
for old in baseline['records']:
 row=ids[old['model_id']];assert values[f'B{row}']==old['task'] and Decimal(values[f'D{row}'])==Decimal(old['current_D']) and Decimal(values[f'G{row}'])==Decimal(old['current_G'])
H=P/'HuihuiGLM46Flash_local_result_sync_verified181_20261119';H.mkdir(exist_ok=False);backup=H/'original_user_workbook.xlsx';shutil.copy2(book,backup);st=book.stat()
plan=dict(target=str(book),source_backup=str(backup),expected_before_sha256=sha(book),sheet_name=sheet_name,sheet_xml=sheet_xml,last_model_row=max(ids.values()),changes=[dict(model_id=MID,task='image-text-to-text',row=355,cell='G355',old_value=None,new_value=None,A100_full_precision_cached_D=values['D355'])],preview_only_no_predicted200_score=True,all_prior180_C_B_D_G_checked=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
(H/'preview_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=(P/'prithiv_workbook_cell_verified180_20261117.mjs').read_text(encoding='utf8').replace('B570:G574','B353:G357')
(P/'huihui_glm46_workbook_cell_verified181_20261119.mjs').write_text(s,encoding='utf8',newline='\n')
print(json.dumps(dict(plan=str(H/'preview_plan.json'),before_SHA256=sha(book),target_cell='G355',old180_preserved=True,no_predicted_score_or_production_write=True),ensure_ascii=False))

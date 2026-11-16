from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as E
from decimal import Decimal
P=Path('D:/code-lv/.tmp/batch_910b_20261030');data=json.loads((P/'R_OSAtlas_verified170_complete_export_Falcon_CPU_actual_20261116.json').read_bytes());ex=data['export']
raw=(json.dumps(ex,ensure_ascii=False,indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==data['export_SHA256'];dest=P/'OSAtlas4B_verified170_incremental_real_results_export_20261116.json';dest.write_bytes(raw)
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');bh=hashlib.sha256(book.read_bytes()).hexdigest();assert bh==ex['server_book_SHA256']==ex['workbook_SHA256']=='195e9af1a5284c764c795516853256bca54e9880ddd7b388cfefaa9a3c31b496'
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 strings=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
 vals={c.get('r'):(''.join(c.find('s:is',ns).itertext()) if c.get('t')=='inlineStr' else strings[int(c.findtext('s:v',namespaces=ns))] if c.get('t')=='s' else c.findtext('s:v',namespaces=ns)) for c in E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:c',ns)}
ids={v:int(k[1:]) for k,v in vals.items() if k.startswith('C') and k!='C1' and v};assert len(ids)==700
within=0;rounds=0
for r in ex['records']:
 row=ids[r['model_id']];assert vals[f'B{row}']==r['task'] and Decimal(vals[f'D{row}'])==Decimal(r['current_D']) and Decimal(vals[f'G{row}'])==Decimal(r['current_G'])
 score=Decimal(vals[f'G{row}']);actual=Decimal(str(r['actual_canonical_metric']))
 if score!=actual:assert r['model_id'] in ex['round9_legal_exact_models'] and score==Decimal(str(round(r['actual_canonical_metric'],9)))
 rounds+=r['model_id'] in ex['round9_legal_exact_models'];d=Decimal(vals[f'D{row}']);within+=int(abs(d-score)/abs(d)<=Decimal('.05') if d else score==0)
assert len(ex['records'])==170 and within==155 and rounds==29
s=book.stat();result=dict(passed=True,changes=0,local_and_server_book_SHA256=bh,all170_current_C_B_D_G_checked=True,numeric_within5pct=155,numeric_outside5pct=15,complete_weights_pending_formal=220,confirmed_missing=310,original_protocol_qualified_delivered=56,legal_round9_models=29,export=str(dest),export_SHA256=data['export_SHA256'],stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns),all_old169_reports_not_rerun=True,workbook_not_written=True)
(P/'OSAtlas4B_repeat_verified170_plan0_fresh_local_stat_20261116.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

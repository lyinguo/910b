import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import workbook_rows
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,sheet,_=workbook_rows(book);state=json.loads((D/'state.json').read_bytes());ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:root=E.fromstring(z.read(sheet))
out=[]
for mid in ['microsoft/Florence-2-large-ft','F16/florence2-large-ft-gufeng_v3','AskUI/PTA-1']:
 row,=[r for r in rows if r['model_id']==mid];dc=root.find('.//s:c[@r="D'+str(row['row'])+'"]',ns);dv=dc.findtext('s:v',namespaces=ns)
 out.append(dict(model_id=mid,table_row=row,actual_A100_D=dv,canonical_state=state['models'][mid]))
resources={}
for n in ['workspace/diagnostics/a100_high_nine_20261015/reference_dataset/eval.jsonl','workspace/datasets/image-to-text-short/eval.jsonl']:
 p=R/n;rec=dict(exists=p.is_file(),path=str(p))
 if p.is_file():b=p.read_bytes();rec.update(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),first_original_row=json.loads(b.splitlines()[0]))
 resources[n]=rec
print(json.dumps(dict(book_SHA256=hashlib.sha256(book.read_bytes()).hexdigest(),own_table_and_state_bindings=out,own_original_dataset_resource_check=resources,no_task_or_data_substitution_no_forward=True),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
paths=['universal_eval/model/adapters/glm41v_base_discriminative.py','universal_eval/model/adapters/gemma4_assistant_adapter.py','universal_eval/model/adapters/high_nine_llava_reference.py','scripts/collect_code_publication_primary_sources.py']
out={'sources':[]}
for rel in paths:
 p=R/rel;b=p.read_bytes();out['sources'].append(dict(path=rel,SHA256=hashlib.sha256(b).hexdigest(),text=b.decode()))
from batch_910b_support import workbook_rows
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,_,_=workbook_rows(book);ids=['zai-org/GLM-4.1V-9B-Base','google/gemma-4-E2B-it-assistant','ICTNLP/llava-mini-llama-3.1-8b']
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
out['actual_workbook_rows']=[]
for mid in ids:
 row,=[x for x in rows if x['model_id']==mid];cell=sheet.find('.//s:c[@r="D'+str(row['row'])+'"]',ns);out['actual_workbook_rows'].append(dict(row=row,current_D=cell.findtext('s:v',namespaces=ns)))
print(json.dumps(out,ensure_ascii=False))

from pathlib import Path
import json,zipfile,xml.etree.ElementTree as E
P=Path(__file__).resolve().parent;d=json.loads((P/'R_Huihui_serial200_next_GLM_storage_current10_20261119.json').read_bytes());print(json.dumps(dict(project_roles=[dict(PID=x['PID'],role=x['role'],argv=x['argv']) for x in d['actual_NN_NF']]),ensure_ascii=False))
q=json.loads((P/'HuihuiGLM46Flash_local_result_sync_verified181_20261119/preview_plan.json').read_bytes());ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(q['source_backup']) as z:
 raw=z.read(q['sheet_xml']);c=E.fromstring(raw).find('.//s:c[@r="G355"]',ns);i=raw.find(b'r="G355"');print(json.dumps(dict(original_G_cell_attributes=dict(c.attrib) if c is not None else None,formula=c.findtext('s:f',namespaces=ns) if c is not None else None,raw_context=raw[max(0,i-4):i+85].decode(),original_book_not_changed=True),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,struct,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';mid='zai-org/GLM-4.1V-9B-Base';root=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration/zai-org/GLM-4.1V-9B-Base');out={'model_id':mid,'own_inventory_root':str(root),'assets':{},'files':{}}
for rel in ['universal_eval/model/adapters/glm41v_base_discriminative.py','configs/Glm4vForConditionalGeneration/zai-org--GLM-4.1V-9B-Base.yaml']:
 p=R/rel;b=p.read_bytes();out['files'][str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_text=b.decode())
for name in ['config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json']:
 p=root/name
 if p.is_file():
  b=p.read_bytes();out['assets'][name]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_text=b.decode())
headers={}
for p in root.glob('*.safetensors'):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
 headers[str(p)]=dict(bytes=p.stat().st_size,header=header)
out['own_source_headers_only_not_weightload_NN_or_qualification']=headers
sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import workbook_rows
book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';rows,sheet,_=workbook_rows(book);row,=[v for v in rows if v['model_id']==mid];out['actual_workbook_row']=row
with zipfile.ZipFile(book) as z:
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};tree=ET.fromstring(z.read(sheet));out['actual_full_D']=tree.find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
print(json.dumps(out,ensure_ascii=False))

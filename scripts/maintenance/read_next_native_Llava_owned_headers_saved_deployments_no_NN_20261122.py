"""Read only own next-candidate saved configurations and headers during serial200."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,struct,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import workbook_rows
import yaml
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
state=json.loads((D/'state.json').read_bytes());book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';rows,sheet,_=workbook_rows(book);out=[]
for mid in ['xtuner/llava-llama-3-8b-v1_1-hf','liuhaotian/llava-v1.5-13b']:
 e=state['models'][mid];r,=[r for r in rows if r['model_id']==mid];item=dict(model_id=mid,entry=e,actual_workbook_row=r,configurations=[],own_weight_roots=[],new_learned_forward=False)
 with zipfile.ZipFile(book) as z:
  ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};item['server_D']=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(r['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 for p in (R/'configs').glob('*/'+mid.replace('/','--')+'.yaml'):
  cfg=yaml.safe_load(p.read_bytes());s=cfg['model']['adapter'].split(':')[0];source=R/(s if s.endswith('.py') else s.replace('.','/')+'.py')
  item['configurations'].append(dict(path=str(p),SHA256=sha(p),value=cfg,original_source=dict(path=str(source),SHA256=sha(source)),saved_original_data_exists=Path(cfg['data']['dataset_path']).is_file(),saved_original_model_path_exists=Path(cfg['model']['model_path']).exists()))
 for root in e['weight_paths']:
  w=Path(root);wr=dict(path=root,readable=w.is_dir())
  if w.is_dir():
   wr['files']=[dict(name=p.name,bytes=p.stat().st_size) for p in w.iterdir() if p.is_file()];wr['assets']={}
   for n in ['config.json','generation_config.json','processor_config.json','preprocessor_config.json','tokenizer_config.json','model.safetensors.index.json']:
    p=w/n
    if p.is_file():wr['assets'][n]=dict(SHA256=sha(p),value=json.loads(p.read_bytes()))
   ip=w/'model.safetensors.index.json';shards=sorted(set(json.loads(ip.read_bytes())['weight_map'].values())) if ip.is_file() else sorted(p.name for p in w.glob('*.safetensors'));wr['headers']={}
   for n in shards:
    p=w/n
    with p.open('rb') as f:size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size)
    header=json.loads(raw);wr['headers'][n]=dict(bytes=p.stat().st_size,header_SHA256=hashlib.sha256(raw).hexdigest(),tensors={k:dict(shape=v['shape'],dtype=v['dtype']) for k,v in header.items() if k!='__metadata__'})
  item['own_weight_roots'].append(wr)
 out.append(item)
print(json.dumps(out,ensure_ascii=False))

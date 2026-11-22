"""Read own saved deployment and physical headers for smaller next candidates."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,struct,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
import yaml
from batch_910b_support import workbook_rows
state=json.loads((D/'state.json').read_bytes());book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';rows,sheet,_=workbook_rows(book);out=[]
for mid,arch in [('unsloth/medgemma-4b-pt','Gemma3ForConditionalGeneration'),('unsloth/gemma-4-E4B-it-MLX-8bit','Gemma4ForConditionalGeneration')]:
 cfgfile=R/'configs'/arch/(mid.replace('/','--')+'.yaml');entry=state['models'][mid];record=dict(model_id=mid,entry=entry,config_exists=cfgfile.is_file(),weight_roots=[],actual_workbook_rows=[r for r in rows if r['model_id']==mid])
 if cfgfile.is_file():
  raw=cfgfile.read_bytes();cfg=yaml.safe_load(raw);record.update(saved_config=dict(path=str(cfgfile),SHA256=hashlib.sha256(raw).hexdigest(),text=raw.decode()),source=[])
  module=cfg.get('model',{}).get('adapter','').split(':')[0]
  if module:
   path=R/module if module.endswith('.py') else R/(module.replace('.','/')+'.py')
   if path.is_file():record['source'].append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),text=path.read_text()))
 for root in entry['weight_paths']:
  w=Path(root);z=dict(path=root,is_dir=w.is_dir())
  if w.is_dir():
   z['top_files']=[dict(name=p.name,bytes=p.stat().st_size) for p in w.iterdir() if p.is_file()];z['assets']={}
   for name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json']:
    p=w/name
    if p.is_file():z['assets'][name]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),value=json.loads(p.read_bytes()))
   index=w/'model.safetensors.index.json';shards=sorted(set(json.loads(index.read_bytes())['weight_map'].values())) if index.is_file() else sorted(p.name for p in w.glob('*.safetensors'))
   z['headers']={}
   for name in shards:
    p=w/name
    with p.open('rb') as f:count=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(count))
    z['headers'][name]=dict(bytes=p.stat().st_size,tensors={k:dict(shape=t['shape'],dtype=t['dtype']) for k,t in header.items() if k!='__metadata__'})
  record['weight_roots'].append(z)
 out.append(record)
print(json.dumps(out,ensure_ascii=False))

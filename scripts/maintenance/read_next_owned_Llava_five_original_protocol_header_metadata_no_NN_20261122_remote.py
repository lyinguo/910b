import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,struct,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';state=json.loads((D/'state.json').read_bytes());out={}
book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 ss=[]
 if 'xl/sharedStrings.xml' in z.namelist():ss=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
 sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
 rows={}
 for row in sheet.findall('.//s:sheetData/s:row',ns):
  c={}
  for cell in row:
   a=cell.attrib.get('r','');n=''.join(x for x in a if x.isalpha());t=cell.attrib.get('t');v=cell.find('s:v',ns);v=v.text if v is not None else None
   if t=='s' and v is not None:v=ss[int(v)]
   elif t=='inlineStr':v=''.join(cell.find('s:is',ns).itertext())
   c[n]=v
  if c.get('C'):rows[c['C']]={'row':int(row.attrib['r']),'task':c.get('B'),'D':c.get('D'),'G':c.get('G')}
for mid in ['aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning','HancomInSpaceAI/HiLLaVA-llama_3_1-8b','remyxai/SpaceLLaVA','ytu-ce-cosmos/Turkish-LLaVA-v0.1','liuhaotian/llava-v1.5-13b']:
 entry=state['models'][mid];wp=Path(entry['weight_paths'][0]);cfg=R/'configs/LlavaLlamaForCausalLM'/(mid.replace('/','--')+'.yaml');x=dict(model_id=mid,current_server=rows.get(mid),state=entry,own_weight_path=str(wp),root_readable=wp.is_dir(),original_R_YAML=str(cfg),original_R_YAML_SHA256=hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.is_file() else None,original_R_YAML_complete_text=cfg.read_text() if cfg.is_file() else None)
 if wp.is_dir():
  assets={}
  for n in ['config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','processor_config.json','chat_template.jinja','special_tokens_map.json','model.safetensors.index.json','pytorch_model.bin.index.json']:
   p=wp/n
   if p.is_file():
    b=p.read_bytes();assets[n]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),value=json.loads(b) if p.suffix=='.json' else b.decode())
  x['own_metadata_assets']=assets;index=assets.get('model.safetensors.index.json',{}).get('value',{}).get('weight_map',{});shards=sorted(set(index.values())) if index else sorted(p.name for p in wp.glob('*.safetensors'))
  head={};all_names=[]
  for n in shards:
   p=wp/n
   if not p.is_file():head[n]=dict(missing=True);continue
   with p.open('rb') as f:size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size);h=json.loads(raw)
   tensors={k:v for k,v in h.items() if k!='__metadata__'};all_names+=list(tensors);head[n]=dict(bytes=p.stat().st_size,header_SHA256=hashlib.sha256(raw).hexdigest(),tensor_count=len(tensors),dtype_counts={t:sum(v['dtype']==t for v in tensors.values()) for t in {v['dtype'] for v in tensors.values()}},special_tensors={k:v for k,v in tensors.items() if any(t in k for t in ['lm_head','embed_tokens','mm_projector','multi_modal_projector','patch_embedding'])})
  x.update(own_SF_header_metadata_only=head,source_tensor_count=len(all_names),own_vision_source_names=[n for n in all_names if 'vision' in n][:8],all_own_source_names=all_names)
 out[mid]=x
print(json.dumps(dict(no_model_NN_CPU_learned_or_second_NPU_process=True,all_project_canonical_untouched=True,candidates=out),ensure_ascii=False))

import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,inspect,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
MID='zai-org/AutoGLM-Phone-9B-Multilingual';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')/MID
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
import yaml,torch
from transformers import AutoProcessor
from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
from batch_910b_support import workbook_rows
torch.set_num_threads(2)
cp=R/'configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B-Multilingual.yaml';cfg=yaml.safe_load(cp.read_bytes());data=cfg['data'];path=Path(data['dataset_path']);ds=DatasetLoader().load(str(path));ds=TaskDataRegistry._resolve_media_paths(ds,str(path.parent));before=len(ds);ds=DataPreprocessor().preprocess(ds,task_type=cfg['task_type'],apply_template=data['apply_template'],filter_empty=data['filter_empty']);assert len(ds)>=200
proc=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);ip=proc.image_processor
book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,sheet,_=workbook_rows(book);row,=[x for x in rows if x['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:sd=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
b=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');snap=Path(b['current_user_workbook_D_reference_snapshot']);orig=Path(b['local_A100_reference_snapshot']);assert sha(snap)==b['current_user_workbook_D_reference_snapshot_sha256'] and sha(orig)==b['local_A100_reference_snapshot_sha256']
values=dict(server=sd,local=js(snap)['models'][MID]['A100_D'],preserved_original=js(orig)['models'][MID]['A100_D'])
result=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,own_saved_config_SHA256=sha(cp),C_unique_row=row['row'],three_D=values,book_SHA256=sha(book),dataset_path=str(path),dataset_SHA256=sha(path),original_preprocess=data,dataset_before=before,dataset_after=len(ds),first200_missing_images=[dict(index=i,path=x['media_path']) for i,x in enumerate(ds[:200]) if not Path(x['media_path']).is_file()],processor_class=type(proc).__name__,image_processor_class=type(ip).__name__,own_image_processor_dict=ip.to_dict(),own_chat_template=(W/'chat_template.jinja').read_text(),own_chat_example=proc.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text='Question\n\nAnswer:')])],tokenize=False,add_generation_prompt=True,enable_thinking=False),actual_processor_source=inspect.getfile(type(proc)),actual_processor_source_SHA256=sha(inspect.getfile(type(proc))),actual_image_processor_source=inspect.getfile(type(ip)),actual_image_processor_source_SHA256=sha(inspect.getfile(type(ip))),actual_image_processor_source_text=Path(inspect.getfile(type(ip))).read_text(),no_learned_CPU_or_NPU=True)
print(json.dumps(result,ensure_ascii=False))

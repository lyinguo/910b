import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import workbook_rows
files={}
for n in ['configs/MllamaForConditionalGeneration/unsloth--Llama-3.2-11B-Vision.yaml','universal_eval/model/adapters/mllama_base_completion.py','scripts/prepare_meta_mllama_own_original_HF_BF16_2048_frontend200_meta_sources_20261120.py']:
 p=R/n
 if p.is_file():
  b=p.read_bytes();files[n]={'SHA256':hashlib.sha256(b).hexdigest(),'complete_text':b.decode()}
rows,_,_=workbook_rows(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'));row=[r for r in rows if r['model_id']=='unsloth/Llama-3.2-11B-Vision']
print(json.dumps({'files':files,'current_C_unique_binding':row,'no_model_or_frontend_NN':True},ensure_ascii=False))
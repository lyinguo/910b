import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');F=R/'workspace/reports/huihui-ai_Huihui-GLM-4.6V-Flash-abliterated/image-text-to-text/20261119_152641';r=json.loads((F/'eval_report.json').read_bytes());out=dict(metric=r['metrics'],detail_schema=list(r['detailed_results'][0]),first_detail=r['detailed_results'][0],metric_adapter=r['pipeline_config']['metrics'],algorithm=r['pipeline_config']['algorithm'],source={})
cfg=r['pipeline_config'];rel=cfg['algorithm']['adapter'].split(':')[0]
for n in [rel,'universal_eval/model/adapters/huihui_glm46flash_own_original_BF16_2048_20261118_frontend.py']:
 p=R/n;out['source'][n]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text())
print(json.dumps(out,ensure_ascii=False))

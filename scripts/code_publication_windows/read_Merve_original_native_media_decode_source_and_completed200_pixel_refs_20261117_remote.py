import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');home=R/'workspace/diagnostics/model_runs/merve_native_vllm_slgljx7_';idx=json.loads((home/'full_numeric_group_index.json').read_bytes());sources={}
for n in ['multimodal/image.py','multimodal/media/image.py','multimodal/utils.py','multimodal/media/base.py']:
 p=Path('/vllm-workspace/vllm/vllm')/n
 if p.is_file():sources[str(p)]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text())
groups=[]
for rec in idx['groups']:
 g=json.loads(Path(rec['path']).read_bytes());groups.append(dict(group_index=g['group_index'],archive=g['archive'],archive_SHA256=g['archive_SHA256'],vision=g['vision'],input_related_actual_worker_sources={n:h for n,h in g['all_actual_worker_imported_sources_and_mapped_shared_libraries_SHA256'].items() if 'multimodal' in n and any(t in n for t in ['image','media','processing'])}))
print(json.dumps(dict(original_native_sources=sources,completed200_groups=groups,no_model_or_verifier_started=True,no_original_files_modified=True),ensure_ascii=False))

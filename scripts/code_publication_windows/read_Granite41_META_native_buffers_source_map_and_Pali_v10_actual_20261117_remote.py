import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,inspect,hashlib
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')];B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
import torch
from universal_eval.model.granite41_own_original_source_20261117.configuration import Granite4VisionConfig
from universal_eval.model.granite41_own_original_source_20261117.modeling import Granite4VisionForConditionalGeneration
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Granite4VisionForConditionalGeneration/ibm-granite/granite-vision-4.1-4b');c=Granite4VisionConfig.from_pretrained(W,local_files_only=True)
with torch.device('meta'):model=Granite4VisionForConditionalGeneration(c)
buffers=[];sources={}
for name,module in model.named_modules():
 own={k:dict(shape=list(t.shape),dtype=str(t.dtype),persistent=k not in module._non_persistent_buffers_set) for k,t in module.named_buffers(recurse=False)}
 if own:
  buffers.append(dict(module=name,cls=type(module).__name__,buffers=own,has_config=hasattr(module,'config')))
  p=Path(inspect.getfile(type(module)));sources[str(p)]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),class_source=inspect.getsource(type(module)))
H=B/'Granite41_own_original_HF_BF16_4096_frontend200_source_META_v3_exact_own_wrapper_layout_20261117';mapping=json.loads((H/'own_source_to_native_META_name_map.json').read_bytes())
J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117';PALI={}
for p in [J/'state.json',J/'completion.json',J/'fresh3/verification_stderr.log']:
 if p.exists():PALI[str(p)]=json.loads(p.read_bytes()) if p.suffix=='.json' else p.read_text()
console=J/'fresh3/console.log'
if console.exists():PALI['console_tail']=console.read_text(errors='replace').splitlines()[-10:]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),own_meta_buffers=buffers,sources=sources,mapping_first3=list(mapping.items())[:3],mapping_count=len(mapping),config_attention_implementations=dict(root=c._attn_implementation,text=c.text_config._attn_implementation,vision=c.vision_config._attn_implementation),Pali=PALI,procs=project_processes(),FLOCK=runner_lock_owners()),ensure_ascii=False))

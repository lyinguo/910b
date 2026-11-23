"""Read the original compatibility layer and installed native interfaces only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,inspect,json
R=Path('/root/projects/comperror')
p=R/'universal_eval/model/adapters/high_nine_kanana_reference.py';b=p.read_bytes()
import transformers.models.qwen2_vl.modeling_qwen2_vl as native
names=['PatchEmbed','Qwen2VisionPatchEmbed','Qwen2VLVisionPatchEmbed','VisionRotaryEmbedding',
       'Qwen2VisionRotaryEmbedding','Qwen2VLVisionRotaryEmbedding','Qwen2VLPreTrainedModel',
       'Qwen2VisionTransformerPretrainedModel','Qwen2VLVisionModel','Qwen2VLVisionBlock']
objects={}
for name in names:
    obj=getattr(native,name,None)
    objects[name]=dict(exists=obj is not None)
    if obj is not None:
        objects[name]['signature']=str(inspect.signature(obj))
        if hasattr(obj,'forward'):objects[name]['forward_signature']=str(inspect.signature(obj.forward))
np=Path(native.__file__);nb=np.read_bytes()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),original_reference=dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),complete_source_text=b.decode('utf8')),
    actual_native_Qwen2=dict(path=str(np),SHA256=hashlib.sha256(nb).hexdigest(),interfaces=objects),
    no_model_construct_or_weight_or_learned_forward=True,no_mutation=True),ensure_ascii=False))

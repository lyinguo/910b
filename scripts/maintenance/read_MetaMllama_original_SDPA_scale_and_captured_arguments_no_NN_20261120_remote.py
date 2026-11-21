import sys
sys.dont_write_bytecode=True
from pathlib import Path
import inspect,json,math
from transformers.models.mllama.modeling_mllama import MllamaTextSelfAttention,MllamaTextCrossAttention,MllamaVisionAttention
R=Path('/root/projects/comperror');g=R/'workspace/diagnostics/model_runs/MetaMllama_own_original_SDPA2048_20261120_121105_669810_1616339/groups/0001/batch_execution.json';x=json.loads(g.read_bytes());calls=x['steps'][0]['actual_native_SDPA_calls'];out=dict(actual_NEW_model_forwards=0,source_scale_expressions={c.__name__:[l.strip() for l in inspect.getsource(c.__init__).splitlines() if 'scaling' in l or 'head_dim' in l] for c in [MllamaTextSelfAttention,MllamaTextCrossAttention,MllamaVisionAttention]},captured_distinct_arguments={})
for r in calls:
 k=r['kind'];d=r['query_shape'][-1];out['captured_distinct_arguments'][k]=dict(actual_scale=r['scale'],head_dim=d,original_pow=d**-.5,reciprocal_sqrt=1./math.sqrt(d),actual_equals_original_pow=(r['scale']==d**-.5),actual_equals_reciprocal_sqrt=(r['scale']==1./math.sqrt(d)))
print(json.dumps(out,ensure_ascii=False))

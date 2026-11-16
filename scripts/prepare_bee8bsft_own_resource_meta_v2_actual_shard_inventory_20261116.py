import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,importlib,shutil,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Bee8BSFT_own_resource_frontend_meta_v2_actual_shard_inventory_no_NPU_20261116';H.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/BeeForConditionalGeneration/Open-Bee/Bee-8B-SFT');S=R/'universal_eval/model/bee8bsft_own_original_20261116';assert S.is_dir() and (S/'__init__.py').is_file()
source={}
for n in ['configuration_bee.py','modeling_bee.py','image_processing_bee.py','processing_bee.py']:
 p=W/n;q=S/n;ast.parse(p.read_bytes());assert sha(p)==sha(q);source[str(q.relative_to(R))]=sha(q)
source[str((S/'__init__.py').relative_to(R))]=sha(S/'__init__.py')
from datetime import datetime,timezone
import torch,numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from universal_eval.model.bee8bsft_own_original_20261116.configuration_bee import BeeConfig
from universal_eval.model.bee8bsft_own_original_20261116.modeling_bee import BeeForConditionalGeneration
from universal_eval.model.bee8bsft_own_original_20261116.image_processing_bee import BeeImageProcessor
from universal_eval.model.bee8bsft_own_original_20261116.processing_bee import BeeProcessor
torch.set_num_threads(4)
cfg=BeeConfig(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='eager'
with torch.device('meta'):model=BeeForConditionalGeneration(cfg)
def mapped(k):
 for a,b in [('language_model.model.','model.language_model.'),('language_model.lm_head.','lm_head.'),('vision_tower.','model.vision_tower.'),('multi_modal_projector.','model.multi_modal_projector.'),('image_newline','model.image_newline')]:
  if k.startswith(a):return b+k[len(a):]
 return k
index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];params=dict(model.named_parameters(remove_duplicate=False));expected={};files={};actual_index={};misplaced=[]
for n in sorted(set(index.values())):
 files[n]=dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n))
 with safe_open(str(W/n),framework='pt',device='cpu') as f:
  misplaced.extend(dict(key=k,index_shard=index.get(k),actual_shard=n) for k in f.keys() if index.get(k)!=n)
  for k in f.keys():
   assert k not in actual_index,('Duplicate own key',k);actual_index[k]=n
   t=f.get_tensor(k);key=mapped(k);assert key in params and t.shape==params[key].shape and t.dtype==torch.bfloat16,(k,key,t.shape)
   h=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();expected[key]=dict(source_key=k,source_file=n,runtime_name=key,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=h,cast_SHA256=h)
assert set(actual_index)==set(index),('Own index keys differ from actual union',set(actual_index)^set(index))
write(H/'original_index_vs_actual_shard_inventory.json',dict(original_index_SHA256=sha(W/'model.safetensors.index.json'),actual_index=actual_index,misplaced=misplaced,own_original_files_unchanged=True,all_keys_unique_and_global_index_key_set_exact=True,load_actual_own_header_locations_without_changing_checkpoint_or_original_index=True))
assert set(expected)==set(params),(set(params)-set(expected),set(expected)-set(params));write(H/'own_source_manifest.json',expected)
tok=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=False);ip=BeeImageProcessor.from_pretrained(W,local_files_only=True);pc=json.loads((W/'processor_config.json').read_bytes());pc.pop('auto_map',None);pc.pop('processor_class',None);processor=BeeProcessor(ip,tok,chat_template=(W/'chat_template.jinja').read_text(),**pc)
assert tok.convert_tokens_to_ids('<image>')==151669 and cfg.text_config.vocab_size==152000 and cfg.text_config.num_hidden_layers==36
assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.name not in files and p.suffix in ('.json','.py','.txt','.jinja')}
write(H/'resource_meta_ready.json',dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,model_id='Open-Bee/Bee-8B-SFT',weight_path=str(W),own_source_count=len(expected),own_weight_files=files,own_assets_SHA256=assets,own_source_manifest=str(H/'own_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_source_manifest.json'),implementation_source_SHA256=source,own_source_raw_bytes_exact=True,meta_parameter_shapes_exact=True,own_processor_import=True,actual_shard_inventory=str(H/'original_index_vs_actual_shard_inventory.json'),actual_shard_inventory_SHA256=sha(H/'original_index_vs_actual_shard_inventory.json'),original_index_shard_assignment_mismatch_count=len(misplaced),original_index_and_checkpoint_unmodified=True,no_NPU_inference=True,own_original_adapter=str(R/'universal_eval/model/adapters/bee_adapter.py'),own_original_adapter_SHA256=sha(R/'universal_eval/model/adapters/bee_adapter.py'),initial_eos_contract='Original adapter runs first token plus loop and only tests EOS after loop token; not silently changed',original_enable_thinking_false_is_existing_adapter_default_not_new_tuning=True))
print(json.dumps(dict(passed=True,home=str(H),source_count=len(expected),own_files=files,own_source_exact=True,no_NPU=True),ensure_ascii=False))

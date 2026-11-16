"""Own VideoChat Flash image graph, saved native Qwen2 path, complete source preflight."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,contextlib,io,traceback,types
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'VCFlash7B_own_image_graph_full_source_preflight_20261116';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/VideoChatFlashQwenForCausalLM/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448');S=R/'universal_eval/model/vcflash7b_own_original_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import torch
 from safetensors import safe_open
 H.mkdir(exist_ok=False);S.mkdir(exist_ok=False);(S/'__init__.py').write_text('',encoding='utf-8');torch.set_num_threads(4)
 raw=(W/'vision_tower_builder.py').read_text();start=raw.index('try:\n    from flash_attn');end=raw.index('\n\n\nclass DropPath',start)
 replacement="""# The saved TransformersVLM path selects the own origin attention backend.
# No fake flash_attn/timm modules. This source builds zero drop-path vision;
# all actual Block.forward math is preserved verbatim and checked below.
use_flash_attn = False
from torch.nn.modules.utils import _pair as to_2tuple
from torch.nn.init import trunc_normal_
"""
 patched=raw[:start]+replacement+raw[end:];(S/'vision_tower_builder.py').write_text(patched,encoding='utf-8')
 def methods(text):
  t=ast.parse(text);return {c.name+'.'+f.name:ast.dump(f,include_attributes=False) for c in t.body if isinstance(c,ast.ClassDef) for f in c.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
 assert methods(raw)==methods(patched)
 (S/'mm_projector_builder.py').write_bytes((W/'mm_projector_builder.py').read_bytes());(S/'constants.py').write_bytes((W/'constants.py').read_bytes())
 original=(W/'modeling_videochat_flash.py').read_text();tree=ast.parse(original);c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LlavaMetaForCausalLM');chosen=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ['get_vision_tower','encode_video_image','prepare_inputs_labels_for_multimodal']];assert len(chosen)==3
 mix="import torch,re\nfrom .constants import IGNORE_INDEX,IMAGE_TOKEN_INDEX\nclass OwnImageMixin:\n"+'\n'.join('    '+line if line else '' for n in chosen for line in ast.get_source_segment(original,n).splitlines())+'\n'
 ast.parse(mix);m2=next(n for n in ast.parse(mix).body if isinstance(n,ast.ClassDef));assert [ast.dump(n,include_attributes=False) for n in chosen]==[ast.dump(n,include_attributes=False) for n in m2.body]
 (S/'image_mixin.py').write_text(mix,encoding='utf-8')
 graph='''"""Own image-only source functions plus the saved standard Qwen2 decoder."""
import torch,types
from transformers import Qwen2Config,Qwen2ForCausalLM
from .vision_tower_builder import build_vision_tower
from .mm_projector_builder import build_vision_projector
from .image_mixin import OwnImageMixin
class OwnVideoChatFlashImage(OwnImageMixin,Qwen2ForCausalLM):
 def __init__(self,cfg):
  with torch.device('meta'):super().__init__(cfg)
  # CPU construction only: no learned CPU forward or grade. The original
  # vision constructor uses linspace.item and CPU analytic sinusoid tables.
  # Every learned parameter is subsequently replaced from own source on NPU.
  with torch.device('cpu'):
   self.model.vision_tower=build_vision_tower(cfg)
   self.model.mm_projector=build_vision_projector(cfg,vision_cfg=self.model.vision_tower.config)
  def getvt(m):return m.vision_tower
  self.model.get_vision_tower=types.MethodType(getvt,self.model)
 def get_model(self):return self.model
'''
 (S/'modeling_image_graph.py').write_text(graph,encoding='utf-8')
 from transformers import Qwen2Config
 from universal_eval.model.vcflash7b_own_original_20261116.modeling_image_graph import OwnVideoChatFlashImage
 cfg=Qwen2Config(**json.loads((W/'config.json').read_bytes()));cfg._attn_implementation='eager';assert cfg.num_hidden_layers==28 and cfg.hidden_size==3584 and cfg.num_key_value_heads==4 and not cfg.mm_llm_compress
 with contextlib.redirect_stdout(io.StringIO()):model=OwnVideoChatFlashImage(cfg)
 assert model.model.vision_tower.config.image_size==448 and model.model.mm_projector.num_image_patches_per_side==8
 assert len(model.model.vision_tower.vision_tower.encoder.blocks)==23 and all(isinstance(l.drop_path,torch.nn.Identity) for l in model.model.vision_tower.vision_tower.encoder.blocks)
 params=dict(model.named_parameters(remove_duplicate=False));index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];expected={};files={};actual={};mismatches=[]
 for p in sorted(W.glob('model-*.safetensors')):
  files[p.name]=dict(bytes=p.stat().st_size,SHA256=sha(p))
  with safe_open(str(p),framework='pt',device='cpu') as f:
   for n in f.keys():
    assert n not in actual;actual[n]=p.name;t=f.get_tensor(n);assert n in params and t.shape==params[n].shape and t.dtype in (torch.float16,torch.bfloat16,torch.float32),(n,t.shape,t.dtype)
    expected[n]=dict(runtime_name=n,source_key=n,source_file=p.name,shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),cast_SHA256=hashlib.sha256(t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    if index.get(n)!=p.name:mismatches.append(n)
 assert set(actual)==set(index)==set(params),(set(params)-set(actual),set(actual)-set(params));put('own_source_manifest.json',expected)
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.name not in files and p.suffix in ('.json','.py','.txt') and p.name!='trainer_state.json'}
 ready=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id='OpenGVLab/VideoChat-Flash-Qwen2-7B_res448',weight_path=str(W),own_weight_files=files,own_assets_SHA256=assets,own_source_manifest=str(H/'own_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_source_manifest.json'),source_tensor_count=len(expected),own_all_source_keys_shapes_bijective=True,own_vision23_runtime_blocks_from_return_idx_minus2=True,own_image_methods_AST_exact=True,vision_all_math_method_AST_exact=True,only_unused_optional_acceleration_imports_removed=True,own_original_raw_sources_preserved=True,zero_drop_path_all_Identity_verified=True,no_NPU_or_learned_CPU_forward=True,original_standard_Qwen2_decoder_source='universal_eval/model/adapters/transformers_vlm.py:_vcf_manual_generate',original_standard_Qwen2_decoder_source_SHA256=sha(R/'universal_eval/model/adapters/transformers_vlm.py'),source_records={str(p.relative_to(R)):sha(p) for p in S.glob('*.py')},own_actual_index_shard_location_mismatches=mismatches)
 put('readiness.json',ready);put('actual_completion.json',dict(UTC=ready['UTC'],actual_returncode=0));print(json.dumps(ready,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NPU_inference=True))
  raise

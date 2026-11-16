"""Own image-only source functions plus the saved standard Qwen2 decoder."""
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

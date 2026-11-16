"""Construct only the actual understanding path; preserve own original forward/preprocessing methods."""
import torch
from transformers import AutoModel,AutoModelForCausalLM,Qwen2TokenizerFast,PreTrainedModel
from .configuration_ovis_u1 import OvisU1Config
from .modeling_ovis_u1 import OvisU1,VisualEmbedding
class OwnOvisUnderstanding(OvisU1):
 def __init__(self,config,weight_path):
  PreTrainedModel.__init__(self,config)
  kw={} if not config.llm_attn_implementation else {'attn_implementation':config.llm_attn_implementation}
  self.llm=AutoModelForCausalLM.from_config(config.llm_config,**kw)
  self.text_tokenizer=Qwen2TokenizerFast.from_pretrained(str(weight_path),local_files_only=True)
  self.visual_tokenizer=AutoModel.from_config(config.visual_tokenizer_config,image_processor_name_or_path=str(weight_path))
  self.vte=VisualEmbedding(config.visual_tokenizer_config.vocab_size,config.hidden_size,device=self.visual_tokenizer.device,dtype=self.visual_tokenizer.dtype)
  if not config.disable_tie_weight:self.llm.tie_weights()
  assert config.hidden_size==self.llm.config.hidden_size==2048
  # No fake/empty visual_generator module: generator is outside the selected image-understanding execution path.

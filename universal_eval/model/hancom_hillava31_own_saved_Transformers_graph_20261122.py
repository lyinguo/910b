"""Only the saved own legacy configuration/name mapping; no learned forward replacement."""
import torch
from transformers import LlamaConfig,CLIPVisionConfig,LlavaConfig,LlavaForConditionalGeneration
def own_config(raw,expanded=True):
 fields=['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','rms_norm_eps','rope_theta','max_position_embeddings','tie_word_embeddings','attention_bias','hidden_act','initializer_range','pretraining_tp']
 values={k:raw[k] for k in fields if k in raw};values['vocab_size']=128257 if expanded else 128256
 tc=LlamaConfig(**values);vc=CLIPVisionConfig(hidden_size=1024,num_hidden_layers=24,num_attention_heads=16,intermediate_size=4096,patch_size=14,image_size=336,num_channels=3,projection_dim=768,layer_norm_eps=1e-5,hidden_act='quick_gelu')
 return LlavaConfig(text_config=tc,vision_config=vc,image_token_index=128256 if expanded else 32000,vision_feature_layer=-2,vision_feature_select_strategy='default',image_aspect_ratio='pad',mm_hidden_size=1024)
def build_own_meta(raw):
 dtype=torch.get_default_dtype()
 try:
  torch.set_default_dtype(torch.float16)
  with torch.device('meta'):model=LlavaForConditionalGeneration(own_config(raw))
 finally:torch.set_default_dtype(dtype)
 return model

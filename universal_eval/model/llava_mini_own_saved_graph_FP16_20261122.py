"""Rebuild precisely the already saved LLaVA-Mini graph; no checkpoint substitution."""
import sys
sys.dont_write_bytecode=True
import torch
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM,CLIPVisionConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from universal_eval.model.adapters import high_nine_llava_reference as original

def original_text_kwargs(raw):
    return dict(
        vocab_size=int(raw.get('vocab_size',128256)),
        hidden_size=int(raw['hidden_size']),
        intermediate_size=int(raw['intermediate_size']),
        num_hidden_layers=int(raw['num_hidden_layers']),
        num_attention_heads=int(raw['num_attention_heads']),
        num_key_value_heads=int(raw.get('num_key_value_heads',raw['num_attention_heads'])),
        max_position_embeddings=int(raw.get('max_position_embeddings',131072)),
        rms_norm_eps=float(raw.get('rms_norm_eps',1e-5)),
        rope_theta=float(raw.get('rope_theta',10000.0)),
        hidden_act=raw.get('hidden_act','silu'),
        attention_bias=bool(raw.get('attention_bias',False)),
        bos_token_id=raw.get('bos_token_id',128000),
        eos_token_id=raw.get('eos_token_id',128001),
        pad_token_id=raw.get('pad_token_id',None),
        tie_word_embeddings=bool(raw.get('tie_word_embeddings',False)),
        torch_dtype=torch.float16,
    )

def build_own_saved_graph(raw,meta=False):
    assert raw['model_type']=='llava_mini_llama'
    assert raw['compressor_size']==1 and raw['prefusion_layer_num']==4 and raw['resolution_ratio']==1
    assert raw['hidden_size']==4096 and raw['num_hidden_layers']==32 and raw['vocab_size']==128256
    # The saved adapter explicitly omits rope_scaling. Preserve this original
    # branch, even though the checkpoint's raw configuration also contains it.
    text_cfg=LlamaConfig(**original_text_kwargs(raw))
    with torch.device('meta' if meta else 'cpu'):
        model=LlamaForCausalLM(text_cfg)
        clip_cfg=CLIPVisionConfig(**original.CLIP_L_336)
        model.model.vision_tower=original._LlavaMiniVisionTower(clip_cfg,int(raw.get('mm_vision_select_layer',-2)),str(raw.get('mm_vision_select_feature','patch')))
        mm_hidden_size=int(raw.get('mm_hidden_size',original.CLIP_L_336['hidden_size']))
        model.model.mm_projector=nn.Sequential(nn.Linear(mm_hidden_size,text_cfg.hidden_size),nn.GELU(),nn.Linear(text_cfg.hidden_size,text_cfg.hidden_size))
        model.model.compressor=original._CompressorResampler(grid_size=int(raw.get('compressor_size',1)),embed_dim=mm_hidden_size,num_heads=8)
        model.model.prefusion_layers=nn.ModuleList([LlamaDecoderLayer(text_cfg,layer_idx=i) for i in range(int(raw.get('prefusion_layer_num',4)))])
    if meta:model=model.to('meta')
    return model

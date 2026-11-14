"""Reviewed original sampling protocol and CPU-derived FP32 M-RoPE frequencies."""
import hashlib,json,pathlib
import torch
import yaml
from universal_eval.model.adapters.qwen25vl_native_910b_adapter import Qwen25VLNative910BAdapter

class Qwen25VLReviewed910BAdapter(Qwen25VLNative910BAdapter):
    def __init__(self,original_config=None,original_config_sha256=None,original_adapter_sha256=None,**kwargs):
        self.original_config=original_config;self.original_config_sha256=original_config_sha256
        self.original_adapter_sha256=original_adapter_sha256
        super().__init__(**kwargs)

    def load(self,model_path,**kwargs):
        original_path=pathlib.Path(self.original_config)
        if hashlib.sha256(original_path.read_bytes()).hexdigest()!=self.original_config_sha256:raise RuntimeError('Original evaluation protocol changed')
        original=yaml.safe_load(original_path.read_text());kind=original['model']['adapter']
        source,cls=kind.rsplit(':',1);source=pathlib.Path(source)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=self.original_adapter_sha256:raise RuntimeError('Original adapter protocol source changed')
        if cls not in ('VLLMModel','VLLMMCChoiceModel','TransformersVLM'):raise RuntimeError('Unreviewed original protocol')
        original_kwargs=original['model'].get('kwargs',{})
        if any(original_kwargs.get(k) for k in ('stop','prompt_prefix','chat_template_kwargs')):raise RuntimeError('Additional original prompt/stop protocol needs separate review')
        budget=original_kwargs.get('max_new_tokens',original_kwargs.get('max_tokens',2048))
        if int(budget)!=self.max_new_tokens:raise RuntimeError('Original generation budget changed')
        expected='constrained_choice' if cls=='VLLMMCChoiceModel' else 'greedy'
        if expected!=self.generation_mode:raise RuntimeError('Original generation mode changed')
        super().load(model_path,**kwargs)
        if original['model']['name']!=self.model_name or original['task_type']!='image-text-to-text':raise RuntimeError('Original model/task differs')
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
        # Compute original analytic FP32 constants on CPU, then transfer without NPU pow/div rounding differences.
        self.model.model.language_model.rotary_emb=Qwen2_5_VLRotaryEmbedding(self.model.config.text_config).to(self.device)
        if cls in ('VLLMModel','VLLMMCChoiceModel'):
            # Original SamplingParams(temperature=0,top_p=1,max_tokens=...) uses repetition_penalty=1.
            # Do not accidentally inherit sampling-only repetition_penalty=1.05 from the checkpoint file.
            self.generation_config.repetition_penalty=1.0
            self.generation_config.no_repeat_ngram_size=0;self.generation_config.num_beams=1
            self.generation_config.temperature=1.0;self.generation_config.top_p=1.0;self.generation_config.top_k=50
        self.generation_config.do_sample=False
        self.lineage['rotary_frequencies']['text_inv_freq']=self.model.model.language_model.rotary_emb.inv_freq.detach().cpu().tolist()
        self.lineage['original_evaluation_protocol']={'config':str(original_path),'config_sha256':self.original_config_sha256,
            'adapter':kind,'adapter_source_sha256':self.original_adapter_sha256,'original_max_tokens':int(budget),
            'effective_token_limit':1 if expected=='constrained_choice' else int(budget),
            'effective_repetition_penalty':float(self.generation_config.repetition_penalty),
            'note':'Original VLLM greedy SamplingParams defaults preserved, including repetition_penalty=1; original TransformersVLM retains own generation_config with do_sample=False. Only original VLLMMCChoiceModel uses a closed single-letter grammar.'}
        self.lineage['effective_generation_config']=self.generation_config.to_dict()
        self.lineage['rotary_frequencies']['derivation']='Original CPU FP32 analytic frequencies transferred exactly to NPU, not BF16-rounded'
        snapshot=pathlib.Path(self.lineage['architecture_source_snapshot'])
        (snapshot/'original_evaluation.yaml').write_bytes(original_path.read_bytes())
        (snapshot/'original_adapter.py').write_bytes(source.read_bytes())

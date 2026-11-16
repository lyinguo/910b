"""Own original Gemma3n native text/audio and weight-free original MobileNet math.

Only the original fixed eval vision interface is implemented privately. The
operator graph has no checkpoint values: every learned tensor comes from this
machine's original safetensors and is verified before any learned inference.
No timm package availability is changed and no fake module is installed.
"""
from pathlib import Path
from dataclasses import dataclass
import json,hashlib,inspect
import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig,AutoModel,GenerationConfig
from transformers.utils import ModelOutput
import transformers.models.gemma3n.modeling_gemma3n as native

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())

@dataclass
class OwnOriginalVisionOutput(ModelOutput):
    last_hidden_state:torch.Tensor|None=None
    pooler_output:torch.Tensor|None=None
    hidden_states:tuple[torch.Tensor,...]|None=None

class OwnOriginalParameterTree(nn.Module):
    def __init__(self,state,parameter_names):
        super().__init__()
        for name,rec in state.items():
            parts=name.split('.');owner=self
            for part in parts[:-1]:
                if part not in owner._modules:owner.add_module(part,nn.Module())
                owner=owner._modules[part]
            dtype=getattr(torch,rec['dtype'].removeprefix('torch.'));value=torch.empty(rec['shape'],dtype=dtype)
            if name in parameter_names:owner.register_parameter(parts[-1],nn.Parameter(value,requires_grad=False))
            else:owner.register_buffer(parts[-1],value)
    def value(self,name):
        owner=self;parts=name.split('.')
        for part in parts[:-1]:owner=getattr(owner,part)
        return getattr(owner,parts[-1])

class OwnOriginalParameterizedVision(nn.Module):
    def __init__(self,config,graph_source,graph_manifest,actual_AA_metadata):
        super().__init__();self.config=config;self.m=js(graph_manifest)
        assert self.m['passed'] and self.m['checkpoint_values_exported'] is False and self.m['checkpoint_tensor_constants_in_graph']==0
        assert config.architecture=='mobilenetv5_300m_enc' and config.do_pooling is False and not config.model_args
        assert sha(graph_source)==self.m['own_original_graph_SHA256']
        self.source_path=str(graph_source);self.source_SHA256=sha(graph_source);self.manifest_path=str(graph_manifest)
        aliases=js(actual_AA_metadata)['actual_parameter_aliases'];prefix='model.vision_tower.timm_model.'
        named={name.removeprefix(prefix) for group in aliases for name in group if name.startswith(prefix)}
        assert named.issubset(self.m['own_state'])
        self.timm_model=OwnOriginalParameterTree(self.m['own_state'],named)
        # Preserve any actual native aliases, including their storage identity.
        for group in aliases:
            names=[name.removeprefix(prefix) for name in group if name.startswith(prefix)]
            if len(names)<2:continue
            value=self.timm_model.value(names[0])
            for name in names[1:]:
                owner=self.timm_model;parts=name.split('.')
                for part in parts[:-1]:owner=getattr(owner,part)
                setattr(owner,parts[-1],value)
        torch._C._set_graph_executor_optimize(False)
        ir=Path(graph_source).read_text();self.fn=torch._C._create_function_from_graph('own_Gemma3n_original_MobileNetV5_weight_free_math',torch._C.parse_ir(ir))
        self.literal_templates=[]
        for rec in self.m.get('original_literal_scalar_tensor_arguments',[]):
            assert rec['scope'].startswith('Original source literal')
            dtype=getattr(torch,rec['dtype'].removeprefix('torch.'));value=torch.tensor(rec['original_value'],dtype=dtype,device='cpu').reshape(rec['shape']);assert tsha(value)==rec['SHA256']
            self.literal_templates.append(value)
        self.observer=None
    def forward(self,pixel_values,output_attentions=None,output_hidden_states=None,return_dict=None,do_pooling=None,use_cache=None,**kwargs):
        assert not self.training and tuple(pixel_values.shape)==(1,3,768,768) and pixel_values.dtype==torch.float16
        assert not output_attentions and not output_hidden_states and do_pooling in [None,False] and not kwargs
        device=next(self.timm_model.parameters()).device;assert pixel_values.device==device
        params=[self.timm_model.value(rec['own_state_key']) for rec in self.m['lowered_parameter_inputs']]
        assert all(t.device==device for t in params)
        literals=[value if rec['original_device']=='cpu' else value.to(device) for value,rec in zip(self.literal_templates,self.m.get('original_literal_scalar_tensor_arguments',[]))]
        result=self.fn(pixel_values,*params,*literals,*([device] if self.m['graph_device_argument'] else []))
        assert len(result)==1+len(self.m['stage_outputs'])
        assert all(t.device==device and t.dtype==torch.float16 for t in result)
        assert tuple(result[0].shape)==(1,2048,16,16)
        if self.observer is not None:self.observer(result,self.m['stage_outputs'])
        output=OwnOriginalVisionOutput(last_hidden_state=result[0],pooler_output=None,hidden_states=None)
        return output if return_dict is not False else (result[0],)

class OwnOriginalGemma3nModel(native.Gemma3nModel):
    def __init__(self,config,graph_source,graph_manifest,actual_AA_metadata):
        native.Gemma3nPreTrainedModel.__init__(self,config)
        config.vision_config._attn_implementation='eager'
        self.vision_tower=OwnOriginalParameterizedVision(config.vision_config,graph_source,graph_manifest,actual_AA_metadata)
        self.vocab_size=config.text_config.vocab_size
        language_model=AutoModel.from_config(config=config.text_config)
        self.language_model=language_model
        self.vocab_size_per_layer_input=config.text_config.vocab_size_per_layer_input
        self.audio_tower=AutoModel.from_config(config.audio_config)
        self.embed_vision=native.Gemma3nMultimodalEmbedder(config.vision_config,config.text_config)
        self.embed_audio=native.Gemma3nMultimodalEmbedder(config.audio_config,config.text_config)
        self.post_init()

class OwnOriginalGemma3nForConditionalGeneration(native.Gemma3nForConditionalGeneration):
    def __init__(self,config,graph_source,graph_manifest,actual_AA_metadata):
        native.Gemma3nPreTrainedModel.__init__(self,config)
        self.model=OwnOriginalGemma3nModel(config,graph_source,graph_manifest,actual_AA_metadata)
        self.lm_head=nn.Linear(config.text_config.hidden_size,config.text_config.vocab_size,bias=False)
        self.post_init()

def jsonable(v):
    if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
    if isinstance(v,set):return sorted([jsonable(x) for x in v],key=repr)
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    if v is None or isinstance(v,(str,int,float,bool)):return v
    raise TypeError('Unexpected diagnostic JSON type: '+type(v).__name__)

"""Private real RegStage operators, bound to this checkpoint's own parameters.

The two graphs were obtained from the installed original timm implementation.
No timm module is installed or substituted and no learned values are embedded.
This interface covers exactly the saved evaluation topology, not training.
"""
from pathlib import Path
import hashlib,json
import torch
from torch import nn
from .original_timm_layers_20261123 import LayerNorm,LayerNorm2d,resample_abs_pos_embed

R=Path(__file__).resolve().parents[3]
MANIFEST=R/'configs/protocols/kanana_own_original_RegStage_BF16_parameterized_graphs_v2_graphtext_20261123.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class RegStage(nn.Module):
    def __init__(self,depth,in_chs,out_chs,stride,dilation,act_layer,norm_layer,**kwargs):
        super().__init__()
        assert depth==2 and in_chs in [1280,4096] and out_chs==1024
        assert stride==dilation==1 and act_layer is nn.SiLU and norm_layer is LayerNorm2d and not kwargs
        self._manifest=json.loads(MANIFEST.read_bytes())
        selected,=[s for s in self._manifest['stages'] if s['input_channels']==in_chs]
        self._selected=selected;self._in_channels=in_chs
        path=R/selected['graph_relative_path'];assert sha(path)==selected['graph_SHA256']
        graph=torch._C.parse_ir(path.read_text(encoding='utf8'))
        for node in graph.nodes():
            assert node.kind() not in ['prim::GetAttr','prim::CallMethod']
            if node.kind()=='prim::Constant' and node.hasAttribute('value'):assert node.kindOf('value')!='t'
        assert len(list(graph.inputs()))==1+len(selected['parameter_inputs'])
        self._function=torch._C._create_function_from_graph('own_Kanana_real_RegStage_'+str(in_chs),graph)
        for rec in selected['parameter_inputs']:
            owner=self;parts=rec['relative_key'].split('.')
            for part in parts[:-1]:
                if part not in owner._modules:owner.add_module(part,nn.Module())
                owner=owner._modules[part]
            owner.register_parameter(parts[-1],nn.Parameter(torch.empty(rec['shape'],dtype=torch.bfloat16),requires_grad=False))
        assert set(self.state_dict())==set(selected['original_state_keys'])

    def forward(self,x):
        assert not self.training and x.ndim==4 and x.shape[1]==self._in_channels and x.dtype==torch.bfloat16
        values=[self.get_parameter(p['relative_key']) for p in self._selected['parameter_inputs']]
        assert all(t.device==x.device and t.dtype==torch.bfloat16 for t in values)
        return self._function(x,*values)

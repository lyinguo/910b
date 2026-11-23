"""Diagnose graph interface flags using tiny synthetic CPU operators only."""
import sys
sys.dont_write_bytecode=True
from datetime import datetime,timezone
import json,torch
from torch import nn
torch.set_num_threads(2);torch.manual_seed(1131)
class Tiny(nn.Module):
    def __init__(self):super().__init__();self.c=nn.Conv2d(2,2,1);self.a=nn.SiLU()
    def forward(self,x):return self.a(self.c(x))
results=[]
for optimize in [False,True]:
    torch._C._set_graph_executor_optimize(optimize)
    m=nn.Sequential(Tiny(),Tiny()).eval();x=torch.randn((1,2,2,2))
    with torch.inference_mode():t=torch.jit.trace(m,x,check_trace=False)
    g=t.inlined_graph.copy();counts=[]
    for i in range(4):
        counts.append(sum(n.kind()=='prim::CallMethod' for n in g.nodes()));torch._C._jit_pass_inline(g)
    results.append(dict(optimize=optimize,remaining_CallMethod_counts=counts,complete_graph=str(g)))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),torch_version=torch.__version__,results=results,
    synthetic_tiny_CPU_only=True,no_checkpoint_or_real_model_or_GPU_or_NPU=True,no_write_or_install=True),ensure_ascii=False))

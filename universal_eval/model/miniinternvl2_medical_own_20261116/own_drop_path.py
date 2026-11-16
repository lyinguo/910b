"""Real stochastic depth. Original own0.1 remains in eval mode."""
import torch
from torch import nn
class DropPath(nn.Module):
 def __init__(self,drop_prob=0.):super().__init__();self.drop_prob=float(drop_prob)
 def forward(self,x):
  if self.drop_prob==0. or not self.training:return x
  keep=1-self.drop_prob;shape=(x.shape[0],)+(1,)*(x.ndim-1);random=keep+torch.rand(shape,dtype=x.dtype,device=x.device);random.floor_();return x.div(keep)*random

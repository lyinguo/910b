"""Exact right-bucket rank using unchanged BF16 comparison operands.

Ascend aclnnSearchSorted does not accept BF16. Comparing the same BF16 values
against the same original FP32 or BF16 strictly increasing boundaries and counting <= implements
torch.bucketize(right=True) exactly. No model dtype or boundary is changed.
"""
import ast,inspect,textwrap,types
import torch
def exact_right_bucketize(values,boundaries,*,right=False):
 assert right is True and values.dtype==torch.bfloat16 and boundaries.dtype in (torch.bfloat16,torch.float32)
 assert values.ndim>=1 and boundaries.ndim==1 and boundaries.numel()==31
 assert bool(torch.isfinite(values).all()) and bool(torch.isfinite(boundaries).all())
 return torch.sum(values[...,None]>=boundaries,dim=-1,dtype=torch.int64)
def install_own_embedding_buckets(model):
 embeddings=model.model.vision_model.embeddings;old=type(embeddings).forward
 tree=ast.parse(textwrap.dedent(inspect.getsource(old)))
 class ReplaceBucketize(ast.NodeTransformer):
  replacements=0
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='bucketize':
    self.replacements+=1;n.func=ast.Name('exact_right_bucketize',ast.Load())
   return n
 fix=ReplaceBucketize();tree=fix.visit(tree);ast.fix_missing_locations(tree);assert fix.replacements==2
 ns=old.__globals__.copy();ns['exact_right_bucketize']=exact_right_bucketize
 exec(compile('from __future__ import annotations\n'+ast.unparse(tree),'own_GraniteMLX_exact_BF16_position_buckets','exec'),ns)
 embeddings.forward=types.MethodType(ns['forward'],embeddings)
 return dict(original_method_module=old.__module__,original_method_qualname=old.__qualname__,replaced_calls=2,operands_dtype='BF16 unchanged',operation='right=True rank via native BF16 >= and integer count',installed_only_on_own_embeddings_instance=True,installed_library_not_modified=True)

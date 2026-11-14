"""Own original instruction plus explicit invalid-label accounting, unchanged generation."""
from universal_eval.algorithm.smol22_original_HF_MC_2109_20261111 import Smol22OriginalMC
from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
class Smol22OriginalMCGuard(Smol22OriginalMC):
 def run(self,model,samples,**kwargs):
  predictions,references,extra,statistics=super().run(model,samples,**kwargs)
  for i,item in enumerate(extra):
   if isinstance(item.get('raw_output'),str) and explicit_choice(item['raw_output']) is None:predictions[i]=''
  return predictions,references,extra,statistics

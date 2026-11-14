"""Keep invalid raw generations out of the legacy Accuracy regex fallback."""
import logging
from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2,explicit_choice
LOG=logging.getLogger(__name__)
def preserve_invalid_predictions(predictions,extra_info):
 result=list(predictions);invalid=[]
 for i,extra in enumerate(extra_info):
  raw=extra.get('raw_output')
  if isinstance(raw,str) and explicit_choice(raw) is None:
   result[i]='';invalid.append(i)
 return result,invalid
class Ovis22BNativeExplicitMCV2(NativeExplicitMCV2):
 def run(self,model,samples,**kwargs):
  assert all(str(row.get('reference','')).strip() in 'ABCDE' and len(str(row.get('reference','')).strip())==1 for row in samples)
  predictions,references,extra,statistics=super().run(model,samples,**kwargs)
  predictions,invalid=preserve_invalid_predictions(predictions,extra)
  statistics.update(strict_invalid_choice_count=len(invalid),strict_invalid_choice_indices=invalid,
   strict_invalid_policy='empty extracted prediction; complete generated raw retained; included in full denominator')
  LOG.info('Strict Ovis choice verification: valid %s, invalid %s, denominator %s; generated raw unchanged',len(samples)-len(invalid),len(invalid),len(samples))
  return predictions,references,extra,statistics

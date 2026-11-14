"""Restore the original explicit quoted option labels; keep prose articles invalid."""
import re
from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2,explicit_choice

def own_explicit_choice(raw):
 choice=explicit_choice(raw)
 if choice is not None:return choice
 if not isinstance(raw,str):return None
 text=raw.strip()
 single=re.fullmatch(r'''(["'])\s*([A-Ea-e])\s*\1\s*[.,;:!?]?''',text)
 if single:return single.group(2).upper()
 quoted_option=re.match(r'''^["']\s*([A-Ea-e])\s*[.)](?=\s|$)''',text)
 return quoted_option.group(1).upper() if quoted_option else None

class SmolDoclingNativeExplicitQuotedLabels(NativeExplicitMCV2):
 def _extract_answer(self,raw_output,is_multiple_choice=False):
  return own_explicit_choice(raw_output) if is_multiple_choice else None
 def run(self,model,samples,**kwargs):
  predictions,references,extra,statistics=super().run(model,samples,**kwargs)
  for i,item in enumerate(extra):
   if own_explicit_choice(item.get('raw_output')) is None:predictions[i]=''
  return predictions,references,extra,statistics

"""Independent original Accuracy normalization, with explicit no-article credit audit."""
import re
def normalized(value):
 return str(value).replace('\u0120',' ').replace('\u010a',' ').strip().strip('*').rstrip('.,;:!?*\'"()[]{}').strip().upper()
def original_extracted(value):
 text=str(value).replace('\u0120',' ').replace('\u010a',' ')
 rules=[r'<answer>\s*([A-Ea-e])\s*</answer>',r'<[^>]*?>\s*([A-Ea-e])\s*</[^>]*?>',r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Ea-e])',r'(?:correct\s+)?option\s+(?:is\s+)?([A-Ea-e])\b',r'(?:choose|select|pick)\s+([A-Ea-e])\b',r'(?:^|\n)\s*([A-Ea-e])\s*$',r'^\s*([A-Ea-e])\s*\.',r'\b([A-Ea-e])\b']
 for i,rule in enumerate(rules):
  flags=re.I if i<5 else (re.M if i==5 else 0)
  found=re.findall(rule,text,flags)
  if found:return (found[0] if i==len(rules)-1 else found[-1]).strip().upper()
 return None
def original_accuracy_sample(prediction,reference):
 if prediction is None:return 0
 p=normalized(prediction);r=normalized(reference)
 if p==r:return 1
 if len(p)==1 and p in 'ABCDE':return 0
 return int(original_extracted(prediction)==r)
def safe_credit_reason(raw,choice,reference,credit):
 """Never accept an original-metric fallback that credits an ordinary article."""
 if not credit:return None
 if choice is not None:return 'NativeExplicitMCV2_explicit_option'
 # The actual disputed record has an uppercase option directly adjoining
 # markup. This cannot be an English article followed by prose.
 match=re.match(r'^\s*([A-E])(?=<[^>]+>)',raw)
 if match and match[1]==normalized(reference):return 'original_Accuracy_uppercase_option_immediately_followed_by_markup'
 raise AssertionError('Original Accuracy credited a non-v2 response without an independently unambiguous option; possible ordinary-article fallback: '+repr(raw))

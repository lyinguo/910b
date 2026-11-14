"""Original stemmed ROUGE-L arithmetic, complete fixed denominator."""
import re
import numpy as np
from universal_eval.metric.interface import IMetric
from universal_eval.metric.rouge_910b_porter import PorterStemmer
def tokens(s):
 terms=re.sub(r'[^a-z0-9]+',' ',str(s).lower()).split();stem=PorterStemmer()
 return [stem.stem(t) if len(t)>3 else t for t in terms if re.fullmatch(r'[a-z0-9]+',t)]
def score_one(pred,ref):
 a=tokens(' '.join(ref) if isinstance(ref,list) else ref);b=tokens(pred or '')
 if not a or not b:return 0.0
 prev=[0]*(len(b)+1)
 for x in a:
  cur=[0]
  for j,y in enumerate(b,1):cur.append(prev[j-1]+1 if x==y else max(prev[j],cur[-1]))
  prev=cur
 precision=prev[-1]/len(b);recall=prev[-1]/len(a)
 return 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
class OriginalGraniteMLXRougeL(IMetric):
 @property
 def metric_name(self):return 'ROUGE-L'
 @property
 def supported_tasks(self):return ['image-to-text','image-text-to-text']
 def compute(self,predictions,references,extra_info=None):
  assert len(predictions)==len(references)>0
  return float(np.mean([score_one(p,r) for p,r in zip(predictions,references)]))

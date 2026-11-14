"""Offline original whitespace/case-sensitive sentence BLEU-4, method1 epsilon0.1, fixed full denominator."""
import collections,math
from universal_eval.metric.interface import IMetric
def original_bleu4(prediction,reference):
 hypothesis=str(prediction or '').strip().split();refs=[reference] if isinstance(reference,str) else reference
 references=[str(r).strip().split() for r in refs]
 if not hypothesis or not references or any(not x for x in references):return 0.
 probabilities=[]
 for n in range(1,5):
  counts=collections.Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1))
  maximum={}
  for ref in references:
   ref_counts=collections.Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
   for gram in counts:maximum[gram]=max(maximum.get(gram,0),ref_counts[gram])
  numerator=sum(min(value,maximum.get(gram,0)) for gram,value in counts.items());denominator=max(1,len(hypothesis)-n+1)
  if n==1 and numerator==0:return 0.
  probabilities.append((numerator if numerator else .1)/denominator)
 closest=min((len(r) for r in references),key=lambda x:(abs(x-len(hypothesis)),x))
 penalty=1. if len(hypothesis)>closest else math.exp(1.-closest/len(hypothesis))
 return penalty*math.exp(math.fsum(.25*math.log(p) for p in probabilities))
class OriginalCaptionBLEU4910B(IMetric):
 @property
 def metric_name(self):return 'BLEU-4'
 @property
 def supported_tasks(self):return ['image-to-text']
 def compute(self,predictions,references,extra_info=None):
  if len(predictions)!=len(references) or not predictions:raise ValueError('Complete fixed sample set required')
  return sum(original_bleu4(p,r) for p,r in zip(predictions,references))/len(predictions)

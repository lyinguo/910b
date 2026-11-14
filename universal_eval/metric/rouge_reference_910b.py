"""Offline English stemmed ROUGE-L F1 using existing NLTK Porter software, with every fixed sample in the denominator."""
import re,math
from universal_eval.metric.interface import IMetric
from universal_eval.metric.rouge_910b_porter import PorterStemmer

def words(text):
    stemmer=PorterStemmer()
    terms=re.sub(r'[^a-z0-9]+',' ',str(text).lower()).split()
    return [stemmer.stem(x) if len(x)>3 else x for x in terms if re.fullmatch(r'[a-z0-9]+',x)]

def rouge_l_f1(prediction,reference):
    a=words(' '.join(reference) if isinstance(reference,list) else reference);b=words(prediction or '')
    if not a or not b:return 0.0
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b,1):current.append(previous[j-1]+1 if x==y else max(previous[j],current[-1]))
        previous=current
    return 2.0*previous[-1]/(len(a)+len(b))

class OfflineStemmedRougeL910B(IMetric):
    @property
    def metric_name(self):return 'ROUGE-L'
    @property
    def supported_tasks(self):return ['image-to-text','image-text-to-text','video-text-to-text']
    def compute(self,predictions,references,extra_info=None):
        if len(predictions)!=len(references) or not predictions:raise ValueError('A complete, nonempty fixed sample set is required')
        scores=[rouge_l_f1(p,r) for p,r in zip(predictions,references)]
        if any(not math.isfinite(x) for x in scores):raise ValueError('Nonfinite ROUGE score')
        return sum(scores)/len(scores)

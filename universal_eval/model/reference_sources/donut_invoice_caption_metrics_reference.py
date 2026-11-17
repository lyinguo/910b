import bisect,collections,math,re
from universal_eval.metric.rouge_910b_porter import PorterStemmer
stem=PorterStemmer()

def words(s):return [stem.stem(v) if len(v)>3 else v for v in re.findall('[a-z0-9]+',str(s).lower())]

def rouge(a,b):
 a=words(a);b=words(' '.join(b) if isinstance(b,list) else b)
 if not a or not b:return 0.
 pos={}
 for i,v in enumerate(b):pos.setdefault(v,[]).append(i)
 tail=[]
 for v in a:
  for n in reversed(pos.get(v,[])):
   at=bisect.bisect_left(tail,n)
   if at==len(tail):tail.append(n)
   else:tail[at]=n
 return 2.*len(tail)/(len(a)+len(b))

def bleu(pred,reference):
 h=str(pred or '').strip().split();refs=[reference] if isinstance(reference,str) else reference;refs=[str(v).strip().split() for v in refs]
 if not h or not refs or any(not v for v in refs):return 0.
 ps=[]
 for n in range(1,5):
  hc=collections.Counter(zip(*(h[k:] for k in range(n))));rc=[collections.Counter(zip(*(v[k:] for k in range(n)))) for v in refs];c=sum(min(x,max(v[g] for v in rc)) for g,x in hc.items())
  if n==1 and c==0:return 0.
  ps.append((c if c else .1)/max(1,len(h)-n+1))
 rlen=min((len(v) for v in refs),key=lambda x:(abs(x-len(h)),x));bp=1. if len(h)>rlen else math.exp(1.-rlen/len(h));return bp*math.exp(math.fsum(.25*math.log(v) for v in ps))

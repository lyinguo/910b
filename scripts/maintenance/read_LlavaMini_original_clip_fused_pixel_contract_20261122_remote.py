import sys
sys.dont_write_bytecode=True
import inspect,json
from transformers import CLIPImageProcessor
items=[]
for n in ['_fuse_mean_std_and_rescale_factor','normalize','center_crop']:
 f=getattr(CLIPImageProcessor,n);items.append(dict(name=n,file=inspect.getfile(f),text=inspect.getsource(f)))
print(json.dumps(items,ensure_ascii=False))

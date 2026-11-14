"""Own Granite MLX HF image frontend with original Pillow10.4 pass order.

Only this private processor instance is changed. Installed modules, original
assets, interpolation, dimensions, tile order, and normalization stay intact.
The complete 200 original A100 pixel/token/mask records are verified separately.
"""
from pathlib import Path
import ast,inspect,json,textwrap,types
from PIL import Image
import transformers.image_transforms as image_transforms
from transformers import AutoProcessor

def original_PIL104_passes(image,size,resample=None,reducing_gap=None):
 assert reducing_gap is None,'Original branch has no reducing_gap'
 width,height=size
 if image.width!=width:image=image.resize((width,image.height),resample=resample)
 if image.height!=height:image=image.resize((image.width,height),resample=resample)
 return image

def own_processor(root):
 processor=AutoProcessor.from_pretrained(str(root),trust_remote_code=True,local_files_only=True)
 ip=processor.image_processor
 tf=ast.parse(textwrap.dedent(inspect.getsource(image_transforms.resize)))
 class OriginalPasses(ast.NodeTransformer):
  replaced=0
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='image' and n.func.attr=='resize':
    self.replaced+=1;return ast.copy_location(ast.Call(func=ast.Name('original_PIL104_passes',ast.Load()),args=[ast.Name('image',ast.Load()),*n.args],keywords=n.keywords),n)
   return n
 fix=OriginalPasses();tf=fix.visit(tf);ast.fix_missing_locations(tf);assert fix.replaced==1
 ns=image_transforms.resize.__globals__.copy();ns['original_PIL104_passes']=original_PIL104_passes
 exec(compile('from __future__ import annotations\n'+ast.unparse(tf),'own_GraniteMLX_PIL104_image_transform','exec'),ns);own_resize=ns['resize']
 base=next(c.__dict__['resize'] for c in type(ip).__mro__[1:] if 'resize' in c.__dict__)
 base_ns=base.__globals__.copy();assert base_ns['np_resize'] is image_transforms.resize;base_ns['np_resize']=own_resize
 own_base=types.FunctionType(base.__code__,base_ns,base.__name__,base.__defaults__,base.__closure__);own_base.__kwdefaults__=base.__kwdefaults__
 top=type(ip).resize;tree=ast.parse(textwrap.dedent(inspect.getsource(top)))
 class OwnBase(ast.NodeTransformer):
  replaced=0
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and n.func.attr=='resize' and isinstance(n.func.value,ast.Call) and isinstance(n.func.value.func,ast.Name) and n.func.value.func.id=='super':
    self.replaced+=1;return ast.copy_location(ast.Call(func=ast.Name('own_base',ast.Load()),args=[ast.Name('self',ast.Load()),*n.args],keywords=n.keywords),n)
   return n
 fix2=OwnBase();tree=fix2.visit(tree);ast.fix_missing_locations(tree);assert fix2.replaced==1
 top_ns=top.__globals__.copy();top_ns['own_base']=own_base
 exec(compile('from __future__ import annotations\n'+ast.unparse(tree),'own_GraniteMLX_PIL104_processor_instance','exec'),top_ns)
 ip.resize=types.MethodType(top_ns['resize'],ip)
 return processor

class OwnGraniteMLXHFInputs:
 def __init__(self,root):self.processor=own_processor(Path(root))
 def prepare(self,prompt,media):
  image=Image.open(media).convert('RGB')
  messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True)
  return self.processor(text=text,images=[image],return_tensors='pt'),text

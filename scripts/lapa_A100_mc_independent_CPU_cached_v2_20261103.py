"""Own-resource independent input replay for the original two Gemma3 letter scorers."""
import pathlib,json,hashlib,re,functools
import torch,numpy as np
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from torchvision.transforms import InterpolationMode
from tokenizers import Tokenizer
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def tr(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fp(t)}
@functools.lru_cache(maxsize=1)
def own_rust_tokenizer(path):return Tokenizer.from_file(path)

def independent_inputs(root,prompt,media):
 root=pathlib.Path(root);pc=json.loads((root/'preprocessor_config.json').read_text());assert pc['resample']==2 and pc['size']=={'height':896,'width':896} and pc['image_mean']==pc['image_std']==[.5,.5,.5]
 with Image.open(media) as image:x=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).unsqueeze(0)
 x=tvF.resize(x,[896,896],interpolation=InterpolationMode.BILINEAR,antialias=True).float();pixels=(x-127.5)/127.5
 matches=list(re.finditer(r'^[A-E]\.\s+.*$',prompt,re.MULTILINE));options='\n'.join(m.group(0).strip() for m in matches)
 question=prompt
 for m in matches:question=question.replace(m.group(0),'')
 question=re.sub(r'\n+\s*Choose\s+the\s+correct\s+option\s*:?\s*','\n',question,flags=re.IGNORECASE).strip()
 completion='Question: '+question+('\n\n'+options+'\n\nAnswer:' if options else '\nAnswer:')
 rendered='<bos><start_of_image>'+completion
 expanded=rendered.replace('<start_of_image>','\n\n<start_of_image>'+'<image_soft_token>'*256+'<end_of_image>\n\n')
 rust=own_rust_tokenizer(str(root/'tokenizer.json'));ids=torch.tensor([rust.encode(expanded,add_special_tokens=True).ids],dtype=torch.int64)
 return {'pixel_values':pixels,'input_ids':ids,'attention_mask':torch.ones_like(ids),'token_type_ids':(ids==262144).to(torch.int64)},rendered,rust

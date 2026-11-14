"""Own T5Gemma2 canonical input, independent of HF processors; original BF16/TV2/Rust math."""
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, AddedToken
import json
from torchvision.transforms.v2 import functional as tvF

def canonical_text(prompt):
    prompt=re.sub(r'\n*Answer the question by providing only the option letter.*|\n*Provide only the final answer\.\s*$','',prompt,flags=re.S).rstrip()
    if re.search(r'(?:^|\n)\s*[A-Ea-e]\.\s',prompt):prompt+='\nAnswer: The answer is'
    return prompt if prompt.startswith('<start_of_image>') else '<start_of_image> '+prompt

class OwnCanonicalFrontend:
    def __init__(self,weight_dir):
        self.weight_dir=Path(weight_dir)
        self.tokenizer=Tokenizer.from_file(str(self.weight_dir/'tokenizer.json'))
        # Original GemmaTokenizer instance marks its own mask_token special; raw stored JSON does not.
        own_map=json.loads((self.weight_dir/'tokenizer_config.json').read_bytes())
        mask=own_map['mask_token'];mask=mask['content'] if isinstance(mask,dict) else mask
        assert mask=='<mask>' and self.tokenizer.token_to_id(mask)==4
        n=self.tokenizer.get_vocab_size()
        self.tokenizer.add_special_tokens([AddedToken(mask,normalized=False,special=True)])
        assert self.tokenizer.get_vocab_size()==n and self.tokenizer.token_to_id(mask)==4
        self.full_image_sequence='\n\n<start_of_image>'+('<image_soft_token>'*256)+'<end_of_image>\n\n'
        assert self.tokenizer.token_to_id('<start_of_image>')==255999
        assert self.tokenizer.token_to_id('<end_of_image>')==256000
        assert self.tokenizer.token_to_id('<image_soft_token>')==256001
    def encode(self,prompt,media_path):
        model_text=canonical_text(prompt)
        assert model_text.count('<start_of_image>')==1
        expanded=model_text.replace('<start_of_image>',self.full_image_sequence)
        enc=self.tokenizer.encode(expanded,add_special_tokens=True)
        ids=torch.tensor([enc.ids],dtype=torch.int64)
        mask=torch.ones_like(ids);modality=(ids==256001).to(torch.int64)
        with Image.open(media_path) as im:
            rgb=im.convert('RGB')
            u8=torch.from_numpy(np.array(rgb,dtype=np.uint8,copy=True)).permute(2,0,1).unsqueeze(0).contiguous()
        resized=tvF.resize(u8,[896,896],interpolation=tvF.InterpolationMode.BILINEAR,antialias=True)
        pixel=tvF.normalize(resized.to(torch.float32),[127.5]*3,[127.5]*3)
        return dict(input_ids=ids,attention_mask=mask,token_type_ids=modality,pixel_values=pixel),model_text,expanded

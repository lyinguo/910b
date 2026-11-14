"""Own image-only SmolVLM2 input contract, without its unused video-number dependency."""
import json
import types
from pathlib import Path
class OwnSmolVLM2ImageInputs:
    def __init__(self,root):
        from transformers import AutoTokenizer,AutoImageProcessor
        self.root=Path(root)
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.root),trust_remote_code=True,local_files_only=True)
        self.image_processor=AutoImageProcessor.from_pretrained(str(self.root),trust_remote_code=True,local_files_only=True)
        # Pillow 10.4, used by the selected A100 run, resizes the horizontal
        # uint8 axis before the vertical one. Bind only this own processor.
        import numpy as np
        from PIL import Image
        from transformers.models.smolvlm.image_processing_pil_smolvlm import get_resize_output_image_size
        def original_axis_resize(_processor,image,size,resample=Image.Resampling.LANCZOS,**kwargs):
            assert not kwargs and int(resample)==int(Image.Resampling.LANCZOS)
            assert isinstance(image,np.ndarray) and image.dtype==np.uint8 and image.ndim==3 and image.shape[0]==3
            if size.longest_edge:
                height,width=get_resize_output_image_size(image,resolution_max_side=size.longest_edge)
            else:
                assert size.height and size.width
                height,width=size.height,size.width
            im=Image.fromarray(np.ascontiguousarray(image.transpose(1,2,0)))
            if im.width!=width:im=im.resize((width,im.height),Image.Resampling.LANCZOS)
            if im.height!=height:im=im.resize((im.width,height),Image.Resampling.LANCZOS)
            return np.asarray(im,dtype=np.uint8).transpose(2,0,1).copy()
        self.image_processor.resize=types.MethodType(original_axis_resize,self.image_processor)

        cfg=json.loads((self.root/'processor_config.json').read_bytes())
        self.image_seq_len=cfg['image_seq_len'];assert self.image_seq_len==64
        chat=json.loads((self.root/'chat_template.json').read_bytes())
        self.chat_template=chat['chat_template'] if isinstance(chat,dict) else chat
        assert isinstance(self.chat_template,str) and self.chat_template
        self.image_token=getattr(self.tokenizer,'image_token','<image>')
        self.fake_image_token=getattr(self.tokenizer,'fake_image_token','<fake_token_around_image>')
        self.global_image_token=getattr(self.tokenizer,'global_image_token','<global-img>')
    def render(self,prompt):
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        return self.tokenizer.apply_chat_template(messages,chat_template=self.chat_template,
            add_generation_prompt=True,tokenize=False)
    def __call__(self,text,image):
        assert isinstance(text,str) and text.count(self.image_token)==1
        # These are the native SmolVLMProcessor image kwargs, not video defaults.
        vision=self.image_processor(images=[[image]],return_row_col_info=True,return_tensors='pt')
        rows=vision.pop('rows',None);cols=vision.pop('cols',None)
        r=0 if rows is None else int(rows[0][0]);c=0 if cols is None else int(cols[0][0])
        if r==0 and c==0:
            expanded=self.fake_image_token+self.global_image_token+self.image_token*self.image_seq_len+self.fake_image_token
        else:
            parts=[]
            for y in range(r):
                parts.append(''.join(self.fake_image_token+f'<row_{y+1}_col_{x+1}>'+self.image_token*self.image_seq_len for x in range(c))+'\n')
            expanded=''.join(parts)+'\n'+self.fake_image_token+self.global_image_token+self.image_token*self.image_seq_len+self.fake_image_token
        before,after=text.split(self.image_token)
        tokens=self.tokenizer([before+expanded+after],add_special_tokens=True,padding=True,
            is_split_into_words=False,return_tensors='pt')
        return dict(vision)|dict(tokens)
    def batch_decode(self,tokens,**kwargs):
        return self.tokenizer.batch_decode(tokens,**kwargs)


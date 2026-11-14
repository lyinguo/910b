"""Own InternVL3.5-2B-Flash inputs, independent of HF image/processor code."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize

MODEL_ID = 'OpenGVLab/InternVL3_5-2B-Flash'

def clean_text(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(text))
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def original_answer_instruction(prompt):
    if 'Choose the correct option' in prompt:
        return prompt + '\n\nReply with only the letter of the correct option.'
    if 'Final Answer:' in prompt or 'Answer:' in prompt:
        return prompt
    return prompt + '\n\nProvide only the final answer.'

def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

class OwnFlash2BInputs:
    def __init__(self, weights_root):
        self.root = Path(weights_root)
        self.config = json.loads((self.root / 'preprocessor_config.json').read_bytes())
        model = json.loads((self.root / 'config.json').read_bytes())
        assert model['architectures'] == ['InternVLChatModel']
        assert model['llm_config']['architectures'] == ['Qwen3ForCausalLM']
        assert model['llm_config']['hidden_size'] == 2048
        assert model['llm_config']['vocab_size'] == 151936
        assert model['flash_mode'] is True
        assert model['force_image_size'] == 448 and model['downsample_ratio'] == .5
        assert self.config['size'] == dict(height=448, width=448)
        assert self.config['do_resize'] and self.config['do_normalize'] and self.config['do_rescale']
        assert self.config['resample'] == 3
        self.tokenizer = Tokenizer.from_file(str(self.root / 'tokenizer.json'))
        self.image_token_id = self.tokenizer.token_to_id('<IMG_CONTEXT>')
        assert self.image_token_id == 151671

    def grid(self, height, width):
        minimum, maximum = self.config['min_patches'], self.config['max_patches']
        choices = [(x, y) for x in range(1, maximum+1) for y in range(1, maximum+1)
                   if minimum <= x*y <= maximum]
        choices.sort(key=lambda xy: xy[0]*xy[1])
        ratio, area = width / height, width * height
        best, delta = (1, 1), float('inf')
        for x, y in choices:
            d = abs(ratio - x/y)
            if d < delta:
                best, delta = (x, y), d
            elif d == delta and area > .5 * 448 * 448 * x * y:
                best = (x, y)
        return best

    def pixels(self, path):
        image = Image.open(path).convert('RGB')
        source = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).unsqueeze(0)
        columns, rows = self.grid(image.height, image.width)
        scaled = resize(source, [rows*448, columns*448], interpolation=InterpolationMode.BICUBIC, antialias=True)
        tiles = [scaled[:, :, y*448:(y+1)*448, x*448:(x+1)*448]
                 for y in range(rows) for x in range(columns)]
        if len(tiles) != 1:
            tiles.append(resize(source, [448, 448], interpolation=InterpolationMode.BICUBIC, antialias=True))
        values = torch.cat(tiles, dim=0).to(torch.float32)
        scale = self.config['rescale_factor']
        mean = (torch.tensor(self.config['image_mean'], dtype=torch.float32) * (1.0 / scale)).view(1, 3, 1, 1)
        std = (torch.tensor(self.config['image_std'], dtype=torch.float32) * (1.0 / scale)).view(1, 3, 1, 1)
        values = values.sub(mean).div(std)
        return values, values.to(torch.bfloat16)

    def prepare(self, actual_model_prompt, media_path):
        f32, bf16 = self.pixels(media_path)
        text = '<|im_start|>user\n<img>' + '<IMG_CONTEXT>' * (256 * f32.shape[0])
        text += '</img>\n' + actual_model_prompt + '<|im_end|>\n<|im_start|>assistant\n'
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        ids = torch.tensor([encoded.ids], dtype=torch.int64)
        mask = torch.tensor([encoded.attention_mask], dtype=torch.int64)
        assert int((ids == self.image_token_id).sum()) == 256 * f32.shape[0]
        return dict(pixels_F32=f32, pixels_BF16=bf16, input_ids=ids, attention_mask=mask, text=text)

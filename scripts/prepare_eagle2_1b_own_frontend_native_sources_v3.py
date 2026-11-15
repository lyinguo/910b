"""Freeze own Eagle2 sources and independently check all 200 frontends, without NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json,os,copy
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
import torch,numpy as np
from PIL import Image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from transformers.image_utils import get_image_size,ChannelDimension
from transformers.image_processing_backends import TorchvisionBackend
from safetensors import safe_open
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from universal_eval.model.adapters.eagle2_1b_own_frontend_v1 import OwnEagle2Inputs,tensor_sha
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Eagle2_5_VLForConditionalGeneration/nvidia/Eagle2-1B');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Eagle2_1B_own_frontend200_native_source_preflight_v3_1845_20261115'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 torch.set_num_threads(4);H.mkdir(exist_ok=False)
 cfg=json.loads((W/'config.json').read_bytes());assert cfg['use_backbone_lora']==cfg['use_llm_lora']==0
 own=R/'universal_eval/model/eagle2_1b_own_native_20261115';own.mkdir(exist_ok=True);assert (own/'__init__.py').read_text()=='"""Own Eagle2-1B NVIDIA source with no unused peft import."""\n';(own/'__init__.py').write_text('"""Own Eagle2-1B NVIDIA source with no unused peft import."""\n')
 source_maps={}
 for name in ['configuration_eagle2_5_vl.py','modeling_eagle2_5_vl.py']:
  src=(W/name).read_text();new='\n'.join(line for line in src.splitlines() if line!='from peft import LoraConfig, get_peft_model')+'\n'
  before=ast.parse(src);after=ast.parse(new);before.body=[n for n in before.body if not(isinstance(n,ast.ImportFrom) and n.module=='peft')]
  assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
  assert (own/name).read_text()==new;(own/name).write_text(new,encoding='utf-8');source_maps[name]=dict(original_SHA256=sha(W/name),new_SHA256=sha(own/name),all_class_and_method_AST_identical=True,unused_peft_import_only_excluded=True)
 from universal_eval.model.eagle2_1b_own_native_20261115.configuration_eagle2_5_vl import Eagle2_5_VLConfig
 from universal_eval.model.eagle2_1b_own_native_20261115.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
 from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
 from transformers.models.siglip.modeling_siglip import SiglipVisionModel
 config=Eagle2_5_VLConfig(**cfg);config._attn_implementation='eager';config.vision_config._attn_implementation='eager';config.text_config._attn_implementation='eager'
 with torch.device('meta'):
  vision=SiglipVisionModel(config.vision_config);language=Qwen2ForCausalLM(config.text_config);model=Eagle2_5_VLForConditionalGeneration(config,vision_model=vision,language_model=language)
 params=dict(model.named_parameters(remove_duplicate=False));meta={n:dict(shape=list(v.shape),dtype=str(v.dtype)) for n,v in params.items()}
 manifest={}
 with safe_open(str(W/'model.safetensors'),framework='pt',device='cpu') as f:
  keys=set(f.keys());mapping={n:n.replace('vision_model.vision_model.','vision_model.') for n in keys};assert len(set(mapping.values()))==len(keys) and set(mapping.values())==set(params)
  for n in sorted(keys):
   t=f.get_tensor(n);assert list(t.shape)==meta[mapping[n]]['shape'] and t.dtype==torch.bfloat16
   manifest[n]=dict(runtime_name=mapping[n],shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=tensor_sha(t),cast_SHA256=tensor_sha(t))
 del model,vision,language
 (H/'own_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 s=(W/'image_processing_eagle2_5_vl_fast.py').read_text();tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Eagle2_5_VLImageProcessorFast')
 methods=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['find_closest_aspect_ratio','_get_image_patches']]
 for n in methods:n.decorator_list=[]
 orig=ast.Module(body=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop'),ast.ClassDef(name='OriginalOwnImageMath',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(orig)
 env=dict(torch=torch,F=F,List=list,get_image_size=get_image_size,ChannelDimension=ChannelDimension);exec(compile(orig,str(W/'image_processing_eagle2_5_vl_fast.py'),'exec'),env);reference=env['OriginalOwnImageMath']();normalizer=TorchvisionBackend()
 inputs=OwnEagle2Inputs(W);dataset=Path('/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl');rows=[json.loads(s) for s in dataset.read_text().splitlines()];assert len(rows)==4329
 records=[]
 from universal_eval.model.adapters.internvl35_flash2b_own_frontend_v3_original_TV2_uint8_F32_1518_20261112 import clean_text
 for i,row in enumerate(rows[:200]):
  path=(dataset.parent/row['media_path']).resolve();prompt=clean_text(row['prompt']);data=inputs.prepare(prompt,path);image=Image.open(path).convert('RGB');src=torch.from_numpy(np.array(image,dtype=np.uint8,copy=True)).permute(2,0,1)
  tiles=reference._get_image_patches(src,min_num=1,max_num=12,size=(448,448),tile_size=448,use_thumbnail=True,interpolation=InterpolationMode.BICUBIC,pad_during_tiling=False)
  ref=normalizer.rescale_and_normalize(torch.stack(tiles),True,1/255,True,[.5]*3,[.5]*3)
  assert torch.equal(ref,data['pixels_F32']) and torch.equal(ref.bfloat16(),data['pixels_BF16']),(i,'Own original image math differs')
  original_text='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image 1><img>'+'<IMG_CONTEXT>'*(len(tiles)*256)+'</img>'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
  assert original_text==data['text'];enc=inputs.tokenizer([original_text],padding=True,return_tensors='pt');assert torch.equal(enc.input_ids,data['input_ids']) and torch.equal(enc.attention_mask,data['attention_mask'])
  records.append(dict(index=i,actual_model_prompt=prompt,media_path=str(path),media_file_SHA256=sha(path),reference=row.get('answer',row.get('reference')),text=data['text'],input_ids=data['input_ids'][0].tolist(),attention_mask=data['attention_mask'][0].tolist(),tiles=len(tiles),**{k+'_SHA256':tensor_sha(data[k]) for k in ['input_ids','attention_mask','pixels_F32','pixels_BF16']}))
 (H/'own_frontend200.json').write_text(json.dumps(records,ensure_ascii=False))
 assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.py','.txt']}
 result=dict(passed=True,model_id='nvidia/Eagle2-1B',weight_path=str(W),own_weight_files={'model.safetensors':dict(SHA256=sha(W/'model.safetensors'),bytes=(W/'model.safetensors').stat().st_size)},own_assets_SHA256=assets,source_tensor_count=len(manifest),own_source_manifest=str(H/'own_source_manifest.json'),own_source_manifest_SHA256=sha(H/'own_source_manifest.json'),own_frontend200=str(H/'own_frontend200.json'),own_frontend200_SHA256=sha(H/'own_frontend200.json'),dataset=str(dataset),dataset_SHA256=sha(dataset),own_native_source_map=source_maps,all_200_original_own_image_math_tokens_masks_exact=True,original_saved_backend='Saved own adapter forces eager; no historical A100 runtime or tensors captured',no_NPU_inference_yet=True,no_generic_stub_initializer_called=True,no_packages_installed_or_data_downloaded=True)
 (H/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

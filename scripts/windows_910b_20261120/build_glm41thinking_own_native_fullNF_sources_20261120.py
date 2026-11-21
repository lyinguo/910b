"""Declare a private owned GLM4.1 Thinking port from verified architecture code."""
import ast,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent;P=Path('D:/code-lv/.tmp/batch_910b_20261030')
old='autoglm_phone9b_single_own_original_BF16_2048';new='glm41thinking_own_original_VLLM_BF16_8192'
oldfiles=[old+'_20261120_native.py','verify_'+old+'_native_v1_20261120.py',old+'_20261120_MC.py']
mapping={};originals={}
def digest(b):return hashlib.sha256(b).hexdigest()
for name in oldfiles:
 raw=(P/name).read_bytes();originals[name]=digest(raw);text=raw.decode()
 text=text.replace(old,new).replace('zai-org/AutoGLM-Phone-9B','zai-org/GLM-4.1V-9B-Thinking').replace('zai-org_AutoGLM-Phone-9B','zai-org_GLM-4.1V-9B-Thinking')
 text=text.replace('AutoGLMPhone9BSingle','GLM41Thinking').replace('AutoGLMPhone9B','GLM41Thinking').replace('AutoGLM-Phone-9B','GLM-4.1V-9B-Thinking')
 text=text.replace('2048','8192').replace('original_enable_thinking=False','original_enable_thinking=None').replace("line['original_enable_thinking'] is False","line['original_enable_thinking'] is None")
 text=text.replace('standard HF original generic default thinkingFalse (own template ignores it) greedy8192 path','saved original VLLM greedy8192 native input and stopping contract, native HF SDPA execution, no original chat thinking override')
 text=text.replace('BF16 standard HF original existing thinkingFalse greedy8192 MMBench Accuracy','BF16 original VLLM8192 Greedy MMBench Accuracy/no thinking override; all native input200 proven; full native HF NPU cache/values audited')
 text=text.replace('autoglm_phone9b_original_existing_thinkingFalse_greedy_explicit_MC','glm41thinking_original_VLLM8192_greedy_explicit_MC')
 if name.endswith('_native.py'):
  needle="  self.lineage=dict(original_enable_thinking=None"
  assert text.count(needle)==1;text=text.replace(needle,"  libs.update(self.inputs.native_media_external_SHA256)\n"+needle)
 elif name.startswith('verify_'):
  needle=" W=Path(p['weight_path'])";assert text.count(needle)==1
  text=text.replace(needle,needle+"\n for n,h in p['original_native_input_library_source_SHA256'].items():check(n,h)")
 target=name.replace(old,new);ast.parse(text);(V/target).write_text(text,encoding='utf8',newline='\n');mapping[target]=digest(text.encode())
frontend=r'''"""Own GLM4.1 Thinking original VLLM media, clean prompt and chat template."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,inspect,os,re,types,typing,torch,numpy as np
from universal_eval.model.reference_sources.glm41thinking_owned_tokenizer_config_20261120 import load_owned_rust
from transformers import AutoProcessor
from vllm.multimodal.media.image import ImageMediaIO
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def flatten_tensors(v):return {n:t for n,t in v.items() if torch.is_tensor(t)}
class OwnGLM41ThinkingInputs:
 def __init__(self,W):
  self.processor=AutoProcessor.from_pretrained(W,local_files_only=True,trust_remote_code=False);self.tokenizer=self.processor.tokenizer;self.rust=load_owned_rust(W);assert self.rust.get_vocab()==self.tokenizer.get_vocab()
  self.media=ImageMediaIO();assert self.media.image_mode=='RGB' and self.media.rgba_background_color==(255,255,255)
  src=Path('/root/projects/comperror/universal_eval/model/adapters/vllm_adapter.py');raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()=='da65f91d00696a2becf43121c793de985eb8364fd18f2a0164ff8d53541602db'
  cl=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='VLLMModel');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_build_message');rx=next(n for n in cl.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_IMAGE_PLACEHOLDER_RE' for x in n.targets))
  ns=dict(re=re,os=os,Optional=typing.Optional,List=list,Dict=dict,Any=typing.Any);exec(compile(ast.Module(body=[rx,fn],type_ignores=[]),str(src),'exec'),ns)
  self.original_build_message=ns['_build_message'];self.original_input=types.SimpleNamespace(_IMAGE_PLACEHOLDER_RE=ns['_IMAGE_PLACEHOLDER_RE'],_prompt_prefix='')
  ext=Path(inspect.getfile(ImageMediaIO));self.native_media_external_SHA256={str(ext):hashlib.sha256(ext.read_bytes()).hexdigest()}
  assert 'enable_thinking' not in (Path(W)/'chat_template.jinja').read_text()
 def prepare(self,prompt,media):
  message=self.original_build_message(self.original_input,prompt,str(media));clean=message[0]['content'][1]['text'];image=self.media.load_file(Path(media))
  messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=clean)])]
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  return text,self.processor(text=text,images=[image],return_tensors='pt',padding=True)
 def check_pixels(self,media,v):
  image=self.media.load_file(Path(media));ip=self.processor.image_processor;raw=ip(images=[image],do_rescale=False,do_normalize=False,return_tensors='pt');assert torch.equal(raw['image_grid_thw'],v['image_grid_thw'])
  a=raw['pixel_values'].numpy().astype(np.float32).reshape(-1,3,2,14,14);f=np.float32(1.0/ip.rescale_factor);mean=(np.asarray(ip.image_mean,np.float32)*f).reshape(1,3,1,1,1);std=(np.asarray(ip.image_std,np.float32)*f).reshape(1,3,1,1,1);assert np.array_equal(((a-mean)/std).reshape(v['pixel_values'].shape),v['pixel_values'].numpy())
  return True
'''
fname=new+'_20261120_frontend.py';ast.parse(frontend);(V/fname).write_text(frontend,encoding='utf8',newline='\n');mapping[fname]=digest(frontend.encode())
oldtree=ast.parse((P/oldfiles[0]).read_bytes());newtree=ast.parse((V/(new+'_20261120_native.py')).read_bytes())
oldclass=next(n for n in oldtree.body if isinstance(n,ast.ClassDef));newclass=next(n for n in newtree.body if isinstance(n,ast.ClassDef));observers={}
for method in ['_record','_install','unload','get_sample_execution','batch_predict','get_execution_lineage']:
 a=next(n for n in oldclass.body if isinstance(n,ast.FunctionDef) and n.name==method);b=next(n for n in newclass.body if isinstance(n,ast.FunctionDef) and n.name==method)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),method;observers[method]=digest(ast.dump(a,include_attributes=False).encode())
# Numeric reconstruction and full-head/40-cache audit remain exact. Only the
# owned front-end/source identity, original8192 budget and protocol lineage differ.
old_nf=ast.parse((P/oldfiles[1]).read_bytes());new_nf=ast.parse((V/('verify_'+new+'_native_v1_20261120.py')).read_bytes())
def numeric_with(tree):
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
 return next(n for n in ast.walk(f) if isinstance(n,ast.With) and 'ReadOnlyNPZ' in ast.unparse(n.items[0].context_expr))
assert ast.dump(numeric_with(old_nf),include_attributes=False)==ast.dump(numeric_with(new_nf),include_attributes=False)
review=dict(passed=True,old_own_architecture_source_SHA256=originals,new_owned_source_SHA256=mapping,exact_original_observer_AST_SHA256=observers,complete_numeric_NPZ_head_KV_mask_RoPE_and_all_values_block_AST_unchanged=True,no_checkpoint_or_task_input_or_output_reused=True,protocol_changes=dict(model_id='zai-org/GLM-4.1V-9B-Thinking',own_original_max_tokens=8192,own_original_no_thinking_override=True,own_original_VLLM_white_background_media_and_exact_prompt_cleaning=True),native_frontend200_must_pass_before_true3=True)
(V/'GLM41Thinking_own_native_source_diff_scope_20261120.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review,ensure_ascii=False))

"""Close the own saved tokenizer registration interface without learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,inspect,json,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'HancomHiLlava31_original_registered_tokenizer_generation_interface_no_NN_v1_20261122'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM/HancomInSpaceAI/HiLLaVA-llama_3_1-8b')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 H.mkdir(exist_ok=False)
 tf=R/'universal_eval/model/adapters/transformers_vlm.py';assert sha(tf)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 assets={n:sha(W/n) for n in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','generation_config.json']}
 assert assets['config.json']=='4e58f37179a393274f602077e6ee02bbc4c12f78330249cd0b99c308876c622c' and assets['tokenizer_config.json']=='48b3610fc48147796997cf6b41eb221e1a055bbf8e2235950251e931b5c635c3'
 # This is the original saved adapter's exact existing registration, not a stub.
 from universal_eval.model.adapters.transformers_vlm import TransformersVLM
 from transformers import AutoConfig,AutoTokenizer,GenerationConfig
 from tokenizers import Tokenizer
 config=AutoConfig.from_pretrained(W,local_files_only=True)
 tokenizer=AutoTokenizer.from_pretrained(W,local_files_only=True)
 generation=GenerationConfig.from_pretrained(W,local_files_only=True)
 raw=json.loads((W/'tokenizer_config.json').read_bytes());rust=Tokenizer.from_file(str(W/'tokenizer.json'))
 assert type(config).__name__=='LlavaLlamaConfig' and config.model_type=='llava_llama'
 assert tokenizer.get_vocab()==rust.get_vocab() and len(tokenizer)==128256 and tokenizer.chat_template==raw['chat_template']
 assert '26 Jul 2024' in tokenizer.chat_template and generation.eos_token_id==[128001,128008,128009] and generation.do_sample and generation.temperature==.5 and generation.top_p==.9
 before_vocab_SHA256=hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
 tokenizer.add_special_tokens({'additional_special_tokens':['<image>']})
 assert len(tokenizer)==128257 and tokenizer.convert_tokens_to_ids('<image>')==128256
 trials=[[128000,9906,11,1917,0,128009],[40,284,264,1296,13],[128006,78191,128007,271,32,13,128009]]
 decoded=[]
 for ids in trials:
  native=tokenizer.decode(ids,skip_special_tokens=True);reference=rust.decode(ids,skip_special_tokens=True);assert native==reference
  decoded.append(dict(ids=ids,native_decode=native,own_Rust_decode=reference))
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,actual_NEW_model_forwards=0,no_learned_CPU_or_NPU_forward=True,no_checkpoint_loaded=True,exact_original_TF_source_SHA256=sha(tf),own_assets_SHA256=assets,config_class=type(config).__name__,tokenizer_class=type(tokenizer).__name__,original_registration_function_source=inspect.getsource(__import__('universal_eval.model.adapters.transformers_vlm',fromlist=['_register_llava_llama'])._register_llava_llama),own_complete_vocab_SHA256=before_vocab_SHA256,own_original_chat_template_unchanged=True,own_special_image_id=128256,own_generation_config=generation.to_dict(),own_representative_decode_controls=decoded,own_existing_frontend200_and_CPU_constructor_not_replayed=True,remedy='Import the exact original saved TransformersVLM registration before own AutoTokenizer; no model graph/math/dtype/budget/EOS/scoring changes.',source_SHA256=sha(__file__))
 put(H/'review.json',result);put(H/'actual_completion.json',dict(UTC=result['UTC'],actual_returncode=0,review_SHA256=sha(H/'review.json'),actual_NEW_model_forwards=0));print(json.dumps(dict(H=str(H),review_SHA256=sha(H/'review.json'),passed=True,actual_NEW_model_forwards=0),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put(H/'actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),actual_NEW_model_forwards=0))
  raise

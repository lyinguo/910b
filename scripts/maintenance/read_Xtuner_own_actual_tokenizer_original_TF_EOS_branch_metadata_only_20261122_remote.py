import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,contextlib
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
with contextlib.redirect_stdout(sys.stderr):
 from transformers import AutoTokenizer,GenerationConfig
 W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM/xtuner/llava-llama-3-8b-v1_1-hf');tok=AutoTokenizer.from_pretrained(W,local_files_only=True,trust_remote_code=True);gen=GenerationConfig.from_pretrained(W,local_files_only=True)
 original=gen.eos_token_id;extra=tok.convert_tokens_to_ids('<|end|>');end=extra[0] if isinstance(extra,list) and extra else (None if isinstance(extra,list) else extra);kw={}
 if end is not None and end>=0 and end!=tok.unk_token_id:
  base=gen.eos_token_id;ids=base if isinstance(base,list) else ([base] if base is not None else [])
  if end not in ids:kw['eos_token_id']=ids+[end]
  kw['pad_token_id']=gen.pad_token_id if gen.pad_token_id is not None else end
 te=tok.eos_token_id;te=te[0] if isinstance(te,list) and te else (None if isinstance(te,list) else te)
 if te is not None and te>=0:
  cur=kw.get('eos_token_id',gen.eos_token_id);ids=cur if isinstance(cur,list) else ([cur] if cur is not None else [])
  if te not in ids:kw['eos_token_id']=ids+[te]
 values=dict(own_raw_generation_eos=original,own_actual_tokenizer_eos_id=tok.eos_token_id,own_actual_tokenizer_eos_token=tok.eos_token,own_actual_tokenizer_unk=tok.unk_token_id,own_phi_end_candidate=extra,exact_original_existing_TF_predict_6011_to6045_EOS_pad_overrides=kw,effective_original_EOS=kw.get('eos_token_id',gen.eos_token_id),no_model_parameter_load_or_forward=True,own_assets_SHA256={n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in ['config.json','generation_config.json','tokenizer_config.json','special_tokens_map.json'] if (W/n).exists()},original_source_SHA256=hashlib.sha256((R/'universal_eval/model/adapters/transformers_vlm.py').read_bytes()).hexdigest())
print(json.dumps(values,ensure_ascii=False))

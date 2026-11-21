import ast,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
old=V/'prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_20261120.py';text=old.read_text()
text=text.replace('GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v1_20261120','GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v2_own_added_tokens_20261120')
text=text.replace(' from tokenizers import Tokenizer',' from universal_eval.model.reference_sources.glm41thinking_owned_tokenizer_config_20261120 import load_owned_rust')
text=text.replace("rust=Tokenizer.from_file(str(W/'tokenizer.json'))","rust=load_owned_rust(W)")
start=text.index(' assets={p.name:sha(p)');end=text.index(" put('own704_physical_source_headers.json'",start)
text=text[:start]+''' assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.py','.txt']}
 previous=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v1_20261120'
 old_failure=js(previous/'actual_completion.json');assert old_failure['actual_returncode']==1 and 'rust.get_vocab()==processor.tokenizer.get_vocab()' in old_failure['error'] and old_failure['no_model_inference']
 files=js(previous/'own_whole_source_files.json');headers=js(previous/'own704_physical_source_headers.json')
 assert len(files)==4 and sum(v['bytes'] for v in files.values())==20585644104 and len(headers)==704 and all(v['dtype']=='BF16' for v in headers.values())
 for name,record in files.items():assert (W/name).stat().st_size==record['bytes']
 # Reuse this checkpoint's just-completed full SHA/header pass, not a second
 # full checkpoint scan during input repair. Readiness and NPU load verify all
 # own source bytes again before real inference.
'''+text[end:]
text=text.replace(" for n,r in files.items():assert sha(W/n)==r['SHA256']"," for n,r in files.items():assert (W/n).stat().st_size==r['bytes']")
text=text.replace('own_whole_weight_bytes=20585644104,','own_whole_weight_bytes=20585644104,own_full_weight_SHA_reused_from_immediate_zero_NN_v1=str(previous),own_added_tokens_decoder_from_own_tokenizer_config_only=True,')
ast.parse(text);p=V/'prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v2_own_added_tokens_20261120.py';p.write_text(text,encoding='utf8',newline='\n')
print(json.dumps(dict(source=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),only_input_tokenizer_configuration_repaired=True,no_old_NN_or_frontend200_replay=True)))

"""Build private own-model sources from the reviewed Gemma4 architecture code.
The source/model identity, original HF input contract and file layout are explicit;
checkpoint values, learned arrays and predictions are never borrowed.
"""
from pathlib import Path
import ast,json,hashlib
V=Path(__file__).resolve().parent
BASE=Path('D:/code-lv/.tmp/cd22s1')
MID='unsloth/gemma-4-E4B-it-MLX-8bit'
OLD='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX'
paths=[]
def write(rel,s):
 ast.parse(s)
 p=V/Path(rel).name
 assert not p.exists(),p
 p.write_text(s,encoding='utf-8',newline='\n')
 paths.append(dict(path=rel,local_path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
def once(s,a,b):
 assert s.count(a)==1,(a[:120],s.count(a))
 return s.replace(a,b)
src=(BASE/'scripts/prepare_prithiv_gemma4_e4b_own_original_BF16_four_shards_frontend200_v1_20261116.py').read_text(encoding='utf-8')
src=src.replace(OLD,MID).replace('PrithivGemma4E4B_own_original_BF16_adapter_template_false_pixel_patch_frontend200_v1_20261116','Gemma4E4BMLX8_own_original_Transformers_BF16_single_SF_frontend200_v1_20261122')
src=src[src.index('import sys'):]
src='"""Own BF16 physical source despite the MLX8 name; original Transformers input/default template. No learned forward."""\n'+src
start=src.index(" original=R/'universal_eval/model/adapters/gemma4_vllm.py'")
end=src.index(" put('source_tensor_files.json'",start)
src=src[:start]+''' contract=js(B/'Gemma4E4BMLX8_own_saved_Transformers_template_generation_contract_no_NN_20261122/own_original_interface_contract.json');assert contract['passed'] and contract['default_equals_false'] and not contract['default_equals_true'] and contract['original_template_passes_no_enable_thinking_kwarg']
 original=R/'universal_eval/model/adapters/transformers_vlm.py';assert sha(original)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3';ast.parse(original.read_bytes())
 headers={};source_files={};tensor_files={}
 name='model.safetensors';weight=W/name
 assert not (W/'model.safetensors.index.json').exists() and weight.stat().st_size==15992595884
 with weight.open('rb') as f:prefix=f.read(8);length=struct.unpack('<Q',prefix)[0];raw=f.read(length)
 headers={k:v for k,v in json.loads(raw).items() if k!='__metadata__'};tensor_files={k:name for k in headers}
 source_files[name]=dict(bytes=weight.stat().st_size,SHA256=sha(weight),header_SHA256=hashlib.sha256(prefix+raw).hexdigest(),tensor_count=len(headers))
 assert len(headers)==2130 and {v['dtype'] for v in headers.values()}=={'BF16'}
''' + src[end:]
oldline=next(l for l in src.splitlines() if l.startswith(" saved=R/"))
src=once(src,oldline," saved=R/'configs/Gemma4ForConditionalGeneration/unsloth--gemma-4-E4B-it-MLX-8bit.yaml';deployment=yaml.safe_load(saved.read_text());assert deployment['model']['name']==MID and deployment['model']['adapter']=='universal_eval/model/adapters/transformers_vlm.py:TransformersVLM' and deployment['model']['kwargs']['max_new_tokens']==2048 and deployment['model']['kwargs']['torch_dtype']=='bfloat16' and deployment['algorithm']['batch_size']==10")
src=once(src," rows=[json.loads(s) for s in DATA.read_text().splitlines()];assert len(rows)==4329;records=[];output=H/'frontend200.jsonl'",''' rows=[json.loads(s) for s in DATA.read_text().splitlines()];assert len(rows)==4329
 original_data=Path(deployment['data']['dataset_path']);original_rows=[json.loads(s) for s in original_data.read_text().splitlines()]
 assert [(r['prompt'],r['reference'],r['media_path']) for r in rows]==[(r['prompt'],r['reference'],r['media_path']) for r in original_rows]
 assert all(clean(r['prompt']) and str(r['reference']).strip() for r in rows),'Original filter_empty must not remove or reorder any row'
 from tokenizers import Tokenizer
 rust=Tokenizer.from_file(str(W/'tokenizer.json'));records=[];output=H/'frontend200.jsonl'
'''.rstrip())
src=once(src,"text=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False,enable_thinking=False)","text=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False)")
src=once(src,"inputs=processor(text=text,images=[image],return_tensors='pt',add_special_tokens=False)","inputs=processor(text=text,images=[image],return_tensors='pt',padding=True)")
src=once(src,"   assert (inputs['input_ids']==config.image_token_id).sum().item()==soft and inputs['input_ids'].shape[1]+2048<=4096",'''   assert (inputs['input_ids']==config.image_token_id).sum().item()==soft and inputs['input_ids'].shape[1]+2048<=config.text_config.max_position_embeddings
   assert text.count(processor.image_token)==1
   expanded=text.replace(processor.image_token,processor.boi_token+processor.image_token*soft+processor.eoi_token)
   independent_ids=rust.encode(expanded,add_special_tokens=False).ids
   assert independent_ids==inputs['input_ids'][0].tolist(),(i,'Independent own Rust image placeholder/token expansion differs')
   assert inputs['attention_mask'].tolist()==[[1]*len(independent_ids)]
   original_media=Path(original_rows[i]['media_path']);original_media=original_media if original_media.is_absolute() else original_data.parent/original_media
   assert sha(original_media)==sha(media),(i,'Own saved original task image differs')
   if i in [0,99,199]:
    np.savez_compressed(H/f'full_input_example_{i:05d}.npz',**{n:t.contiguous().numpy() for n,t in inputs.items()})''')
src=once(src,"t=t.to(torch.bfloat16) if t.is_floating_point() else t;actual[n]", "actual[n]")
src=src.replace('own_original_thinking_false_from_saved_adapter=True','own_original_no_enable_thinking_override=True').replace("generation=dict(max_new_tokens=2048,do_sample=False)","generation=dict(max_new_tokens=2048,do_sample=False,temperature=0.0)")
src=src.replace("phase='own_four_shards_full_file_SHA_and_exact_index_complete_no_NN'","phase='own_single_SF_all_BF16_file_SHA_and_header_complete_no_NN'")
src=src.replace('original_adapter_enable_thinking_false_preserved_not_newly_added=True,','original_template_default_false_preserved_without_any_override=True,original_saved_filter_empty_fixed200_equal=True,all200_independent_Rust_token_expansions_equal=True,original_dataset_path=str(original_data),original_dataset_SHA256=sha(original_data),')
src=src.replace('original2048_budget_and4096_request_limit_preserved=True','original2048_budget_and_native_context_preserved=True,original_native_context_length=config.text_config.max_position_embeddings,actual_source_is_BF16_not_native_8bit=True,original_pixel_input_dtype="torch.float32"')
write('scripts/prepare_gemma4_e4b_MLX8_own_original_Transformers_BF16_single_SF_frontend200_v1_20261122.py',src)

# The already accepted original HF graph and all lossless observation math are unchanged.
for rel,newrel in [('universal_eval/model/prithiv_gemma4_e4b_own_20261116/observer.py','universal_eval/model/gemma4_e4b_MLX8_own_20261122/observer.py'),('universal_eval/model/adapters/prithiv_gemma4_e4b_own_original_BF16_native_v1_20261116.py','universal_eval/model/adapters/gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v1_20261122.py')]:
 s=(BASE/rel).read_text(encoding='utf-8').replace(OLD,MID).replace('prithiv_gemma4_e4b_own_20261116','gemma4_e4b_MLX8_own_20261122').replace('PrithivGemma4E4BOwnBF16','Gemma4E4BMLX8OwnBF16').replace('PrithivGemma4E4B_own_original_BF16_','Gemma4E4BMLX8_own_original_BF16_')
 if '/adapters/' in rel:
  s=once(s,"return_tensors='pt',padding=True,add_special_tokens=False)","return_tensors='pt',padding=True)")
  s=once(s,"        inputs={k:v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in inputs.items()}\n", "        assert inputs['pixel_values'].dtype==torch.float32  # original HF wrapper passes F32 input; native model handles its original cast\n")
  s=once(s,"{'max_new_tokens':2048,'do_sample':False} and inputs['input_ids'].shape[1]+2048<=4096","{'max_new_tokens':2048,'do_sample':False,'temperature':0.0} and inputs['input_ids'].shape[1]+2048<=self.model.config.text_config.max_position_embeddings")
  s=s.replace('max_request_model_length=4096','max_request_model_length=self.model.config.text_config.max_position_embeddings')
  s=s.replace('for saved vLLM persistent cache','from the saved native HF generate cache')
  s=s.replace('Historical vLLM engine bytes were not saved.','Historical converted engine bytes were not saved; current own physical source is BF16, not native 8bit.')
 write(newrel,s)
write('universal_eval/model/gemma4_e4b_MLX8_own_20261122/__init__.py','"""Own registered BF16 Gemma4-E4B source and original saved HF deployment."""\n')

s=(BASE/'scripts/verify_prithiv_gemma4_e4b_own_original_BF16_native_v2_exact_own_report_path_20261116.py').read_text(encoding='utf-8').replace(OLD,MID)
s=once(s,"line['max_request_model_length']==4096","line['max_request_model_length']==own['text_config']['max_position_embeddings']")
s=once(s,"p['original_enable_thinking_false_from_saved_adapter']","p['original_template_default_false_without_override'] and p['original_no_enable_thinking_kwarg']")
start=s.index("    tasks=[Path(x) for x in profile")
end=s.index('    score=sum(scores)/count',start)
s=s[:start]+'''    tasks=[Path(x) for x in profile if Path(x).name=='task_time.csv' or re.fullmatch(r'task_time_[^.]+\\.csv',Path(x).name)];kernel_count=0
    assert tasks,'Actual complete native task_time/kernel_details correspondence is required'
    for t in tasks:
        with t.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        core_types={'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'}
        selected=[r for r in rows if r['kernel_type'] in core_types]
        assert selected and all(r['Device_id']=='4' and r['kernel_name']!='N/A' and Decimal(r['task_time(us)'])>0 for r in selected)
        actual_keys=[(r['Device_id'],r['stream_id'],r['task_id']) for r in selected]
        detail_path=check(t.with_name('kernel_details.csv'))
        with detail_path.open(encoding='utf-8-sig',newline='') as f:detail_rows=list(csv.DictReader(f))
        assert all(r['Accelerator Core'] in core_types and r['Device_id']=='4' and Decimal(r['Duration(us)'])>0 for r in detail_rows)
        detail_keys=[(r['Device_id'],r['Stream ID'],r['Task ID']) for r in detail_rows]
        assert len(actual_keys)==len(set(actual_keys))==len(detail_keys)==len(set(detail_keys)) and set(actual_keys)==set(detail_keys)
        kernel_count+=len(actual_keys)
    assert kernel_count>100,'Real original vision and prefill NPU kernels required'
''' + s[end:]
write('scripts/verify_gemma4_e4b_MLX8_own_original_Transformers_BF16_native_v1_20261122.py',s)
(V/'Gemma4MLX8_own_frontend_native_primary_paths_20261122.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{k:r[k] for k in ['path','SHA256']} for r in paths],ensure_ascii=False))

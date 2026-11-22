"""Own saved Transformers deployment, template and input interface. No learned forward."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import os, json, hashlib, ast, zipfile, xml.etree.ElementTree as E
from datetime import datetime, timezone
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
R=Path('/root/projects/comperror'); D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='unsloth/gemma-4-E4B-it-MLX-8bit'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration')/MID
H=B/'Gemma4E4BMLX8_own_saved_Transformers_template_generation_contract_no_NN_20261122'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def js(p): return json.loads(Path(p).read_bytes())
def put(n,x):
 p=H/n
 with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 import torch,yaml
 from PIL import Image
 from transformers import AutoConfig,AutoProcessor,AutoModelForImageTextToText,GenerationConfig
 from universal_eval.algorithm.inference import GreedyInference
 from batch_910b_support import workbook_rows
 torch.set_num_threads(4)
 protected=[D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx']
 before={str(p):sha(p) for p in protected}
 saved=R/'configs/Gemma4ForConditionalGeneration/unsloth--gemma-4-E4B-it-MLX-8bit.yaml'
 assert sha(saved)=='2fdbfd5d2e5a27e2e2470699e7b15e3297e098c05353e076a57e3339ca748c69'
 cfg=yaml.safe_load(saved.read_bytes());assert cfg['model']['name']==MID
 assert cfg['model']['adapter']=='universal_eval/model/adapters/transformers_vlm.py:TransformersVLM'
 assert cfg['model']['kwargs']['max_new_tokens']==2048 and cfg['model']['kwargs']['torch_dtype']=='bfloat16'
 assert cfg['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs=dict(extract_answer=True))
 original=R/'universal_eval/model/adapters/transformers_vlm.py'
 assert sha(original)=='507b9405d8ec6db48d008a7b4a29b23f448d6f04936545b8c0ce6b82c25336c3'
 source=original.read_text();ast.parse(source)
 config=AutoConfig.from_pretrained(str(W),local_files_only=True)
 processor=AutoProcessor.from_pretrained(str(W),local_files_only=True)
 with torch.device('meta'): model=AutoModelForImageTextToText.from_config(config)
 assert type(model).__name__=='Gemma4ForConditionalGeneration' and 'Gemma4Unified' not in type(model).__name__
 assert config.dtype==torch.bfloat16 and config.text_config.num_hidden_layers==42 and config.vision_config.num_hidden_layers==16
 data=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';row=json.loads(data.read_text().splitlines()[0])
 import re
 prompt=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',row['prompt']);prompt=re.sub(r'[ \t]+',' ',prompt);prompt=re.sub(r'\n{3,}','\n\n',prompt).strip();prompt=re.sub(r'<image\d+>\s*','',prompt)
 prompt=GreedyInference._add_answer_instruction(prompt)
 message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
 default=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False)
 false=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False,enable_thinking=False)
 true=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=False,enable_thinking=True)
 media=Path(row['media_path']);media=media if media.is_absolute() else data.parent/media
 image=Image.open(media).convert('RGB')
 native=processor(text=default,images=[image],return_tensors='pt',padding=True)
 noextra=processor(text=default,images=[image],return_tensors='pt',add_special_tokens=False)
 def describe(inputs):
  return {n:dict(shape=list(t.shape),dtype=str(t.dtype),SHA256=hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()) for n,t in inputs.items()}
 gen=GenerationConfig.from_pretrained(str(W),local_files_only=True)
 bind=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'
 rows,sheet,_=workbook_rows(book);entry,=[r for r in rows if r['model_id']==MID];assert entry['score'] is None
 with zipfile.ZipFile(book) as z:
  ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  serverD=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(entry['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 preserved=Path(bind['local_A100_reference_snapshot']);current=Path(bind['current_user_workbook_D_reference_snapshot'])
 assert sha(preserved)==bind['local_A100_reference_snapshot_sha256'] and sha(current)==bind['current_user_workbook_D_reference_snapshot_sha256']
 ownassets={p.name:dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.model','.jinja']}
 parent=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4ForConditionalGeneration/prithivMLmods/gemma-4-E4B-it-Uncensored-MAX')
 inputonly_compare={n:dict(own_SHA256=x['SHA256'],same_as_existing_architecture_input_asset=(parent/n).is_file() and sha(parent/n)==x['SHA256']) for n,x in ownassets.items() if n in ['processor_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json','config.json']}
 assert before=={str(p):sha(p) for p in protected}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,original_saved_config=str(saved),original_saved_config_SHA256=sha(saved),original_saved_weight_location=cfg['model']['model_path'],same_ID_registered_weight_location=str(W),old_converted_location_readable=Path(cfg['model']['model_path']).exists(),actual_native_meta_class=type(model).__name__,meta_tensors=len(model.state_dict()),historical_A100_runtime_claimed=False,learned_CPU_or_NPU_forward=False,original_template_passes_no_enable_thinking_kwarg=True,default_equals_false=default==false,default_equals_true=default==true,formatted_default=default,formatted_false=false,formatted_true=true,native_processor_inputs=describe(native),explicit_no_add_special_inputs=describe(noextra),native_default_and_no_extra_tokens_equal=torch.equal(native['input_ids'],noextra['input_ids']),original_generate_kwargs=dict(max_new_tokens=2048,do_sample=False,temperature=0.0),own_generation_config=gen.to_dict(),own_assets=ownassets,input_only_existing_architecture_asset_comparison=inputonly_compare,workbook_row=entry['row'],workbook_task=entry['task'],three_D=dict(server=serverD,local=js(current)['models'][MID]['A100_D'],preserved_original=js(preserved)['models'][MID]['A100_D']),canonical_SHA256=before)
 put('own_original_interface_contract.json',result);put('actual_completion.json',dict(actual_returncode=0,UTC=result['UTC'],no_learned_forward=True))
 print(json.dumps({k:result[k] for k in ['passed','UTC','model_id','actual_native_meta_class','meta_tensors','default_equals_false','default_equals_true','native_default_and_no_extra_tokens_equal','native_processor_inputs','three_D','input_only_existing_architecture_asset_comparison']},ensure_ascii=False))
if __name__=='__main__':
 H.mkdir(parents=True,exist_ok=False)
 try: main()
 except BaseException:
  import traceback
  put('actual_completion.json',dict(actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),no_learned_forward=True,error=traceback.format_exc()));raise

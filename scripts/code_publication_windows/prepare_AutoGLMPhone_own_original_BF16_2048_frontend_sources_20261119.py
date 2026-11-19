"""Prepare own AutoGLM input audit source; no learned inference or active source edits."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent
def sha(v):return hashlib.sha256(v).hexdigest()
src=(P/'prepare_huihui_glm46flash_own_original_BF16_2048_20261118_frontend200.py').read_text(encoding='utf8')
replacements={
 'huihui-ai/Huihui-GLM-4.6V-Flash-abliterated':'zai-org/AutoGLM-Phone-9B-Multilingual',
 'HuihuiGLM46Flash_own_original_BF16_2048_frontend200_meta_v1_20261118':'AutoGLMPhone9B_own_original_BF16_2048_frontend200_meta_v1_20261119',
 'configs/Glm4vForConditionalGeneration/huihui-ai--Huihui-GLM-4.6V-Flash-abliterated.yaml':'configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B-Multilingual.yaml',
 'b21abd87e78261d7d3621411d7602a54614b34168db395c6e6c020549e69bca7':'b93e0b08e370db8fe08d0435a2d6fad6ae0f9653db931a4383587327d9c7a982',
 "cfg['algorithm']['batch_size']==4":"cfg['algorithm']['batch_size']==10",
 '20585644008':'20585644112',
 "type(processor).__name__=='Glm46VProcessor' and type(ip).__name__=='Glm46VImageProcessor'":"type(processor).__name__=='Glm4vProcessor' and type(ip).__name__=='Glm4vImageProcessor'",
 "dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'":"dataset=Path(cfg['data']['dataset_path']);assert str(dataset)=='/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl' and sha(dataset)=='2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451'",
 "apply_template=False,filter_empty=False":"apply_template=cfg['data']['apply_template'],filter_empty=cfg['data']['filter_empty']",
 "assert '/nothink' in text and text.endswith('<think></think>\\n')":"assert text==processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False) and text.endswith('<|assistant|>\\n') and '/nothink' not in text and '<think>' not in text",
 'original_existing_adapter_enable_thinking_false_and2048_outer4_inner1_preserved':'original_existing_adapter_default_ignored_by_own_template_and2048_outer10_inner1_preserved',
 'outer_batch=4':'outer_batch=10',
 "own_whole_weight_bytes=20585644048":"own_whole_weight_bytes=20585644112",
}
for old,new in replacements.items():
 assert old in src,old
 src=src.replace(old,new)
src=src.replace("   out.write(json.dumps(value,ensure_ascii=False)+'\\n');rows.append(value);put('progress.json'", "   if i in [0,99,199]:np.savez_compressed(H/('own_complete_input_example_'+str(i)+'.npz'),**{n:t.detach().cpu().numpy() for n,t in actual.items() if torch.is_tensor(t)})\n   out.write(json.dumps(value,ensure_ascii=False)+'\\n');rows.append(value);put('progress.json'")
assert 'huihui' not in src.lower() and '20585644048' not in src
assert 'filter_empty=cfg' in src and 'outer_batch=10' in src
name='prepare_autoglm_phone9b_own_original_BF16_2048_20261119_frontend200.py'
raw=src.encode();ast.parse(raw);target=P/name;target.write_bytes(raw)
relative='scripts/'+name
data=dict(relative=relative,SHA256=sha(raw),base64=base64.b64encode(raw).decode())
payload="import sys\nsys.dont_write_bytecode=True\nimport ast,base64,hashlib,json,os,subprocess\nfrom pathlib import Path\nR=Path('/root/projects/comperror')\nv=json.loads("+repr(json.dumps(data))+ ")\np=R/v['relative'];raw=base64.b64decode(v['base64']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)\nsubprocess.run([sys.executable,'-B',str(p)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)\n"
(P/'run_stage_AutoGLMPhone9B_own_input200_20261119_remote.py').write_text(payload,encoding='utf8',newline='\n')
(P/'AutoGLMPhone9B_own_input200_source_scope_20261119.json').write_text(json.dumps(dict(source=relative,SHA256=sha(raw),own_dataset_preserved=True,own_GLM4v_processor_and_own_chat_template_preserved=True,own_704_BF16_and5_SF=True,own_outer10_max2048=True,no_learned_CPU_or_NPU=True),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(source=relative,SHA256=sha(raw),bytes=len(raw))))

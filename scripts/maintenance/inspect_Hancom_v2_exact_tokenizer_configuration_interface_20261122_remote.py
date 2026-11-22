"""Read the exact own tokenizer interface and current quiet state; no learned model."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
monitor=R/'scripts/A100_alignment_incremental_monitor_20261103.py'
assert hashlib.sha256(monitor.read_bytes()).hexdigest()=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
run=subprocess.run(['python3','-B',str(monitor)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes
from transformers.models.auto import tokenization_auto
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaLlamaForCausalLM/HancomInSpaceAI/HiLLaVA-llama_3_1-8b')
def record(p):
 raw=p.read_bytes();return dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),complete_value=json.loads(raw))
p=Path(tokenization_auto.__file__);lines=p.read_text().splitlines()
tf=R/'universal_eval/model/adapters/transformers_vlm.py';tflines=tf.read_text().splitlines()
sections=[]
for i,l in enumerate(tflines):
 if 'AutoTokenizer' in l or 'register(' in l or 'GenerationConfig' in l:
  sections.append(dict(line=i+1,text='\n'.join(f'{j+1}: {tflines[j]}' for j in range(max(0,i-8),min(len(tflines),i+12)))))
print(json.dumps(dict(monitor_returncode=run.returncode,monitor_stdout=run.stdout,monitor_stderr=run.stderr,lock_owners=runner_lock_owners(),processes=project_processes(),free_bytes=__import__('shutil').disk_usage(R).free,tokenizer_implementation=dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),lines700_780='\n'.join(f'{j+1}: {lines[j]}' for j in range(699,min(len(lines),780)))),own_assets=[record(W/n) for n in ['config.json','tokenizer_config.json','generation_config.json','special_tokens_map.json'] if (W/n).is_file()],original_TF_SHA256=hashlib.sha256(tf.read_bytes()).hexdigest(),original_TF_relevant_sections=sections),ensure_ascii=False))

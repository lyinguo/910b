import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast,difflib
P=Path(__file__).parent
packet=json.loads((P/'R_Gemma4E2B_existing_primary_source_contracts_own_next_candidate_20261117.json').read_text(encoding='utf8'))
origin='scripts/prepare_coder_gemma4_e2b_own_saved_HF_default_template_F32_frontend200_v1_20261116.py'
row=packet['sources']['/root/projects/comperror/'+origin];old=row['text'];assert hashlib.sha256(old.encode()).hexdigest()==row['SHA256']
new=old.replace('coder3101/gemma-4-E2B-it-qat-q4_0-unquantized-heretic','deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit').replace('CoderGemma4E2B_own_saved_HF_default_template_F32_pixel_patch_frontend200_v1_20261116','DeadbydawnGemma4E2B_own_saved_HF_default_template_F32_pixel_patch_frontend200_v1_20261117').replace('coder3101--gemma-4-E2B-it-qat-q4_0-unquantized-heretic.yaml','deadbydawn101--gemma-4-E2B-Heretic-Uncensored-mlx-4bit.yaml').replace("dict(device='cuda',max_new_tokens=2048,torch_dtype='auto')","dict(device='auto',max_new_tokens=2048,torch_dtype='auto')")
new=new.replace('1951','2011')
new=new.replace("native=model.state_dict();unused=set();missing=set(native)-set(headers);extra=set(headers)-set(native)","native=model.state_dict();unused={f'model.language_model.layers.{i}.self_attn.{n}' for i in range(15,35) for n in ['k_proj.weight','v_proj.weight','k_norm.weight']};missing=set(native)-set(headers);extra=set(headers)-set(native)")
new=new.replace('native_shared_unused_source_tensors=0','native_shared_unused_source_tensors=60')
new=new.replace('H.mkdir(parents=True,exist_ok=True)','H.mkdir(parents=True,exist_ok=False)')
assert new!=old and 'extra==unused' in new and 'original_enable_thinking_argument=None' in new
ast.parse(new)
rel='scripts/prepare_deadbydawn_gemma4_e2b_own_saved_HF_default_template_F32_frontend200_v1_20261117.py'
(P/'DeadbydawnGemma4E2B_own_frontend_source_reviewed_diff_20261117.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=origin,tofile=rel)),encoding='utf8')
raw=new.encode();h=hashlib.sha256(raw).hexdigest();(P/Path(rel).name).write_bytes(raw)
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
rel=__REL__;raw=base64.b64decode(__DATA__);h=__SHA__;assert hashlib.sha256(raw).hexdigest()==h;ast.parse(raw);p=R/rel
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
I=B/'DeadbydawnGemma4E2B_own_frontend200_no_NN_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:
 proc=subprocess.Popen(['python3','-B',str(p)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=proc.pid,source=str(p),source_SHA256=h,source_reviewed_from_own_saved_HF_deployment=True,learned_CPU_or_NPU_model_forward=False,no_original_runner_lock_taken=True,no_old_model_stage_repeated=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__REL__',repr(rel)).replace('__DATA__',repr(base64.b64encode(raw).decode())).replace('__SHA__',repr(h))
ast.parse(remote);(P/'launch_DeadbydawnGemma4E2B_own_original_F32_frontend200_metadata_only_20261117_remote.py').write_bytes(remote.encode())
print(json.dumps(dict(source=rel,SHA256=h,no_NN_in_deployment=True)))

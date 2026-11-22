"""Prepare the next checkpoint's own original inputs while the sole GLM NN runs."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
old=(V/'prepare_HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py').read_text(encoding='utf8')
assert 'assert exact' in old and 'no_learned_CPU_or_NPU_forward=True' in old
s=old.replace('HancomInSpaceAI/HiLLaVA-llama_3_1-8b','ytu-ce-cosmos/Turkish-LLaVA-v0.1').replace('HancomInSpaceAI--HiLLaVA-llama_3_1-8b','ytu-ce-cosmos--Turkish-LLaVA-v0.1').replace('HancomHiLlava31','TurkishLlava')
s=s.replace('624d81316384e7bff93ba2fd7024f3e7b47051198eb3deff2313aa400a83f84e','2a5e440def37d2d882d587d8077de0e2ea280de46d708213b966944ba30e88b7')
needle="raw['rope_scaling']==dict(factor=8.0,high_freq_factor=4.0,low_freq_factor=1.0,original_max_position_embeddings=8192,rope_type='llama3')"
assert needle in s;s=s.replace(needle,"raw['rope_scaling'] is None and raw['vocab_size']==128257 and raw['pad_token_id']==128256 and raw['eos_token_id']==128009")
s=s.replace('16709592504','16709608888').replace('row=338','row=689').replace('D338','D689').replace('current_row=338','current_row=689')
s=s.replace("assert all(Decimal(v)==Decimal('.715') for v in three.values())","assert all(Decimal(str(v))==Decimal(str(server_D)) for v in three.values())")
s=s.replace('own_checkpoint_embedding_and_head_rows=128256','own_checkpoint_embedding_and_head_rows=128257').replace('original_saved_resize_token_embeddings_required=image_id>=128256','original_saved_resize_token_embeddings_required=image_id>=128257')
s=s.replace('original_saved_legacy_reconstruction_omits_raw_llama31_rope_scaling=True','original_saved_legacy_reconstruction_and_own_null_rope_scaling_same=True')
s=s.replace('Own checkpoint/frontend/original processor contract; no learned forward or dispatch.','Own Turkish checkpoint/direct original processor contract, complete frontend; no learned forward or dispatch.')
assert all(x not in s for x in ['Hancom','16709592504','row=338','D338',"Decimal('.715')"]);ast.parse(s)
source='scripts/prepare_TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py';(V/Path(source).name).write_text(s,encoding='utf8')
stage={source:s.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n;ast.parse(b)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nB=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'TurkishLlava_own_original_frontend200_no_NN_actual_launcher_v1_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr(source)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr(source)+",source_SHA256="+repr(sha(s.encode()))+",all_primary_SHA256={n:r['SHA256'] for n,r in P.items()},no_learned_CPU_NPU_forward_or_original_runner_EX=True,no_current_GLM_source_job_deadline_signal_or_canonical_mutation=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_TurkishLlava_own_original_frontend200_no_NN_v1_20261122_remote.py').write_text(carrier,encoding='utf8');print(json.dumps(dict(source=source,SHA256=sha(s.encode()),own_original_BF16_storage_auto_FP16_inputs_only=True,not_ready_or_NPU3=True)))

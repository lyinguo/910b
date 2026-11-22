"""Fix only the own source header guard after an exact zero-forward v1 failure."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
model1='universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v1_20261122.py'
prep1='scripts/prepare_hancom_hillava31_own_original_Transformers_FP16_2048_true3_v1_20261122.py'
cont1='scripts/continue_hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.py'
queue1='configs/queues/hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.json'
model2=model1.replace('native_v1_','native_v2_own_BF16_guard_');prep2=prep1.replace('true3_v1_','true3_v2_own_BF16_guard_');cont2=cont1.replace('serial_v1_','serial_v2_own_BF16_guard_');queue2=queue1.replace('serial_v1_','serial_v2_own_BF16_guard_')
old=(V/Path(model1).name).read_text(encoding='utf8').encode('utf8');assert sha(old)=='948208b087c6fcaaed829d627d5de60c17b077de85d0b0ee5fe98dc2026ce17d'
a='source.dtype in [torch.float16,torch.float32]';b='source.dtype == torch.bfloat16';assert old.decode().count(a)==1
new=old.decode().replace(a,b);assert new.replace(b,a)==old.decode();ast.parse(new)
prep=(V/Path(prep1).name).read_text(encoding='utf8');assert sha(prep.encode())=='79d0d11bf7e2855066740a00cda60b91f8695bd98a681d6bf7e41a75e2766cc2'
for p1,p2 in [(model1,model2),(prep1,prep2),(cont1,cont2),(queue1,queue2)]:prep=prep.replace(p1,p2)
prep=prep.replace('source_readiness_v1_20261122','source_readiness_v2_own_BF16_guard_20261122').replace('true3_v1_real_FLOCK_20261122','true3_v2_own_BF16_guard_real_FLOCK_20261122').replace('native-v1-20261122','native-v2-own-BF16-guard-20261122').replace('native_v1_20261122.json','native_v2_own_BF16_guard_20261122.json')
marker=' H.mkdir(exist_ok=False)\n';assert prep.count(marker)==1
audit=""" H.mkdir(exist_ok=False)
 old_job=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122';old_done=js(old_job/'completion.json');old_console=(old_job/'fresh3/console.log').read_text()
 assert old_done['actual_returncode']==1 and 'source.dtype in [torch.float16,torch.float32]' in old_console and 'AssertionError' in old_console
 assert not any((R/'workspace/diagnostics/model_runs').glob('HancomHiLlava31_own_original_FP16_2048_*/generation_progress.jsonl'))
 old_source=R/OLD_MODEL_PATH;assert sha(old_source)=='948208b087c6fcaaed829d627d5de60c17b077de85d0b0ee5fe98dc2026ce17d'
 old_text=old_source.read_text();new_text=(R/MODEL).read_text();assert new_text.replace('source.dtype == torch.bfloat16','source.dtype in [torch.float16,torch.float32]')==old_text
 create(H/'exact_zero_forward_source_dtype_guard_recovery.json',dict(passed=True,old_job=str(old_job),old_completion_SHA256=sha(old_job/'completion.json'),old_console_SHA256=sha(old_job/'fresh3/console.log'),old_source_SHA256=sha(old_source),only_source_guard_F16_F32_to_exact_own_BF16=True,original_FP16_cast_all_math_budget_input_EOS_verifier_unchanged=True,all_original_failure_evidence_retained=True))
""".replace('OLD_MODEL_PATH',repr(model1))
prep=prep.replace(marker,audit)
prep=prep.replace("static={str(p):sha(p) for p in [dataset,afs,", "static={str(p):sha(p) for p in [H/'exact_zero_forward_source_dtype_guard_recovery.json',old_job/'completion.json',old_job/'fresh3/console.log',dataset,afs,")
prep=prep.replace("rels=[MODEL,GRAPH,ALGO,NF,SELF,CONTROLLER,QUEUE,", "rels=[MODEL,GRAPH,ALGO,NF,SELF,CONTROLLER,QUEUE,"+repr(model1)+",")
cont=(V/Path(cont1).name).read_text(encoding='utf8');assert sha(cont.encode())=='2262775096aa0ce35217ee818a314b9dc39947d672238439ae02756fe6d9910c'
for p1,p2 in [(prep1,prep2),(cont1,cont2),(queue1,queue2)]:cont=cont.replace(p1,p2)
# Private own paths only. The immutable previous More path remains exactly original.
cont=cont.replace('HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122','HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_own_BF16_guard_20261122')
cont=cont.replace('source_readiness_v1_20261122','source_readiness_v2_own_BF16_guard_20261122').replace('true3_v1_real_FLOCK_20261122','true3_v2_own_BF16_guard_real_FLOCK_20261122')
# Restore exact previous More job names if the broad suffix touches references.
cont=cont.replace('LlavaMore_own_original_FP16_discriminative_letter_true3_v2_own_BF16_guard_real_FLOCK_20261122','LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122')
cont=cont.replace('first200_v1_20261122','first200_v2_own_BF16_guard_20261122').replace('first200-v1-20261122','first200-v2-own-BF16-guard-20261122').replace('fixed200_serial_v1_real_FLOCK_20261122','fixed200_serial_v2_own_BF16_guard_real_FLOCK_20261122')
for n,s in [(model2,new),(prep2,prep),(cont2,cont)]:ast.parse(s);(V/Path(n).name).write_text(s,encoding='utf8')
qold=json.loads((V/'HancomHiLlava31_own_native_fullNF_serial_all_primary_v1_20261122.json').read_bytes())
base=json.loads((V/'hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.json').read_bytes()) if (V/'hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.json').exists() else dict(model_id='HancomInSpaceAI/HiLLaVA-llama_3_1-8b',max_concurrent_models=1,exact_source_SHA256={n:h for n,h in qold.items() if n.endswith('.py') and not n.startswith('scripts/code_publication_windows/')})
sources=base['exact_source_SHA256'];sources={n:h for n,h in sources.items() if n not in [model1,prep1,cont1]};sources.update({model1:sha(old),model2:sha(new.encode()),prep2:sha(prep.encode()),cont2:sha(cont.encode())});base.update(exact_source_SHA256=sources,private_exact_BF16_source_guard_recovery_after_zero_NN=True,no_math_or_requested_FP16_cast_change=True)
stage={model2:new.encode(),prep2:prep.encode(),cont2:cont.encode(),queue2:json.dumps(base,ensure_ascii=False,indent=2).encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes()}
manifest={n:sha(v) for n,v in stage.items()};(V/'HancomHiLlava31_private_exact_BF16_guard_v2_primary_paths_20261122.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
payload={n:dict(SHA256=sha(v),base64=base64.b64encode(v).decode()) for n,v in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n\n if p.suffix=='.py':ast.parse(b)\n p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nD=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';old=B/'HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122/actual_completion.json';assert json.loads(old.read_bytes())['actual_returncode']==1\nsys.path[:0]=[str(R),str(R/'scripts')];from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes\nassert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]\nH=B/'HancomHiLlava31_private_exact_BF16_guard_v2_actual_launcher_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr(cont2)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr(cont2)+",source_SHA256="+repr(manifest[cont2])+",serial_max1=True,old_zero_NN_failure_evidence_unchanged=True,only_exact_BF16_source_guard_recovery=True,no_requested_FP16_math_budget_or_EOS_change=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_HancomHiLlava31_private_exact_BF16_guard_v2_20261122_remote.py').write_text(carrier,encoding='utf8');print(json.dumps(dict(primary_paths=manifest,only_model_diff=a+' -> '+b)))

"""Restore only the original saved registration in new immutable Hancom sources."""
from pathlib import Path
import ast,base64,difflib,hashlib,json
V=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
model2='universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v2_own_BF16_guard_20261122.py'
prep2='scripts/prepare_hancom_hillava31_own_original_Transformers_FP16_2048_true3_v2_own_BF16_guard_20261122.py'
cont2='scripts/continue_hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_own_BF16_guard_20261122.py'
queue2='configs/queues/hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_own_BF16_guard_20261122.json'
nf1='scripts/verify_hancom_hillava31_own_original_Transformers_FP16_2048_full_values_v1_20261122.py'
model3=model2.replace('v2_own_BF16_guard','v3_original_registration');prep3=prep2.replace('v2_own_BF16_guard','v3_original_registration');cont3=cont2.replace('v2_own_BF16_guard','v3_original_registration');queue3=queue2.replace('v2_own_BF16_guard','v3_original_registration');nf2=nf1.replace('full_values_v1','full_values_v2_original_registration')
registration='from universal_eval.model.adapters.transformers_vlm import TransformersVLM as _OwnSavedRegistration\n'
old_model=(V/Path(model2).name).read_text(encoding='utf8');assert sha(old_model.encode())=='f67dbaaadad917b612daeadf2826c44d81d23712d312c00e250e6af7295f7d1f'
model=old_model.replace('from transformers.cache_utils import DynamicLayer\n',registration+'from transformers.cache_utils import DynamicLayer\n');assert model.replace(registration,'')==old_model
old_nf=(V/Path(nf1).name).read_text(encoding='utf8');assert sha(old_nf.encode())=='e60a28b60f00225f3cc28433c7ff60f8ffb92384cd7bfb8b29137a3e3bcc1177'
needle='from egm8b_original_independent_cpu_910b import sha,fp,tr\n';assert old_nf.count(needle)==1
nf=old_nf.replace(needle,registration+needle);assert nf.replace(registration,'')==old_nf
def without_registration(s):
 tree=ast.parse(s);tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='universal_eval.model.adapters.transformers_vlm')];return ast.dump(tree,include_attributes=False)
assert without_registration(model)==without_registration(old_model) and without_registration(nf)==without_registration(old_nf)
prep=(V/Path(prep2).name).read_text(encoding='utf8');assert sha(prep.encode())=='ab19c1797b3b9183aa10b730eb48abac30b3441d97286c225ebb5d0cae7cdda0'
for a,b in [(model2,model3),(prep2,prep3),(cont2,cont3),(queue2,queue3),(nf1,nf2)]:prep=prep.replace(a,b)
prep=prep.replace('v2_own_BF16_guard','v3_original_registration').replace('v2-own-BF16-guard','v3-original-registration')
start=prep.index(" old_job=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122'")
end=prep.index(" ready=js(FRONT/'readiness.json')",start)
audit=""" old_job=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v2_own_BF16_guard_real_FLOCK_20261122';old_done=js(old_job/'completion.json');old_console=(old_job/'fresh3/console.log').read_text()
 assert old_done['actual_returncode']==1 and "self._tokenizer=AutoTokenizer.from_pretrained" in old_console and "'PreTrainedConfig' object has no attribute 'max_position_embeddings'" in old_console
 assert not any((R/'workspace/diagnostics/model_runs').glob('HancomHiLlava31_own_original_FP16_2048_*/generation_progress.jsonl'))
 old_source=R/OLD_MODEL;assert sha(old_source)=='f67dbaaadad917b612daeadf2826c44d81d23712d312c00e250e6af7295f7d1f'
 old_nf=R/OLD_NF;assert sha(old_nf)=='e60a28b60f00225f3cc28433c7ff60f8ffb92384cd7bfb8b29137a3e3bcc1177'
 original_registration=REGISTRATION
 assert (R/MODEL).read_text().replace(original_registration,'')==old_source.read_text() and (R/NF).read_text().replace(original_registration,'')==old_nf.read_text()
 diagnostic=B/'HancomHiLlava31_original_registered_tokenizer_generation_interface_no_NN_v1_20261122';review=js(diagnostic/'review.json')
 assert sha(diagnostic/'review.json')=='b697ff522a04ee17896b1cac7a864ece7ac2edc124f0042730c9a320e86d26c2' and review['passed'] and review['actual_NEW_model_forwards']==0 and js(diagnostic/'actual_completion.json')['actual_returncode']==0
 from transformers import AutoTokenizer,GenerationConfig
 own_path=Path(js(FRONT/'readiness.json')['own_weight_path']);tokenizer=AutoTokenizer.from_pretrained(own_path,local_files_only=True);generation=GenerationConfig.from_pretrained(own_path,local_files_only=True)
 assert len(tokenizer)==128256 and generation.eos_token_id==[128001,128008,128009] and tokenizer.chat_template==js(own_path/'tokenizer_config.json')['chat_template']
 create(H/'exact_zero_forward_original_registration_recovery.json',dict(passed=True,old_job=str(old_job),old_completion_SHA256=sha(old_job/'completion.json'),old_console_SHA256=sha(old_job/'fresh3/console.log'),old_model_SHA256=sha(old_source),old_verifier_SHA256=sha(old_nf),only_original_saved_registration_import_in_model_and_verifier=True,all_model_math_observers_FP16_cast_budget_EOS_thresholds_unchanged=True,diagnostic_review_SHA256=sha(diagnostic/'review.json'),original_v1_and_v2_failure_evidence_retained=True,own_existing_frontend200_and_constructor_reused_without_replay=True))
""".replace('OLD_MODEL',repr(model2)).replace('OLD_NF',repr(nf1)).replace('REGISTRATION',repr(registration))
prep=prep[:start]+audit+prep[end:]
prep=prep.replace("H/'exact_zero_forward_source_dtype_guard_recovery.json'","H/'exact_zero_forward_original_registration_recovery.json',diagnostic/'review.json',diagnostic/'actual_completion.json',B/'HancomHiLlava31_own_original_Transformers_FP16_2048_source_readiness_v2_own_BF16_guard_20261122/exact_zero_forward_source_dtype_guard_recovery.json'")
prep=prep.replace('rels=[MODEL,GRAPH,ALGO,NF,SELF,CONTROLLER,QUEUE,','rels=[MODEL,GRAPH,ALGO,NF,SELF,CONTROLLER,QUEUE,'+repr(model2)+','+repr(nf1)+",'scripts/diagnose_HancomHiLlava31_original_registered_tokenizer_generation_interface_no_NN_v1_20261122.py',")
cont=(V/Path(cont2).name).read_text(encoding='utf8');assert sha(cont.encode())=='9884531d39d6f1587d555085e516b036a09879695985723bf29fd9d24990c67b'
cont=cont.replace('v2_own_BF16_guard','v3_original_registration').replace('v2-own-BF16-guard','v3-original-registration')
for name,s in [(model3,model),(nf2,nf),(prep3,prep),(cont3,cont)]:ast.parse(s);(V/Path(name).name).write_bytes(s.encode())
carrier_tree=ast.parse((V/'stage_launch_HancomHiLlava31_private_exact_BF16_guard_v2_20261122_remote.py').read_text(encoding='utf8'))
payload_node=next(n.value for n in ast.walk(carrier_tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='P' for t in n.targets))
old_payload=json.loads(base64.b64decode(ast.literal_eval(payload_node.args[0].args[0])))
qbytes=base64.b64decode(old_payload[queue2]['base64']);assert sha(qbytes)=='d16e7678de81364919936403da59ca3a2ee4f46a6a91ba0044988c0e73801b24';q=json.loads(qbytes)
sources={n:h for n,h in q['exact_source_SHA256'].items() if n not in [model2,prep2,cont2,nf1]}
sources.update({model2:sha(old_model.encode()),nf1:sha(old_nf.encode()),model3:sha(model.encode()),nf2:sha(nf.encode()),prep3:sha(prep.encode()),cont3:sha(cont.encode()),'scripts/diagnose_HancomHiLlava31_original_registered_tokenizer_generation_interface_no_NN_v1_20261122.py':'fc2070dffdb40634ff15a585334aaddba86227bdc77f44b1bbb666ad304ea4b8'})
q.update(exact_source_SHA256=sources,private_original_saved_registration_recovery_after_zero_NN=True,no_model_math_or_requested_FP16_cast_budget_EOS_threshold_change=True)
stage={model3:model.encode(),nf2:nf.encode(),prep3:prep.encode(),cont3:cont.encode(),queue3:json.dumps(q,ensure_ascii=False,indent=2).encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes()}
manifest={n:sha(b) for n,b in stage.items()};(V/'HancomHiLlava31_private_original_registration_v3_primary_paths_20261122.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
diff=''.join(difflib.unified_diff(old_model.splitlines(True),model.splitlines(True),fromfile=model2,tofile=model3))+''.join(difflib.unified_diff(old_nf.splitlines(True),nf.splitlines(True),fromfile=nf1,tofile=nf2));(V/'HancomHiLlava31_private_v3_exact_registration_complete_runtime_and_NF.diff').write_text(diff,encoding='utf8')
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n\n if p.suffix=='.py':ast.parse(b)\n p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nD=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';old=B/'HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_own_BF16_guard_20261122/actual_completion.json';assert json.loads(old.read_bytes())['actual_returncode']==1\nsys.path[:0]=[str(R),str(R/'scripts')];from A100_alignment_incremental_monitor_20261103 import runner_lock_owners,project_processes\nassert not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]\nH=B/'HancomHiLlava31_private_original_registration_v3_actual_launcher_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr(cont3)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr(cont3)+",source_SHA256="+repr(manifest[cont3])+",serial_max1=True,all_old_zero_NN_failure_evidence_unchanged=True,only_original_saved_registration_recovery=True,no_requested_FP16_math_budget_EOS_threshold_change=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_HancomHiLlava31_private_original_registration_v3_20261122_remote.py').write_text(carrier,encoding='utf8');print(json.dumps(dict(primary_paths=manifest,complete_model_and_verifier_diff=diff),ensure_ascii=False))

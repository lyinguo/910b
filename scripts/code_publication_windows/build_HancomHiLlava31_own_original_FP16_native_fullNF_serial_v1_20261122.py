"""Adapt only proven architecture code to own BF16-storage original FP16 protocol."""
from pathlib import Path
import ast, base64, hashlib, json
V = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
prefix = 'HancomHiLlava31'
replacements = [
 ('xtuner/llava-llama-3-8b-v1_1-hf','HancomInSpaceAI/HiLLaVA-llama_3_1-8b'),
 ('xtuner--llava-llama-3-8b-v1_1-hf','HancomInSpaceAI--HiLLaVA-llama_3_1-8b'),
 ('xtuner_llava3','hancom_hillava31'),('xtuner_v1','hancom_hillava31_v1'),('XtunerLlava3','HancomHiLlava31'),('OwnXtuner','OwnHancomHiLlava31'),('Own Xtuner','Own HancomHiLlava31'),
 ('native_v2_IModel_20261122','native_v1_20261122'),
 ('source_readiness_v2_IModel_20261122','source_readiness_v1_20261122'),
 ('true3_v2_IModel_real_FLOCK_20261122','true3_v1_real_FLOCK_20261122'),
 ('true3_v2_IModel_20261122','true3_v1_20261122'),
 ('serial_v2_IModel_20261122','serial_v1_20261122'),
 ('serial_v2_IModel_real_FLOCK_20261122','serial_v1_real_FLOCK_20261122'),
 ('first200_v2_IModel_20261122','first200_v1_20261122'),
 ('native-v2-IModel-20261122','native-v1-20261122'),
 ('first200-v2-IModel-20261122','first200-v1-20261122'),
 ('own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_original_tokenizer_fallback_20261122','own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122'),
 ('own_original_FP16_actual_CPU_loader_resize_remap_no_forward_v3_full_loading_info_schema_20261122','own_original_auto_FP16_actual_CPU_loader_resize_remap_no_forward_v1_20261122'),
 ('inspect_hancom_hillava31_own_original_FP16_loader_resize_remap_no_forward_v3_full_loading_info_schema_20261122.py','inspect_HancomHiLlava31_own_original_auto_FP16_loader_resize_remap_no_forward_v1_20261122.py'),
 ('prepare_hancom_hillava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py','prepare_HancomHiLlava31_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v1_20261122.py'),
]
def own(s):
 for a,b in replacements:s=s.replace(a,b)
 return s
model_old=(V/'xtuner_llava3_own_original_Transformers_FP16_2048_native_v2_IModel_20261122.py').read_bytes()
nf_old=(V/'verify_xtuner_llava3_own_original_Transformers_FP16_2048_full_values_v1_20261122.py').read_bytes()
prep_old=(V/'prepare_xtuner_llava3_own_original_Transformers_FP16_2048_true3_v2_IModel_20261122.py').read_bytes()
controller_old=(V/'continue_xtuner_llava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122.py').read_bytes()
assert sha(model_old)=='1c168f5aabe5e0718e176ee6bda98ba9b2e49337964df7908b306c1a0110e8d1'
assert sha(nf_old)=='34b6bb348272d9966f58f4e5acbbb9b878a607d65c0b653c34065c28d851a4b8'
assert sha(prep_old)=='f0ab2ab7212f92415d0261e2e45b593a192aa67a1b23e07577b8a1723299a517'
assert sha(controller_old)=='e32c627d0013571c10d8a662b2abbb69fa1ec49255f3bb5d46571b869cae523d'
graph=own((V/'xtuner_llava3_own_saved_Transformers_graph_20261122.py').read_text(encoding='utf8'))
algo=own((V/'xtuner_llava3_own_original_FP16_2048_MC_20261122.py').read_text(encoding='utf8'))
model=own(model_old.decode())
model=model.replace('universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118','universal_eval.utils.complete_standard_LZMA_NPZ_parallel_values_20261122')
model=model.replace('Own682F16+4F32 physical source', 'Own686BF16 physical source').replace("source_dtype_counts={'F16':682,'F32':4}","source_dtype_counts={'BF16':686}")
model=model.replace('model.generation_config.eos_token_id==128001','model.generation_config.eos_token_id==[128001,128008,128009]').replace('gen.eos_token_id==128001','gen.eos_token_id==[128001,128008,128009]')
model=model.replace('if t==128001','if t in [128001,128008,128009]').replace('/2048/EOS128001','/2048/ownEOS128001-128008-128009')
nf=own(nf_old.decode()).replace("{'F16':682,'F32':4}","{'BF16':686}").replace("{'torch.float16':682,'torch.float32':4}","{'torch.bfloat16':686}")
nf=nf.replace("generation['eos_token_id']==128001","generation['eos_token_id']==[128001,128008,128009]").replace('if t in [128001]','if t in [128001,128008,128009]')
cleanup="    if json.loads((wp/'tokenizer_config.json').read_bytes()).get('clean_up_tokenization_spaces',False):\n        for a,b in [(' .','.'),(' ?','?'),(' !','!'),(' ,',','),(\" ' \",\"'\"),(\" n't\",\"n't\"),(\" 'm\",\"'m\"),(\" 's\",\"'s\"),(\" 've\",\"'ve\"),(\" 're\",\"'re\")]:raw=raw.replace(a,b)\n"
assert cleanup in nf
nf=nf.replace(cleanup,"    # Own installed HF5 TokenizersBackend ignores cleanup; independently require its exact Rust output.\n    from transformers import AutoTokenizer\n    own_tokenizer=AutoTokenizer.from_pretrained(wp,local_files_only=True,trust_remote_code=True)\n    assert raw.strip()==own_tokenizer.decode(e['decoded_token_ids'],skip_special_tokens=True).strip()\n")
prep=own(prep_old.decode())
a=prep.index(' assert not runner_lock_owners()');b=prep.index(' import inspect\n',a)
prep=prep[:a]+prep[b:]
a=prep.index(' old_tree=');b=prep.index(" ready=js(FRONT/'readiness.json')",a)
prep=prep[:a]+" H.mkdir(exist_ok=False)\n"+prep[b:]
prep=prep.replace("ready['own_frontend200_SHA256']=='a3138288a36ca9d973f1d236042c833c8e2f5bbd00a21d3edc1e8c00014c18ef'","ready['own_frontend200_SHA256']=='9745e8b0845d5ef250c64d3e2491160ef1a9bc276bd82507e3cd585c9d02a5be'")
prep=prep.replace("=='2b87704d2041de0f677d233025f36fa0b29a9c64a98842452ffc5cc4ca82abc3'",'')
prep=prep.replace("=='a3fcb4ce473f725470e237b0fae9b8be2f028d0182b02132b46a46c0ac7fd7e3'",'')
prep=prep.replace("ctor['original_generation_config']['eos_token_id']==128001","ctor['original_generation_config']['eos_token_id']==[128001,128008,128009]")
prep=prep.replace('dict(row=686,model_id=MID','dict(row=338,model_id=MID').replace('max_position_embeddings==8192','max_position_embeddings==131072').replace('score=.71,','score=.715,')
prep=prep.replace(",H/'zero_forward_v1_IModel_recovery_review.json',old/'actual_completion.json',old_job/'fresh3/console.log'",'')
prep=prep.replace(",'universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v1_20261122.py'",'')
prep=prep.replace("'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py'","'universal_eval/utils/complete_standard_LZMA_NPZ_parallel_values_20261122.py','scripts/tests/test_complete_standard_LZMA_NPZ_parallel_values_20261122.py'")
prep=prep.replace('Own2048/outer10/Greedy/EOS128001','Own2048/outer10/Greedy/EOS[128001,128008,128009]').replace('Own686 physical sources cast','Own686 BF16 physical sources cast')
prep=prep.replace("ready_for_unique_original_runner_true3=True","ready_for_unique_original_runner_true3=True,ready_not_NPU_execution=True,own_original_system_date_26Jul2024_and_BF16_to_original_FP16_preserved=True")
controller=own(controller_old.decode())
controller=controller.replace('Serial own Xtuner IModel repair genuine3/full NF, complete shape capacity then new200.','Own Hancom original serial genuine3/fullNF after More actual release, no extra small3 D gate.')
marker="    with (H/'readiness_stdout.log').open('xb') as out"
pos=controller.index(marker)
controller=controller[:pos]+"""    chain=B/'HancomHiLlava31_own_frontend200_then_original_CPU_constructor_no_NN_v1_20261122'
    while not (chain/'actual_completion.json').is_file():
        unchanged({str(R/n):h for n,h in queue['exact_source_SHA256'].items()})
        put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_own_frontend_CPU_constructor_actual_exit_no_NN',no_original_runner_lock_held=True));time.sleep(20)
    assert js(chain/'actual_completion.json')['actual_returncode']==0
"""+controller[pos:]
controller=controller.replace("unchanged({str(R/n):h for n,h in queue['exact_source_SHA256'].items()});assert quiet()","unchanged({str(R/n):h for n,h in queue['exact_source_SHA256'].items()})")
marker="    unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert quiet() and batch.device_idle(job['spec'])"
pos=controller.index(marker)
wait="""    previous=B/'LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122'
    while True:
        done_path=previous/'actual_completion.json';done_previous=js(done_path) if done_path.is_file() else None
        qualified200=False
        if done_previous and done_previous.get('phase')=='NEWfresh3_fixed200_natural_actual_completion':
            completed=done_previous.get('completion',{})
            for item in completed.get('accepted',[]):
                if item.get('sample_count')==200:
                    p200=js(item['proof']);qualified200=bool(p200.get('passed') and p200.get('original_protocol_final_qualification_complete') and p200.get('formal_alignment_within_5pct'))
        if done_previous and quiet() and not qualified200:break
        put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_previous_More_actual_NN_NF_controller_EX_release_no_NN',previous_actual_completion_exists=bool(done_previous),previous_qualified200_publication_must_precede_next_NN=qualified200,own_source_ready=True,no_original_runner_lock_held=True,max_concurrent_models=1));time.sleep(20)
"""
controller=controller[:pos]+wait+controller[pos:]
controller=controller.replace("p2['static_evidence_SHA256']={accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')}","p2['static_evidence_SHA256'].update({accepted['proof']:accepted['proof_sha256'],str(H/'own_first200_measurement_approval.json'):sha(H/'own_first200_measurement_approval.json')})")
stage={
 'universal_eval/model/adapters/hancom_hillava31_own_original_Transformers_FP16_2048_native_v1_20261122.py':model,
 'universal_eval/model/hancom_hillava31_own_saved_Transformers_graph_20261122.py':graph,
 'universal_eval/algorithm/hancom_hillava31_own_original_FP16_2048_MC_20261122.py':algo,
 'scripts/verify_hancom_hillava31_own_original_Transformers_FP16_2048_full_values_v1_20261122.py':nf,
 'scripts/prepare_hancom_hillava31_own_original_Transformers_FP16_2048_true3_v1_20261122.py':prep,
 'scripts/continue_hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.py':controller,
}
for n,s in stage.items():
 # This exact name belongs to the immutable previous More controller, not this model.
 own_text=s.replace('LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122','')
 assert 'xtuner' not in own_text.lower(), n
 ast.parse(s);(V/Path(n).name).write_text(s,encoding='utf8')
qpath='configs/queues/hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.json'
queue=dict(model_id='HancomInSpaceAI/HiLLaVA-llama_3_1-8b',max_concurrent_models=1,own_original_BF16_storage_auto_nonCPU_FP16=True,no_original_math_or_generation_protocol_change=True,exact_source_SHA256={n:sha(s.encode()) for n,s in stage.items()},wait_actual_own_CPU_then_source_prepare_then_previous_More_actual_release=True,no_G_or_canonical_write=True)
stage_bytes={n:s.encode() for n,s in stage.items()}
stage_bytes[qpath]=json.dumps(queue,ensure_ascii=False,indent=2).encode()
stage_bytes['scripts/code_publication_windows/'+Path(__file__).name]=Path(__file__).read_bytes()
manifest={n:sha(b) for n,b in stage_bytes.items()}
(V/'HancomHiLlava31_own_native_fullNF_serial_all_primary_v1_20261122.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage_bytes.items()}
cont='scripts/continue_hancom_hillava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122.py'
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];p=R/n\n if p.suffix=='.py':ast.parse(b)\n p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nD=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'HancomHiLlava31_own_native_serial_after_More_v1_actual_launcher_20261122';H.mkdir(exist_ok=False)\nwith (H/'stdout.json').open('x') as out,(H/'stderr.log').open('x') as err:p=subprocess.Popen(['python3','-B',str(R/"+repr(cont)+")],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)\nx=dict(PID=p.pid,source="+repr(cont)+",source_SHA256="+repr(manifest[cont])+",serial_max1=True,wait_own_CPU_then_More_actual_release=True,no_current_NN_source_or_job_mutation=True,no_G_write=True);(H/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(x,ensure_ascii=False))\n"
ast.parse(carrier);(V/'stage_launch_HancomHiLlava31_own_native_after_More_serial_v1_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(primary_paths=manifest,no_current_More_mutation=True,own_expected_frontend_SHA256='9745e8b0845d5ef250c64d3e2491160ef1a9bc276bd82507e3cd585c9d02a5be')))

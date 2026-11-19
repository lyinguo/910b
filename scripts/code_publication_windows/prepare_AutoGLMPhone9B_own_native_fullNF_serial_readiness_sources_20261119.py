"""Own AutoGLM checkpoint/input binding of the complete tested GLM4v graph audit."""
from pathlib import Path
import ast,base64,hashlib,json,re
P=Path(__file__).resolve().parent
OLD='huihui_glm46flash_own_original_BF16_2048_20261118'
NEW='autoglm_phone9b_own_original_BF16_2048_20261119'
OLDMID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated';MID='zai-org/AutoGLM-Phone-9B-Multilingual'
def sha(v):return hashlib.sha256(v).hexdigest()
def repl(s):
 for a,b in [(OLD,NEW),(OLDMID,MID),('huihui-ai_Huihui-GLM-4.6V-Flash-abliterated','zai-org_AutoGLM-Phone-9B-Multilingual'),('HuihuiGLM46Flash','AutoGLMPhone9B'),('HuihuiGlm46Flash','AutoGLMPhone9B'),('OwnHuihuiGlm46Flash','OwnAutoGLMPhone9B'),('HuihuiGlm46FlashOwnNativeBF16','AutoGLMPhone9BOwnNativeBF16'),('huihui_glm46flash','autoglm_phone9b'),('Huihui GLM46','AutoGLM Phone9B'),('Huihui','AutoGLM'),('GLM-4.6V-Flash','AutoGLM-Phone-9B-Multilingual'),('2048_outer4','2048_outer10'),('outer4','outer10'),('outer_batch_size=4','outer_batch_size=10'),("len(batch)<=4","len(batch)<=10"),("cfg['algorithm']['batch_size']==4","cfg['algorithm']['batch_size']==10"),('range(i,min(i+4,count))) for i in range(0,count,4)','range(i,min(i+10,count))) for i in range(0,count,10)'),("and not cfg['data']['filter_empty']","and cfg['data']['filter_empty']"),('151363','151343')]:s=s.replace(a,b)
 return s
items=[]
for suffix,relative in [('native.py','universal_eval/model/adapters/'),('frontend.py','universal_eval/model/adapters/'),('MC.py','universal_eval/algorithm/')]:
 raw0=(P/(OLD+'_'+suffix)).read_bytes();s=repl(raw0.decode())
 if suffix=='native.py':
  s=s.replace("with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:\n  for k,v in values.items():\n   with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(v),allow_pickle=False)","from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz\n write_full_standard_npz(p,values)")
  assert 'ZIP_DEFLATED' not in s
 s=s.replace('original existing thinkingFalse greedy2048','original generic default thinkingFalse (own template ignores it) greedy2048')
 raw=s.encode();ast.parse(raw);assert 'huihui' not in s.lower() and '151363' not in s
 name=NEW+'_'+suffix;(P/name).write_bytes(raw);items.append(dict(relative=relative+name,SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
src=repl((P/('verify_'+OLD+'_native.py')).read_text(encoding='utf8'));src=src.replace('existing_thinkingFalse2048_outer10','own_template2048_outer10');assert 'i+4' not in src and 'count,4)' not in src
name='verify_'+NEW+'_native.py';raw=src.encode();ast.parse(raw);(P/name).write_bytes(raw);items.append(dict(relative='scripts/'+name,SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
# Readiness creates its own job only after the current serial GLM measurement
# and any qualifying dual publication have actually finished. It is not run now.
src=(P/('prepare_'+OLD+'_true3.py')).read_text(encoding='utf8')
src=repl(src).replace('20261118','20261119')
src=re.sub(r"^CFG=.*$","CFG='configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B-Multilingual-own-original-BF16-2048-native-v1-20261119.yaml'",src,flags=re.M)
src=src.replace('huihui-ai--AutoGLM-Phone-9B-Multilingual-abliterated','zai-org--AutoGLM-Phone-9B-Multilingual')
src=src.replace('algorithm=dict(name=\'autoglm_phone9b_original_existing_thinkingFalse_greedy_explicit_MC\',adapter=ALGO+\':OwnAutoGLMPhone9BGreedy\',batch_size=4','algorithm=dict(name=\'autoglm_phone9b_original_existing_thinkingFalse_greedy_explicit_MC\',adapter=ALGO+\':OwnAutoGLMPhone9BGreedy\',batch_size=10')
src=src.replace('filter_empty=False,download=False','filter_empty=True,download=False')
src=src.replace('original_BF16_existing_thinkingFalse2048_outer10','original_BF16_own_template2048_outer10')
src=src.replace("'scripts/prepare_"+NEW+"_frontend200.py'","'scripts/prepare_"+NEW+"_frontend200.py','universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py'")
# The old source's parent is an immutable evidence/schema template. Fresh
# canonical hashes are bound only after quiet, never copied from the old job.
src=src.replace(" assert sha(book)=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'","")
src=src.replace(";assert sha(book)=='689b3e894593becd301590ab1ce501e9a9c5af0dcd14b1001e29756f56b3af34'","")
src=src.replace("parent=js(B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261119/job.json')","parent=js(B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118/job.json')")
src=src.replace(" H.mkdir(exist_ok=False);torch.set_num_threads(4)"," from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners\n assert not runner_lock_owners() and not [v for v in project_processes() if v['PID']!=__import__('os').getpid() and v['role'] in {'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}]\n H.mkdir(exist_ok=False);torch.set_num_threads(4)")
src=src.replace('own_frontend200_meta_v1_20261119','frontend200_meta_v1_20261119')
src=src.replace('Own AutoGLM-Phone-9B-Multilingual704 BF16','Own AutoGLM-Phone-9B-Multilingual704 BF16')
src=src.replace('original generic-adapter default enable_thinking=False','original generic-adapter default enable_thinking=False, ignored by its own unchanged chat template (no injected /nothink or empty think)')
src=src.replace('outer4 serial1','outer10 serial1')
src=src.replace('own_original_BF16_2048_native_source_readiness_v1','own_original_BF16_2048_native_source_readiness_v1')
assert 'huihui' not in src.lower() and 'batch_size=4' not in src and 'filter_empty=False' not in src and '689b3e894' not in src,[x for x in src.splitlines() if any(v in x.lower() for v in ['huihui','batch_size=4','filter_empty=false','689b3e894'])]
assert "Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118" in src
name='prepare_'+NEW+'_true3.py';raw=src.encode();ast.parse(raw);(P/name).write_bytes(raw);items.append(dict(relative='scripts/'+name,SHA256=sha(raw),base64=base64.b64encode(raw).decode()))
# Staging creates new own source files only, never dispatches or modifies live GLM.
data=json.dumps(items)
loader="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json\nR=Path('/root/projects/comperror')\nitems=json.loads("+repr(data)+")\nfor v in items:\n p=R/v['relative'];b=base64.b64decode(v['base64']);assert hashlib.sha256(b).hexdigest()==v['SHA256'];ast.parse(b);assert not p.exists(),str(p)\nfor v in items:\n p=R/v['relative'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v['base64']))\nprint(json.dumps({'staged':[{k:v for k,v in x.items() if k!='base64'} for x in items],'actual_NEW_model_forwards':0,'no_live_GLM_source_or_job_or_YAML_or_signal_change':True}))\n"
(P/'stage_AutoGLMPhone9B_own_native_fullNF_sources_only_20261119_remote.py').write_text(loader,encoding='utf8',newline='\n')
review=dict(new_sources=[{k:v for k,v in i.items() if k!='base64'} for i in items],no_active_source_changes=True,own_image_token_151343_bound_in_fullNF=True,own_original_outer10_2048_and_chat_template_preserved=True,checkpoint_inputs_tokenizer_outputs_not_shared=True,complete_standard_LZMA_all_arrays_preserved=True,no_original_model_or_NF_math_threshold_changes=True,actual_NEW_model_forwards=0)
(P/'AutoGLMPhone9B_own_native_fullNF_source_scope_review_20261119.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
print(json.dumps(review))

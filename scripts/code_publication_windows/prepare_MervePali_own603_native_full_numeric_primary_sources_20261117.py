import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,copy,json,hashlib,base64,textwrap
P=Path(__file__).parent
selected=json.loads((P/'R_Merve_selected_spawn_loader_and_Gemma_config_no_NN_20261117.json').read_bytes());name='/root/projects/comperror/universal_eval/model/adapters/merve_paligemma_A100_native_vllm_20261110_v3_spawn_adapter.py';old=selected[name]['text'];assert hashlib.sha256(old.encode()).hexdigest()=='71274f0c063a391933502b974e5d6bc53d6ed1e38485f737f2c6d165fc2feabe'
cls=next(x for x in ast.parse(old).body if isinstance(x,ast.ClassDef));load=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='load');newload=copy.deepcopy(load);call,=[x for x in ast.walk(newload) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='LLM'];ext='universal_eval.model.merve_pali_full_native_value_20261117.worker_named_RPC.OwnMerveFullValueWorkerExtension';call.keywords.append(ast.keyword(arg='worker_extension_cls',value=ast.Constant(ext)))
assert len([k for k in call.keywords if k.arg=='worker_extension_cls'])==1
observer=(P/'pali10pt448_own_preserved_vllm_full_value_named_worker_RPC_v4_20261117.py').read_text();oc=next(x for x in ast.parse(observer).body if isinstance(x,ast.ClassDef) and x.name=='Pali10Pt448OwnFullValuesNamedRPCV4');oc.name='MervePaliOwn603FullNativeValues';oc.bases=[ast.Name(id='MerveNativeNamedWorkerEngine',ctx=ast.Load())];osource=ast.unparse(oc).replace("args=(group,)","args=(group,len(batch))").replace('new_full_numeric_phase_old_compatibility3_and_B10_preserved','new_full_numeric_phase_old_completed_compatibility3_preserved')
adapter='''"""Own Merve original vLLM BF16/2048, standard observation-only named RPC."""
import sys,os,tempfile,json,hashlib
from pathlib import Path
from universal_eval.model.adapters.merve_paligemma_A100_native_vllm_20261110_v3_spawn_adapter import MervePaliGemmaNative910B,MODEL_ID,MAX_TOKENS,EVIDENCE_ROOT,LOG,digest
EXTENSION=__EXT__
class MerveNativeNamedWorkerEngine(MervePaliGemmaNative910B):
__LOAD__
__OBSERVER__
'''.replace('__EXT__',repr(ext)).replace('__LOAD__',textwrap.indent(ast.unparse(newload),'    ')).replace('__OBSERVER__',osource)
worker_record=json.loads((P/'R_Pali10pt_v10_actual_and_all200_capacity_inputs1_20261117.json').read_bytes())['files']['/root/projects/comperror/universal_eval/model/pali10pt448_full_native_value_20261117/worker_v2_ABI.py'];worker=worker_record['text'];assert hashlib.sha256(worker.encode()).hexdigest()==worker_record['SHA256']=='9a80c84d6ef5916801ba39e1be0a86b06637fb40485fc71b795df8e6cd0a1784'
worker=worker.replace('Pali10pt','Merve PaliGemma').replace("'google/paligemma2-10b-pt-448'","'merve/paligemma_vqav2'").replace('==42','==18').replace('== 42','== 18').replace('full_all42','full_all18').replace('reshape(n,8,256)','reshape(n,1,256)').replace('kc.shape[1]==8','kc.shape[1]==1').replace('kc.shape[2]==8','kc.shape[2]==1').replace('attention.q_size==4096 and attention.kv_size==2048','attention.q_size==2048 and attention.kv_size==256').replace('output.shape[-1]==8192','output.shape[-1]==2560')
rpc='''"""Own Merve named RPC observers with unchanged original resource guards."""
from pathlib import Path
import shutil
class OwnMerveFullValueWorkerExtension:
 def own_pali_install_full_value_capture(self,root,headers,home):
  from universal_eval.model.merve_pali_full_native_value_20261117.worker import install_capture
  assert shutil.disk_usage(Path(home)).free>3*1024**3+1024**3+512*1024**2
  return install_capture(self,root,headers,home)
 def own_pali_begin_full_value_capture(self,group,count):
  from universal_eval.model.merve_pali_full_native_value_20261117.worker import begin_capture
  observer=self._own_pali10pt_full_capture;assert 1<=count<=10
  assert shutil.disk_usage(observer.home).free>3*1024**3+1024**3+512*1024**2+count*2*1024**3
  return begin_capture(self,group)
 def own_pali_finish_full_value_capture(self):
  from universal_eval.model.merve_pali_full_native_value_20261117.worker import finish_capture
  observer=self._own_pali10pt_full_capture;exact=sum(a.nbytes for a in observer.arrays.values());assert shutil.disk_usage(observer.home).free>3*1024**3+1024**3+512*1024**2+exact+1024*1024
  return finish_capture(self)
'''
nf=(P/'verify_pali10pt448_own_preserved_vllm_full_native_true3_v9_original_async_EOS_commit_20261117.py').read_text()
nf=nf.replace('from verify_paligemma2_10b_pt448_A100_native_vllm_v2_20261110 import verify as verify_original_contract','from verify_merve_native_vllm_910b_20261110 import verify as verify_original_contract').replace("MID='google/paligemma2-10b-pt-448'","MID='merve/paligemma_vqav2'")
nf=nf.replace("assert count==3,'This source only audits the new missing full-value3 phase, never formal200'","assert count in [3,200]")
nf=nf.replace("job=js(p['full_numeric_evidence_job']);assert job['model_id']==MID and job['stages']==[3] and not job['spec']['approved']","job=js(p['full_numeric_evidence_job']);assert job['model_id']==MID and ((job['stages']==[3] and not job['spec']['approved'] and count==3) or (job['stages']==[3,200] and job['spec']['approved']))")
line=" check(p['old_compat3_proof'],p['old_compat3_proof_SHA256']);check(p['old_B10_proof'],p['old_B10_proof_SHA256']);assert p['old_compat3_proof_SHA256']=='13f4049cb3e6440948ec628140a72fff27e5eaa5a8e856dc547fa4f581e12917' and p['old_B10_proof_SHA256']=='bf0f1c1bb504fa98af0de2f1999a8c8f8405d326f712553ffc7466713b871b1a'"
assert line in nf;nf=nf.replace(line," check(p['old_compat3_proof'],p['old_compat3_proof_SHA256']);assert p['old_compat3_proof_SHA256']=='b5bd29b39bb3833fb0679dc7f86dd2558f33a581dadcd13eae01fbb7e1d89903'")
nf=nf.replace('len(headers)==903','len(headers)==603').replace("len(idx['groups'])==1 and idx['actual_completed_samples']==3 and idx['actual_completed_outer_groups']==1","len(idx['groups'])==(count+9)//10 and idx['actual_completed_samples']==count and idx['actual_completed_outer_groups']==(count+9)//10").replace("loaded['own_source_count']==903","loaded['own_source_count']==603")
nf=nf.replace('transformers.models.gemma2.configuration_gemma2 import Gemma2Config','transformers.models.gemma.configuration_gemma import GemmaConfig').replace('Gemma2Config','GemmaConfig').replace('d81e5abd688436da573ae299a65598d01ea871a8874d0cabeb5149967855ca07','ac4b6de5c5a977278d652ca8d392b5fe10c7e81cdfb72ac691b8e2c3198967e9')
nf=nf.replace("text['num_hidden_layers']==42 and text['hidden_size']==3584 and text['num_key_value_heads']==8","text['num_hidden_layers']==18 and text['hidden_size']==2048 and text['num_key_value_heads']==1")
nf=nf.replace("expected_softcap=text['final_logit_softcapping']","expected_softcap=text.get('final_logit_softcapping',None)")
nf=nf.replace('assert len(trace)==3','assert len(trace)==count').replace('assert len(emitted)==3','assert len(emitted)==count').replace('head.shape==(257216,3584)','head.shape==(257216,2048)').replace('full_all42_layer_KV','full_all18_layer_KV')
nf=nf.replace("assert len(internal_request_map)==len(set(internal_request_map.values()))==len(emitted) and set(internal_request_map.values())==set(emitted)","assert len(internal_request_map)==len(set(internal_request_map.values()))==min(10,count-10*g['group_index']) and set(internal_request_map.values())=={row['request_id'] for row in trace if row['group_index']==g['group_index']}")
nf=nf.replace("(1024,3584)","(256,2048)").replace("len(frame['text_qkv'])==len(frame['cache'])==len(frame['rotary'])==42","len(frame['text_qkv'])==len(frame['cache'])==len(frame['rotary'])==18")
nf=nf.replace("value=(torch.tanh(raw/expected_softcap)*expected_softcap).to(actual.dtype)","value=(raw if expected_softcap is None else torch.tanh(raw/expected_softcap)*expected_softcap).to(actual.dtype)")
nf=nf.replace("inp.shape[-1]==3584 and packed.shape[-1]==8192","inp.shape[-1]==2048 and packed.shape[-1]==2560").replace('packed[:,6144:]','packed[:,2304:]').replace('packed[:,4096:6144].reshape(-1,8,256)','packed[:,2048:2304].reshape(-1,1,256)').replace('all42_layer_native','all18_layer_native').replace('all903_own_sources','all603_own_sources')
nf=nf.replace("assert total_calls==len(g['frames'])","assert total_calls==sum(js(x['path'])['actual_native_forward_calls'] for x in idx['groups'][:g['group_index']+1])")
nf=nf.replace("len(visual_pixels)==3","len(visual_pixels)==count").replace("len(visual_intermediates)==3","len(visual_intermediates)==count").replace("zip(visual_pixels,visual_intermediates))","zip(visual_pixels[:3],visual_intermediates[:3]))")
nf=nf.replace("assert all(a['prompt_token_ids']==b['prompt_token_ids'] for a,b in zip(trace,prior))","assert all(a['prompt_token_ids']==b['prompt_token_ids'] for a,b in zip(trace[:3],prior))")
oldstop="""   sampled=all_selected[row['request_id']];assert eos in sampled and row['finish_reason']=='stop' and row['stop_reason'] is None
   stop=sampled.index(eos)+1;committed=sampled[:stop];trailing=sampled[stop:]
   assert committed==row['generated_token_ids'] and len(trailing)<=1 and all(token==eos for token in trailing),('Exact original EOS commit prefix or preserved in-flight tail mismatch',row['index'])"""
newstop="""   sampled=all_selected[row['request_id']];assert row['stop_reason'] is None
   if eos in sampled[:2048]:assert row['finish_reason']=='stop';stop=sampled.index(eos)+1
   else:assert row['finish_reason']=='length';stop=2048
   committed=sampled[:stop];trailing=sampled[stop:]
   assert committed==row['generated_token_ids'] and len(trailing)<=1 and all(0<=token<257216 for token in trailing),('Exact original EOS/budget commit prefix or preserved single in-flight native tail mismatch',row['index'])"""
assert oldstop in nf;nf=nf.replace(oldstop,newstop)
nf=nf.replace('old_compatibility3_and_B10_unchanged=True','old_completed_compatibility3_unchanged=True').replace('original_protocol_final_qualification_complete=False','original_protocol_final_qualification_complete=base[\'formal_alignment_within_5pct\']')
# The old model-specific diagnostic CLI may never be used for this model.
nf=nf[:nf.index("if __name__=='__main__':")]+"""if __name__=='__main__':
 import contextlib
 with contextlib.redirect_stdout(sys.stderr):value=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
"""
nf=nf.replace('all42','all18').replace('Own Pali10pt','Own Merve PaliGemma')
nn='scripts/verify_merve_pali_own603_original_native_BF16_2048_full_values_v1_20261117.py';an='universal_eval/model/adapters/merve_pali_own603_original_native_full_values_v1_20261117.py'
files={an:adapter,nn:nf,'universal_eval/model/merve_pali_full_native_value_20261117/worker.py':worker,'universal_eval/model/merve_pali_full_native_value_20261117/worker_named_RPC.py':rpc,'universal_eval/model/merve_pali_full_native_value_20261117/__init__.py':''}
for n,s in files.items():ast.parse(s);(P/('Merve_primary_'+Path(n).name)).write_bytes(s.encode())
testa=next(x for x in ast.parse(adapter).body if isinstance(x,ast.ClassDef) and x.name=='MerveNativeNamedWorkerEngine');testb=copy.deepcopy(next(x for x in testa.body if isinstance(x,ast.FunctionDef) and x.name=='load'));call,=[x for x in ast.walk(testb) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='LLM'];call.keywords=[k for k in call.keywords if k.arg!='worker_extension_cls'];assert ast.dump(testb,include_attributes=False)==ast.dump(load,include_attributes=False)
payload={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}
(P/'MervePali_own603_native_full_numeric_primary_source_payload_20261117.json').write_bytes(json.dumps(payload).encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))

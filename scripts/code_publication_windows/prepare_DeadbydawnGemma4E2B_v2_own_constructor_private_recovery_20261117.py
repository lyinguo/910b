import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,base64
P=Path(__file__).parent
old='universal_eval/model/adapters/deadbydawn_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261117.py'
native='universal_eval/model/adapters/deadbydawn_gemma4_e2b_own_original_BF16_native_v2_own_constructor_binding_20261117.py'
raw='''"""Own constructor model-ID binding; inherited original math is unchanged."""
from pathlib import Path
from universal_eval.model.adapters.deadbydawn_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261117 import DeadbydawnGemma4E2BOwnBF16,MID
class DeadbydawnGemma4E2BOwnBF16V2(DeadbydawnGemma4E2BOwnBF16):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and npu_devices==[4]
        self.protocol_file=Path(protocol_file);self.protocol_sha256=protocol_sha256;self.devices=npu_devices
'''
ast.parse(raw)
old_measure=(P/'measure_deadbydawn_gemma4_e2b_own_original_BF16_actual_capacity_v1_saved_HF_F32_20261117.py').read_text(encoding='utf8')
measure=old_measure.replace('DeadbydawnGemma4E2B_own_original_BF16_true3_v1_saved_HF_F32_real_FLOCK_20261117','DeadbydawnGemma4E2B_own_original_BF16_true3_v2_own_constructor_binding_real_FLOCK_20261117').replace('DeadbydawnGemma4E2B_own_original_BF16_complete_true3_capacity_review_v1_saved_HF_F32_20261117','DeadbydawnGemma4E2B_own_original_BF16_complete_true3_capacity_review_v2_own_constructor_binding_20261117')
# The stage observer is exactly inherited from the frozen own V1 implementation.
measure=measure.replace("assert 'self.index<3' in source.read_text() # actual inherited _install stage observer","assert 'self.index<3' in source.read_text() # unchanged actual V1 _install inherited by the own V2 constructor subclass")
mrel='scripts/measure_deadbydawn_gemma4_e2b_actual_capacity_v2_own_constructor_binding_20261117.py'
old_cont=(P/'continue_deadbydawn_gemma4_e2b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261117.py').read_text(encoding='utf8')
cont=old_cont.replace('v1_saved_HF_F32','v2_own_constructor_binding').replace('v1-saved-HF-F32-approved-first200','v2-own-constructor-binding-approved-first200')
cont=cont.replace('measure_deadbydawn_gemma4_e2b_own_original_BF16_actual_capacity_v2_own_constructor_binding_20261117.py',Path(mrel).name)
cont=cont.replace("H.mkdir(exist_ok=True);guard=", "H.mkdir(exist_ok=False);guard=")
needle=" assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]"
assert needle in cont
cont=cont.replace(needle,""" import time
 diagnostic=B/'DonutInvoice_existing_complete200_v6_present_raw_full_numeric_independent_diagnostic_real_FLOCK_20261117'
 while True:
  live=[p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners()
  if (diagnostic/'actual_completion.json').exists() and not live and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_real_existing_Donut_numeric_review_completion_and_original_lock_release',actual_Donut_review_completion_exists=(diagnostic/'actual_completion.json').exists(),actual_project_NN_or_NF=live,actual_FLOCK=locks,no_NPU_model_started=True));time.sleep(15)
 put('preceding_Donut_independent_diagnostic_actual_completion_preserved.json',js(diagnostic/'actual_completion.json'))""")
crel='scripts/continue_deadbydawn_gemma4_e2b_true3_capacity_fresh3_fixed200_v2_own_constructor_binding_20261117.py'
ready='''"""New private constructor-only recovery, never edit or accept the failed V1 job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,copy,importlib,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
OLD=B/'DeadbydawnGemma4E2B_own_original_BF16_true3_v1_saved_HF_F32_real_FLOCK_20261117';H=B/'DeadbydawnGemma4E2B_own_original_BF16_native_v2_own_constructor_binding_readiness_20261117';J=B/'DeadbydawnGemma4E2B_own_original_BF16_true3_v2_own_constructor_binding_real_FLOCK_20261117'
NATIVE=__NATIVE__;MEASURE=__MEASURE__;CONTROLLER=__CONTROLLER__;SELF=__SELF__
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 old=js(OLD/'job.json');assert js(OLD/'completion.json')['actual_returncode']==1 and not js(OLD/'completion.json')['accepted']
 protected={str(f):sha(f) for f in [OLD/'job.json',OLD/'completion.json',D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]}
 cfg=js(R/old['spec']['config']);p=js(cfg['model']['kwargs']['protocol_file']);assert p['model_id']==MID
 mod=importlib.import_module('universal_eval.model.adapters.deadbydawn_gemma4_e2b_own_original_BF16_native_v2_own_constructor_binding_20261117');cls=mod.DeadbydawnGemma4E2BOwnBF16V2
 instance=cls(**cfg['model']['kwargs']);assert instance.protocol_file==Path(cfg['model']['kwargs']['protocol_file']) and instance.devices==[4] and instance.model_name==MID
 bad=dict(cfg['model']['kwargs']);bad['protocol']={'model_id':'another-model'}
 try:cls(**bad)
 except AssertionError:pass
 else:raise AssertionError('Foreign model constructor must be rejected')
 parent=cls.__mro__[1]
 unchanged_methods=['load','_install','batch_predict','get_sample_execution','_predict_original_compute_and_capture']
 for name in unchanged_methods:assert getattr(cls,name) is getattr(parent,name),name
 review=dict(passed=True,own_model_constructor_accepts_exact_own_ID=True,foreign_model_constructor_rejected=True,only_constructor_binding_changed=True,original_load_predict_generate_and_observer_function_objects_unchanged=True,old_failed_job_not_replaced=True,no_NPU_model_forward=True)
 create(H/'constructor_binding_review.json',review)
 for name in [NATIVE,MEASURE,CONTROLLER,SELF]:ast.parse((R/name).read_bytes());p['implementation_source_SHA256'][name]=sha(R/name)
 p['static_evidence_SHA256'][str(H/'constructor_binding_review.json')]=sha(H/'constructor_binding_review.json')
 pp=R/'configs/protocols/deadbydawn_gemma4_e2b_own_original_BF16_native_v2_own_constructor_binding_20261117.json';create(pp,p)
 cf=R/'configs/Gemma4ForConditionalGeneration/deadbydawn101--gemma-4-E2B-Heretic-Uncensored-mlx-4bit-own-original-BF16-2048-native-v2-own-constructor-binding-20261117.yaml';cfg['name']=cf.stem;cfg['model']['adapter']=NATIVE+':DeadbydawnGemma4E2BOwnBF16V2';cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 job=copy.deepcopy(old);job['created_at']=datetime.now(timezone.utc).isoformat();job['spec']['config']=str(cf.relative_to(R));job['spec']['review_revision']='own_constructor_only_v2_no_original_math_change';job['spec']['source_sha256'].update({name:sha(R/name) for name in [NATIVE,MEASURE,CONTROLLER,SELF,str(pp.relative_to(R)),str(cf.relative_to(R))]});job['protected_canonical_hashes']={name:sha(name) for name in old['protected_canonical_hashes']}
 batch.check_sources(job['spec'])
 for count in [3,200]:_,c=batch.config_for(job['spec'],js(D/'state.json')['models'][MID],count);validate_config(c,MID)
 scope(job['spec'],[3],False);assert not inspect.isabstract(cls);create(J/'job.json',job)
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,ready_for_original_runner_true3=True,true3_job=str(J),job_SHA256=sha(J/'job.json'),standalone_YAML=str(cf),constructor_review_SHA256=sha(H/'constructor_binding_review.json'),no_NPU_before_original_runner=True,old_failed_job_and_complete_frontend200_unchanged=True)
 assert protected=={name:sha(name) for name in protected};create(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
'''
rrel='scripts/prepare_deadbydawn_gemma4_e2b_native_readiness_v2_own_constructor_binding_20261117.py'
ready=ready.replace('__NATIVE__',repr(native)).replace('__MEASURE__',repr(mrel)).replace('__CONTROLLER__',repr(crel)).replace('__SELF__',repr(rrel))
items={native:raw.encode(),mrel:measure.encode(),crel:cont.encode(),rrel:ready.encode()}
for name,s in items.items():ast.parse(s);(P/Path(name).name).write_bytes(s)
payload={name:dict(data=base64.b64encode(s).decode(),SHA256=hashlib.sha256(s).hexdigest()) for name,s in items.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';ITEMS=__ITEMS__
for rel,item in ITEMS.items():
 raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['SHA256'];ast.parse(raw);f=R/rel
 if f.exists():assert f.read_bytes()==raw
 else:f.write_bytes(raw)
I=B/'DeadbydawnGemma4E2B_private_v2_constructor_recovery_readiness_actual_launch_20261117';I.mkdir(exist_ok=False)
with (I/'stdout.json').open('xb') as out,(I/'stderr.log').open('xb') as err:p=subprocess.run(['python3','-B','scripts/prepare_deadbydawn_gemma4_e2b_native_readiness_v2_own_constructor_binding_20261117.py'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,sources={n:x['SHA256'] for n,x in ITEMS.items()},stdout=(I/'stdout.json').read_text(),stderr=(I/'stderr.log').read_text(),no_NPU_dispatch=True);(I/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False));raise SystemExit(p.returncode)
'''.replace('__ITEMS__',repr(payload));ast.parse(remote);(P/'deploy_DeadbydawnGemma4E2B_private_v2_constructor_recovery_readiness_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))

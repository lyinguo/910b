"""New private constructor-only recovery, never edit or accept the failed V1 job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,copy,importlib,inspect
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
OLD=B/'DeadbydawnGemma4E2B_own_original_BF16_true3_v1_saved_HF_F32_real_FLOCK_20261117';H=B/'DeadbydawnGemma4E2B_own_original_BF16_native_v2_own_constructor_binding_readiness_20261117';J=B/'DeadbydawnGemma4E2B_own_original_BF16_true3_v2_own_constructor_binding_real_FLOCK_20261117'
NATIVE='universal_eval/model/adapters/deadbydawn_gemma4_e2b_own_original_BF16_native_v2_own_constructor_binding_20261117.py';MEASURE='scripts/measure_deadbydawn_gemma4_e2b_actual_capacity_v2_own_constructor_binding_20261117.py';CONTROLLER='scripts/continue_deadbydawn_gemma4_e2b_true3_capacity_fresh3_fixed200_v2_own_constructor_binding_20261117.py';SELF='scripts/prepare_deadbydawn_gemma4_e2b_native_readiness_v2_own_constructor_binding_20261117.py'
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

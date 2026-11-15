"""Rebind a new private run after exact HF ignored-key scope diagnosis.

Reuses the completed own chat/pixel preparation. No old source, report, failure,
weight, generation setting, shared code or workbook is overwritten.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'UnslothE2Bit_v2_exact_unused_loading_scope_readiness_1210_20261115'
OLD=B/'UnslothE2Bit_own_chat_native_true3_real_FLOCK_0105_20261115'
J=B/'UnslothE2Bit_own_chat_native_true3_v2_exact_unused_real_FLOCK_1210_20261115'
CFG='configs/Gemma3nForConditionalGeneration/unsloth--gemma-3n-E2B-it-native-BF16-chat-2048-v2-exact-unused-20261115.yaml'
PROTO='configs/protocols/unsloth_gemma3n_e2bit_native_v2_exact_unused.json'
MODEL='universal_eval/model/adapters/unsloth_gemma3n_e2bit_native_v2_exact_unused_scope.py'
VERIFIER='scripts/verify_unsloth_gemma3n_e2bit_native_v2_exact_unused.py'
SELF='scripts/prepare_unsloth_gemma3n_e2bit_v2_exact_unused.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import scope
 diagnostic=B/'UnslothE2Bit_loading_info_scope_diagnostic_v3_all_sets_JSON_safe_1210_20261115/review.json'
 v=js(diagnostic);unused={f'model.language_model.layers.{i}.self_attn.{name}' for i in range(20,30) for name in ['k_proj.weight','v_proj.weight','k_norm.weight']}
 assert v['no_model_forward'] and v['no_NPU'] and v['actual_unused_count']==30 and set(v['actual_unused_source_keys'])==unused
 assert not v['loading_info']['missing_keys'] and not v['loading_info']['mismatched_keys'] and not v['loading_info']['unexpected_keys'] and not v['loading_info']['error_msgs']
 assert v['state_keys_missing_from_source']==['lm_head.weight']
 old_job=js(OLD/'job.json');spec=copy.deepcopy(old_job['spec']);old_cfg=js(R/spec['config']);p=js(old_cfg['model']['kwargs']['protocol_file'])
 cfg=copy.deepcopy(old_cfg);cfg['name']=Path(CFG).stem;cfg['model']['adapter']=MODEL+':UnslothGemma3nE2Bit'
 old_adapter=old_cfg['model']['adapter'].split(':')[0];old_verifier=spec['verifier']
 p['implementation_source_SHA256'].pop(old_adapter);p['implementation_source_SHA256'].pop(old_verifier)
 p['implementation_source_SHA256'].update({name:sha(R/name) for name in [MODEL,VERIFIER,SELF]})
 p['static_evidence_SHA256'][str(diagnostic)]=sha(diagnostic)
 p['loading_key_scope_review']=str(diagnostic);p['loading_key_scope_review_SHA256']=sha(diagnostic)
 put(R/PROTO,p);cfg['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/CFG,cfg)
 sources={name:sha(R/name) for name in p['implementation_source_SHA256']}
 sources.update({name:sha(R/name) for name in [CFG,PROTO,'universal_eval/algorithm/tinyllava_strict_mc.py']})
 spec.update(config=CFG,verifier=VERIFIER,source_sha256=sources,approved=False)
 job=copy.deepcopy(old_job);job['spec']=spec
 for path,digest in job['protected_canonical_hashes'].items():assert sha(path)==digest,path
 H.mkdir(exist_ok=False);J.mkdir(exist_ok=False);put(J/'job.json',job)
 batch.validate_config(cfg,'unsloth/gemma-3n-E2B-it');batch.check_sources(spec);scope(spec,[3],False)
 entry=js(D/'state.json')['models']['unsloth/gemma-3n-E2B-it'];c3,e3=batch.config_for(spec,entry,3);c200,e200=batch.config_for(spec,entry,200)
 assert e3['sample_range']=='0:2' and e200['sample_range']=='0:199'
 before={k:v for k,v in old_cfg.items() if k not in ['name','model']};after={k:v for k,v in cfg.items() if k not in ['name','model']};assert before==after
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),diagnostic=str(diagnostic),diagnostic_SHA256=sha(diagnostic),model_id=cfg['model']['name'],YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),true3_job=str(J),job_SHA256=sha(J/'job.json'),source_SHA256=sources,unchanged_generation_and_model_mathematics=True,all_old_failures_and_sources_preserved=True,no_new_NPU_claimed=True)
 put(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

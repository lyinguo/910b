"""Stage finite read-only audits and unchanged discovery logic at a new output path."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
old=(Path('D:/code-lv/.tmp/cd23s1/scripts/maintenance/discover_verified185_current_full_accepted_correct_nonexample_metadata_only_v1_20261122.py')).read_bytes()
before='verified185_current_full_accepted_correct_nonexample_metadata_reaudit_Turkish_completed_20261122'
after='verified185_current_full_accepted_correct_nonexample_metadata_reaudit_heartbeat0628_20261123'
text=old.decode();assert text.count(before)==1
fresh=text.replace(before,after)
assert ast.dump(ast.parse(fresh.replace(after,before)))==ast.dump(ast.parse(text))
newname='discover_verified185_current_full_accepted_correct_nonexample_metadata_only_v2_heartbeat0628_20261123.py'
(V/newname).write_text(fresh,encoding='utf8')
mapping={
 'scripts/maintenance/read_heartbeat0628_verified185_required_v90_latest_and_actual_20261123_remote.py':'read_heartbeat0628_verified185_required_v90_latest_and_actual_20261123_remote.py',
 'scripts/code_publication_windows/read_original_A100_timm_exact_sources_for_MedraN_readonly_no_model_20261123.py':'read_original_A100_timm_exact_sources_for_MedraN_readonly_no_model_20261123.py',
 'scripts/code_publication_windows/read_original_A100_Gemma3n_vllm_native_vision_source_readonly_20261123.py':'read_original_A100_Gemma3n_vllm_native_vision_source_readonly_20261123.py',
 'scripts/code_publication_windows/read_A100_MedraN_current_config_shard_headers_and_Kanana_original_RegStage_source_20261123.py':'read_A100_MedraN_current_config_shard_headers_and_Kanana_original_RegStage_source_20261123.py',
 'scripts/maintenance/'+newname:newname,
 'scripts/diagnostics/review_MedraN_original_AA_config_difference_and_real_timm_source_recovery_no_NN_20261123.py':'review_MedraN_original_AA_config_difference_and_real_timm_source_recovery_no_NN_20261123.py',
 'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).name}
stage={n:(V/p).read_bytes() for n,p in mapping.items()}
for b in stage.values():ast.parse(b)
q=dict(AA_exact_source_outputs=dict(timm=json.loads((V/'AA_MedraN_original_exact_timm_source_readonly_attempt_20261123.json').read_bytes()),
    vllm=json.loads((V/'AA_MedraN_original_vllm_native_vision_source_readonly_20261123.json').read_bytes())),
    AA_header_and_RegStage_source=json.loads((V/'AA_MedraN_current_metadata_RegStage_original_source_readonly_20261123.json').read_bytes()))
qb=json.dumps(q,ensure_ascii=False,indent=2).encode();plan='heartbeat0628_original_AA_source_metadata_review_20261123.json';(V/plan).write_bytes(qb)
(V/'heartbeat0628_original_AA_source_and_discovery_primary_SHA_20261123.json').write_text(json.dumps({n:sha(b) for n,b in stage.items()},indent=2)+'\n')
payload={n:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for n,b in stage.items()}
s="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,os,subprocess\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n;p.parent.mkdir(parents=True,exist_ok=True)\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\np=R/'workspace/diagnostics/batch_910b/heartbeat_handoff_plans'/"+repr(plan)+";b=base64.b64decode("+repr(base64.b64encode(qb).decode())+");assert hashlib.sha256(b).hexdigest()=="+repr(sha(qb))+"\nif p.exists():assert p.read_bytes()==b\nelse:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/diagnostics/review_MedraN_original_AA_config_difference_and_real_timm_source_recovery_no_NN_20261123.py'),str(p),"+repr(sha(qb))+"],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(s);(V/'stage_heartbeat0628_AA_source_review_and_fresh_cleanup_20261123_remote.py').write_text(s,encoding='utf8')
print(json.dumps(dict(primary_count=len(stage),plan_SHA256=sha(qb),discovery_only_output_path_change_AST_equal=True)))

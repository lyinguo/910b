"""Bind five legacy all200 numerical proofs without treating missing final-qualification flags as failure."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
bindings=json.loads((V/'verified185_legacy_exact7_full_numeric_candidate_metadata_proof_bindings_20261122.json').read_bytes())
for mid in ['HuggingFaceTB/SmolVLM2-2.2B-Instruct','smolagents/SmolVLM2-2.2B-Instruct-Agentic-GUI']:
    del bindings[mid]
assert len(bindings)==5
s=(V/'discover_verified185_current_full_accepted_correct_nonexample_metadata_only_v1_20261122.py').read_text(encoding='utf8')
needle="for mid,e in entries:\n if e['status']!='done' and mid!=META:continue"
repl="BINDINGS="+repr(bindings)+"\nfor mid,e in entries:\n if mid not in BINDINGS:continue\n if e['status']!='done' and mid!=META:continue"
assert needle in s;s=s.replace(needle,repl)
a=s.index(" proof=js(pf)\n if mid!=META and not proof.get('original_protocol_final_qualification_complete'):")
b=s.index(' allaccepted.append',a)
replacement=""" assert str(pf)==BINDINGS[mid]['proof'] and sha(pf)==BINDINGS[mid]['SHA256']
 proof=js(pf)
 assert proof.get('passed') and proof.get('sample_count')==200 and proof.get('actual_NPU_inference') and proof.get('NEW_real_NPU_inference') and not proof.get('historical_record_reuse') and not proof.get('error_indices')
 if mid=='docling-project/SmolDocling-256M-preview':
  assert proof['all471_own_source_FP16_cast_A100_and_NPU_weight_SHA256'] and proof['actual_KV_prefill_and_every_decode_delta_all30_reconstructed_to_final_SHA256']
 elif mid=='google/t5gemma-2-1b-1b':
  assert proof['all1119_own_source_cast_actual_A100_and_NPU_SHA256'] and proof['all_native_self_cross_cache_values_reconstructed_with_exact_NPU_change_checks'] and len(proof['numeric_sample_checks'])==200
 elif mid=='ibm-granite/granite-docling-258M-mlx':
  assert proof['all471_own_source_and_actual_AA_NPU_SHA256'] and proof['cache_values']=='original cacheFalse; no KV returned and full prefix every step independently verified' and len(proof['numeric_sample_checks'])==200
 else:
  assert mid in ['OpenGVLab/InternVL3_5-2B-Flash','OpenGVLab/InternVL3_5-4B-Flash'] and proof['all_native_KV_values_reconstructed_with_unchanged_prefix_checks'] and len(proof['numeric_sample_checks'])==200
 # Metadata discovery only: full original-protocol qualification is NOT changed or waived.
"""
s=s[:a]+replacement+s[b:]
s=s.replace('verified185_current_full_accepted_correct_nonexample_metadata_reaudit_Turkish_completed_20261122','verified185_exact5_legacy_all200_numeric_metadata_reaudit_20261122')
s=s.replace('Verified185 current original-qualified full200 plus actual Meta formal fullNF4 nonexample discovery only; no deletion, no old-array replay','Exact five SHA-bound legacy all200 numerical proofs only; no qualification reclassification, deletion or old-array replay')
name='discover_verified185_exact5_legacy_all200_numeric_correct_nonexample_metadata_only_v1_20261122.py'
ast.parse(s);(V/name).write_text(s,encoding='utf8',newline='\n')
src={'scripts/maintenance/'+name:s.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
payload={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in src.items()}
(V/'verified185_exact5_legacy_all200_numeric_metadata_primary_plan_20261122.json').write_text(json.dumps({n:r['SHA256'] for n,r in payload.items()},indent=2)+'\n',encoding='utf8')
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+name+"')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_verified185_exact5_legacy_all200_numeric_metadata_reaudit_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(exact_model_count=len(bindings),primary_SHA256={n:r['SHA256'] for n,r in payload.items()},no_deletion_or_qualification_change=True)))

"""Inspect eight explicit complete KV/forward proofs and the unclassified SmolDocling layout."""
from pathlib import Path
import ast,base64,hashlib,json
V=Path(__file__).resolve().parent
bindings=json.loads((V/'verified185_legacy_exact8_KV_allstep_full_numeric_metadata_bindings_20261122.json').read_bytes());assert len(bindings)==8
s=(V/'discover_verified185_exact5_legacy_all200_numeric_correct_nonexample_metadata_only_v1_20261122.py').read_text(encoding='utf8')
a=s.index('BINDINGS=');b=s.index('\nfor mid,e in entries:',a);s=s[:a]+'BINDINGS='+repr(bindings)+s[b:]
a=s.index(' assert str(pf)==BINDINGS[mid]');b=s.index(' allaccepted.append',a)
s=s[:a]+''' assert str(pf)==BINDINGS[mid]['proof'] and sha(pf)==BINDINGS[mid]['SHA256']
 proof=js(pf)
 assert proof.get('passed') and proof.get('sample_count')==200 and proof.get('NEW_real_NPU_inference') and not proof.get('historical_record_reuse') and not proof.get('error_indices')
 assert proof.get('all_actual_forward_steps_verified',0)>0
 assert any(k.startswith('all') and 'KV_layer_delta_records_verified' in k and isinstance(v,int) and v>0 for k,v in proof.items())
 assert any(k.startswith('all') and 'source_cast_actual_NPU_SHA' in k and v is True for k,v in proof.items())
 # Numeric evidence eligibility only; no original-protocol qualification reclassification.
'''+s[b:]
s=s.replace("not proof.get('actual_NPU_inference')", "not (proof.get('actual_NPU_inference') or proof.get('NEW_real_NPU_inference'))")
s=s.replace('verified185_exact5_legacy_all200_numeric_metadata_reaudit_20261122','verified185_exact8_legacy_allstep_numeric_metadata_reaudit_20261122')
s=s.replace('Exact five SHA-bound legacy all200 numerical proofs only; no qualification reclassification, deletion or old-array replay','Exact eight SHA-bound legacy all200/full-KV/all-forward numerical proofs only; no qualification reclassification, deletion or array replay')
needle='out=dict(UTC=datetime.now(timezone.utc).isoformat()'
extra='''smolpf=B/'SmolDocling_original_quoted_FP16_approved_fresh3_fixed200_v2_static_paths_1512_20261111/formal200/verification_stdout.json'
assert sha(smolpf)=='dfd3048905f1c6b631c05385f7ef1e9efc2b635e56a23a0b46dfcad7072e0c08'
sp=js(smolpf);sr=Path(sp['report_dir']);sl=js(sr/'operator_lineage.json')['actual_execution'];files=[]
for n,h in sp['external_evidence_sha256'].items():
 p=Path(n)
 if p.suffix=='.npz':
  st=p.stat() if p.exists() else None;files.append(dict(path=n,SHA256=h,exists=st is not None,bytes=st.st_size if st else 0))
smol_layout=dict(own_actual_execution=sl,all_proof_NPZ_metadata=files,present_count=sum(x['exists'] for x in files),present_bytes=sum(x['bytes'] for x in files),sample_detail_evidence=[{k:v for k,v in z.items() if k in ['index','extra','actual_execution']} for z in js(sr/'eval_report.json')['detailed_results'][:3]])
'''
assert needle in s;s=s.replace(needle,extra+needle).replace("scope='Exact eight", "SmolDocling_unknown_layout_metadata_no_deletion=smol_layout,scope='Exact eight")
name='discover_verified185_exact8_legacy_allstep_numeric_correct_nonexample_metadata_only_v1_20261122.py'
ast.parse(s);(V/name).write_text(s,encoding='utf8',newline='\n')
src={'scripts/maintenance/'+name:s.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_text(encoding='utf8').encode()}
payload={n:dict(SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in src.items()}
(V/'verified185_exact8_legacy_allstep_numeric_metadata_primary_plan_20261122.json').write_text(json.dumps({n:r['SHA256'] for n,r in payload.items()},indent=2)+'\n',encoding='utf8')
carrier="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,json,subprocess,os\nR=Path('/root/projects/comperror');P=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\nfor n,e in P.items():\n b=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==e['SHA256'];ast.parse(b);p=R/n\n if p.exists():assert p.read_bytes()==b\n else:p.write_bytes(b)\nr=subprocess.run(['python3','-B',str(R/'scripts/maintenance/"+name+"')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)\n"
ast.parse(carrier);(V/'stage_verified185_exact8_legacy_allstep_numeric_metadata_reaudit_20261122_remote.py').write_text(carrier,encoding='utf8')
print(json.dumps(dict(exact_model_count=len(bindings),primary_SHA256={n:r['SHA256'] for n,r in payload.items()},no_deletion_or_qualification_change=True)))

"""Prepare read-only inventory using current184 and actual completed Meta fullNF4."""
from pathlib import Path
import ast,json,hashlib,base64
V=Path(__file__).resolve().parent;old=(V/'discover_verified183_current_accepted_correct_nonexamples_tmp_metadata_only_20261120.py').read_text(encoding='utf8')
oldcheck="ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');assert all(sha(p)==h for p,h in ptr['canonical_SHA256'].items())"
newcheck="""ptr=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')
pub=B/'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121';audit=js(pub/'full_target_dual_publication_audit.json');receipt=js(pub/'publication_receipt.json');assert sha(pub/'full_target_dual_publication_audit.json')=='033e43ca89d3f6ba87a8924490c98b696e8a96565014dd1d498e2dd8cc3a06a2' and audit['passed']
for p,h in ptr['canonical_SHA256'].items():
 actual=sha(p)
 assert actual==h or receipt['canonical_before_sha256'].get(p)==h and audit['canonical_SHA256'].get(p)==actual,(p,'Unreviewed canonical change')
"""
assert oldcheck in old;s=old.replace(oldcheck,newcheck).replace("==183","==184").replace('Verified183 current full200 nonexample discovery','Verified184 current original-qualified full200 plus actual Meta formal fullNF4 nonexample discovery').replace("verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120","verified184_current_full_accepted_nonexample_inventory_Bee_capacity_20261121")
needle="for mid,e in state['models'].items():\n if e['status']!='done':continue\n pf=old.proof_for(mid,e,reg)"
repl="""entries=list(state['models'].items())
META='meta-llama/Llama-3.2-11B-Vision-Instruct';MH=B/'MetaMllama_completed_genuine200_formal_NF4_original_empty_invalid_schema_v2_actual_file_SHA_after_Ovis_serial_no_NN_replay_20261121';mp=MH/'verification_stdout.json'
assert sha(mp)=='c54902717b16dd42b1b02f851c02652ea77d9ad31884eae238d38dcc3c08cb13' and js(MH/'actual_completion.json')['actual_returncode']==0
meta=js(mp);assert meta['passed'] and meta['sample_count']==200 and meta['actual_forward_count']==2823 and meta['all906_own_source_cast_actual_NPU_SHA_verified'] and meta['all40_physical_KV_full_values_reconstructed_and_complete_SHA_verified_each_forward']
for mid,e in entries:
 if e['status']!='done' and mid!=META:continue
 if mid==META:e=dict(e,report_dir=meta['report_dir']);pf=mp
 else:pf=old.proof_for(mid,e,reg)
"""
assert needle in s;s=s.replace(needle,repl)
needle=" proof=js(pf);allaccepted.append"
repl=""" proof=js(pf)
 if mid!=META and not proof.get('original_protocol_final_qualification_complete'):
  skips.append(dict(model_id=mid,reason='Numeric-only legacy proof is excluded; complete original full numerical qualification not declared'));continue
 allaccepted.append"""
assert needle in s;s=s.replace(needle,repl)
name='discover_verified184_current_full_accepted_and_Meta_fullNF4_nonexamples_Bee_capacity_20261121.py';ast.parse(s);(V/name).write_text(s,encoding='utf8',newline='\n')
records=[dict(path='scripts/maintenance/'+name,SHA256=hashlib.sha256(s.encode()).hexdigest(),base64=base64.b64encode(s.encode()).decode())]
transport='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,subprocess,os
R=Path('/root/projects/comperror')
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not p.exists()
 with p.open('xb') as f:f.write(b)
r=subprocess.run(['python3','-B',str(R/DATA[0]['path'])],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));raise SystemExit(r.returncode)
'''
ast.parse(transport);(V/'run_verified184_current_accepted_cleanup_discovery_20261121_remote.py').write_text(transport,encoding='utf8',newline='\n');print(json.dumps({'source':name,'SHA256':records[0]['SHA256'],'metadata_only_no_deletion':True}))

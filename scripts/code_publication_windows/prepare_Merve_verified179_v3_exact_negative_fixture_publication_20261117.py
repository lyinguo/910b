import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
from decimal import Decimal
P=Path(__file__).parent;H=P/'MervePaliCompact_local_result_sync_verified179_20261117'
old=json.loads((P/'R_Merve_writer_v2_failed_isolated_case_source_unchanged_baseline_20261117.json').read_bytes())['files']['scripts/tests/Merve_actual_formal_acceptance_manifest_writer_11_cases_template_20261117.py']
assert old['SHA256']=='0e88d225e71ac77c2a87f614cd4dbe0bcd79d2f031e5581f16f773b49190255e'
assert abs(Decimal('.10339966521681303')-Decimal('.1'))/Decimal('.1')<Decimal('.05')
assert abs(Decimal('.10339966521681303')-Decimal('.09'))/Decimal('.09')>Decimal('.05')
t=old['text'];needle='<v>0.1</v>';assert t.count(needle)==1;t=t.replace(needle,'<v>0.09</v>');ast.parse(t)
tn='scripts/tests/Merve_actual_formal_acceptance_writer_11_cases_exact_outside_reference_template_v3_20261117.py';th=hashlib.sha256(t.encode()).hexdigest()
Q=json.loads((H/'reviewed_server_publication_inputs_v2.json').read_bytes());Q.update(test_template=tn,test_template_SHA256=th,test_fixture_review=dict(original_negative_fixture_D='.1',actual_relative_error='0.0339966521681303',corrected_negative_fixture_D='.09',corrected_relative_error=str(abs(Decimal('.10339966521681303')-Decimal('.09'))/Decimal('.09')),no_production_D_or_score_or_guard_changed=True,all_11_original_cases_retained=True));raw=json.dumps(Q,ensure_ascii=False,indent=2).encode();qh=hashlib.sha256(raw).hexdigest();(H/'reviewed_server_publication_inputs_v3.json').write_bytes(raw)
pub=(P/'publish_MervePaliCompact_verified179_dual_transaction_v2_real_formal_acceptance_schema_20261117.py').read_text(encoding='utf8');pub=pub.replace('MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v2_eligible_phase_schema_20261117','MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117').replace('_verified179_v2_formal_acceptance_writer_isolated_','_verified179_v3_exact_negative_fixture_writer_isolated_');ast.parse(pub);pn='scripts/publish_MervePaliCompact_verified179_dual_transaction_v3_exact_negative_fixture_20261117.py';ph=hashlib.sha256(pub.encode()).hexdigest();(P/Path(pn).name).write_bytes(pub.encode())
supervisor=(P/'run_MervePaliCompact_verified179_dual_publication_v2_actual_completion_20261117.py').read_text(encoding='utf8').replace('MervePaliCompact_verified179_publication_review_inputs_v2_real_formal_acceptance_schema_20261117','MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117').replace('scripts/publish_MervePaliCompact_verified179_dual_transaction_v2_real_formal_acceptance_schema_20261117.py',pn).replace('e33341391844002f9b747108a0bcfd353925452b7205d77eec35743515d09a37',ph).replace('6f6bc12dcbd6b0308c59bbdfa7328da0fc37bc986e131dee93d077cffd0b5473',qh);ast.parse(supervisor);sn='scripts/run_MervePaliCompact_verified179_dual_publication_v3_exact_negative_fixture_actual_completion_20261117.py';sh=hashlib.sha256(supervisor.encode()).hexdigest();(P/Path(sn).name).write_bytes(supervisor.encode())
launch=(P/'launch_Merve_verified179_v2_real_acceptance_schema_reviewed_publication_20261117_remote.py').read_text(encoding='utf8')
# Regenerate the embedded reviewed inputs and supervisor, preserving all original dispatch checks.
parsed=ast.parse(launch)
for n in ast.walk(parsed):
 if isinstance(n,ast.Constant) and isinstance(n.value,str):
  replacements={'scripts/publish_MervePaliCompact_verified179_dual_transaction_v2_real_formal_acceptance_schema_20261117.py':pn,'e33341391844002f9b747108a0bcfd353925452b7205d77eec35743515d09a37':ph,'6f6bc12dcbd6b0308c59bbdfa7328da0fc37bc986e131dee93d077cffd0b5473':qh,'scripts/run_MervePaliCompact_verified179_dual_publication_v2_actual_completion_20261117.py':sn,'d21bcc2270c81d18e1c391584a9f79bd7db5dd52b18a503ac71412e9c5731c06':sh,'MervePaliCompact_verified179_publication_review_inputs_v2_real_formal_acceptance_schema_20261117':'MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117'}
  if n.value in replacements:n.value=replacements[n.value]
  elif n.value.startswith('R:'):pass
  else:
   try:b=base64.b64decode(n.value,validate=True)
   except Exception:continue
   if b==(H/'reviewed_server_publication_inputs_v2.json').read_bytes():n.value=base64.b64encode(raw).decode()
   elif b==(P/'run_MervePaliCompact_verified179_dual_publication_v2_actual_completion_20261117.py').read_bytes():n.value=base64.b64encode(supervisor.encode()).decode()
launch=ast.unparse(parsed)+'\n';ast.parse(launch)
files={tn:dict(SHA256=th,data=base64.b64encode(t.encode()).decode()),pn:dict(SHA256=ph,data=base64.b64encode(pub.encode()).decode())}
prefix='import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport hashlib,base64,ast\nR=Path("/root/projects/comperror")\nF='+repr(files)+'\nfor name,v in F.items():\n p=R/name;assert not p.exists();r=base64.b64decode(v["data"]);assert hashlib.sha256(r).hexdigest()==v["SHA256"];ast.parse(r);p.write_bytes(r)\n'
launch=prefix+launch;(P/'launch_Merve_verified179_v3_exact_negative_fixture_reviewed_publication_20261117_remote.py').write_bytes(launch.encode())
reader=(P/'read_Merve_verified179_v2_server_book_for_local_CAS_20261117_remote.py').read_text(encoding='utf8').replace('MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v2_eligible_phase_schema_20261117','MervePaliCompact_original_native_BF16_verified179_blank_G_publication_v3_exact_negative_fixture_20261117').replace('MervePaliCompact_verified179_publication_review_inputs_v2_real_formal_acceptance_schema_20261117','MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117');ast.parse(reader);(P/'read_Merve_verified179_v3_server_book_for_local_CAS_20261117_remote.py').write_bytes(reader.encode())
sync=(P/'sync_Merve_verified179_v2_local_exact_zip_CAS_20261117.py').read_text(encoding='utf8').replace('R_MervePaliCompact_verified179_v2_server_book_for_local_CAS_20261117.json','R_MervePaliCompact_verified179_v3_server_book_for_local_CAS_20261117.json');ast.parse(sync);(P/'sync_Merve_verified179_v3_local_exact_zip_CAS_20261117.py').write_bytes(sync.encode())
print(json.dumps(dict(publisher=pn,publisher_SHA256=ph,test_template_SHA256=th,supervisor_SHA256=sh,reviewed_inputs_SHA256=qh,only_negative_test_fixture_changed_and_new_isolated_scopes=True,no_model_or_D_or_actual_score_or_shared_writer_change=True)))

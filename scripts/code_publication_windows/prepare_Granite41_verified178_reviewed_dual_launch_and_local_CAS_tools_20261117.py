import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json,hashlib
P=Path(__file__).parent
s=(P/'prepare_Deadbydawn_verified177_reviewed_dual_publication_launch_and_local_CAS_tools_20261117.py').read_text()
s=s.replace('DeadbydawnGemma4E2B','Granite41').replace('verified177','verified178').replace('Deadbydawn_verified177','Granite41_verified178').replace('R_Deadbydawn_v3_full_evidence_for_publication_final_20261117.json','R_Granite41_completed_full_evidence_verified178_publication_20261117.json')
s=s.replace('publish_Granite41_verified178_single_original_lock_dual_transaction_v2_finish_reference_scope_20261117.py','publish_Granite41_verified178_single_original_lock_dual_transaction_v1_20261117.py').replace('cda1648ff5cd99a7945678eb31f764ce690f338696ef73b751f9bf6e23fc06c9','39819715c06c32954ae27e7a4c50b04205aed4bd5c239842d14202e264045069')
s=s.replace('before_239.png','before_364.png').replace('after_239.png','after_364.png').replace('G239','G364').replace('run_Granite41_verified178_dual_publication_actual_completion_v2','run_Granite41_verified178_dual_publication_actual_completion_v1')
# Generate from the actual reviewed previous sync tools, then bind every input
# and count to this new transaction. No generic production workbook rewrite.
pos=s.index("# All local mutations below")
s=s[:pos]+'''sync=(P/'sync_Deadbydawn_verified177_local_exact_zip_CAS_20261117.py').read_text().replace('DeadbydawnGemma4E2B','Granite41').replace('Deadbydawn_verified177','Granite41_verified178').replace('verified177','verified178').replace('G239','G364').replace("['done']==177","['done']==178").replace('all_prior176','all_prior177')
ast.parse(sync);(P/'sync_Granite41_verified178_local_exact_zip_CAS_20261117.py').write_bytes(sync.encode())
reader=(P/'read_Deadbydawn_verified177_server_book_for_local_CAS_20261117_remote.py').read_text().replace('DeadbydawnGemma4E2B','Granite41').replace('verified177','verified178').replace("['done']==177","['done']==178").replace('deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit','ibm-granite/granite-vision-4.1-4b')
ast.parse(reader);(P/'read_Granite41_verified178_server_book_for_local_CAS_20261117_remote.py').write_bytes(reader.encode())
print(json.dumps(dict(reviewed_inputs_SHA256=qh,publisher_SHA256=ph,supervisor_SHA256=sh,all_tools_prepared_only_no_production_write=True)))
'''
assert 'R_Deadbydawn_v3' not in s;ast.parse(s);exec(compile(s,str(P/'prepare_Granite41_verified178_reviewed_actual_generated_20261117.py'),'exec'),{'__file__':__file__,'__name__':'__main__'})

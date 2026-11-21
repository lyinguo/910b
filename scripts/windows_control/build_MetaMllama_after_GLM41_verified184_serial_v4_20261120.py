from pathlib import Path
import ast,copy,difflib,hashlib,json
V=Path(__file__).resolve().parent
parent=V/'continue_MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.py'
raw=parent.read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(raw)=='e14a7f393ee6b357862925fc2abc9673618cd9285e52576d3e2469e707059391'
s=raw.decode()
name='continue_MetaMllama_completed_genuine3_after_GLM41_verified184_dual_release_shaped_storage_v4_serial_20261120.py'
cfgname='MetaMllama_completed_genuine3_after_GLM41_verified184_dual_release_shaped_storage_v4_serial_20261120.json'
replacements={
 'scripts/'+parent.name:'scripts/'+name,
 'configs/queues/MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.json':'configs/queues/'+cfgname,
 'configs/protocols/meta_mllama_completed_genuine3_shaped_storage_LZMA_v3_approved_first200_serial_20261120.json':'configs/protocols/meta_mllama_completed_genuine3_after_GLM41_verified184_shaped_storage_v4_approved_first200_serial_20261120.json',
 'configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct-own-original-SDPA-BF16-2048-complete-shaped-LZMA-v3-approved-first200-serial-20261120.yaml':'configs/MllamaForConditionalGeneration/meta-llama--Llama-3.2-11B-Vision-Instruct-own-original-SDPA-BF16-2048-complete-shaped-LZMA-v4-after-GLM41-verified184-approved-first200-serial-20261120.yaml',
 "H=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120'":"H=B/'MetaMllama_completed_genuine3_after_GLM41_verified184_dual_release_shaped_storage_v4_serial_20261120'",
 "J=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'":"J=B/'MetaMllama_completed_genuine3_after_GLM41_verified184_shaped_storage_v4_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'"}
for a,b in replacements.items():assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
s=s.replace("cfg=js(R/CONFIG);assert cfg['model_id']==MID and cfg['max_concurrent_models']==1", "cfg=js(R/CONFIG);assert cfg['model_id']==MID and cfg['max_concurrent_models']==1 and cfg['after_qualified_GLM41_verified184_dual_publication_only'] is True")
helper='''
PUB=B/'GLM41Thinking_original_VLLM_BF16_8192_LZMA_verified184_blank_G_publication_20261120'
RESERVE=B/'GLM41Thinking_verified184_reviewed_publication_window_v2_actual_runner_directory_after_same_formal200_original_EX_20261120'
OLDWAIT=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_wait_actual_capacity_first200_serial_20261120'
OLDJ=B/'MetaMllama_completed_genuine3_shaped_storage_LZMA_v3_actual_capacity_approved_NEWfresh3_fixed200_serial_real_FLOCK_20261120'

def verify_authorized_GLM41_dual_transition(original_job):
 from A100_alignment_evidence_runner_20261103 import unchanged
 audit=js(PUB/'full_target_dual_publication_audit.json');pub=js(PUB/'publication_receipt.json');dual=js(PUB/'dual_publication_completed.json')
 assert audit['passed'] and dual['completed'] and pub['completed'] and pub['original_protocol_final_qualification_complete']
 assert audit['canonical_counts']=={'done':184,'needs_adaptation':206,'skipped_no_complete_weights':310}
 assert audit['numeric_within5pct']==169 and audit['numeric_outside5pct']==15 and audit['qualified_delivered']==70
 assert audit['old183_state_registry_G_C_B_D_preserved'] and audit['all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved']
 target,=pub['targets'];assert target['model_id']=='zai-org/GLM-4.1V-9B-Thinking' and target['workbook_write']['row']==696
 assert sha(target['formal_proof'])==target['formal_proof_sha256'];pf=js(target['formal_proof'])
 assert pf['passed'] and pf['sample_count']==200 and pf['original_protocol_final_qualification_complete'] and pf['formal_alignment_within_5pct']
 assert set(pf['relative_error_to_current_A100_D'])=={'server','local','preserved_original'}
 from decimal import Decimal
 assert all(Decimal(x)<=Decimal('.05') for x in pf['relative_error_to_current_A100_D'].values())
 before=original_job['protected_canonical_hashes'];after=dict(before);allowed={str(D/'state.json'),str(D/'registry.json'),str(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json'),str(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'))}
 assert set(audit['canonical_SHA256'])==allowed
 for p,h in before.items():
  if p in allowed:
   assert h==(sha(PUB/'before_local_binding_verbatim.json') if p==str(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json') else pub['canonical_before_sha256'][p])
   assert sha(p)==audit['canonical_SHA256'][p];after[p]=sha(p)
  else:assert sha(p)==h,(p,'Unrelated canonical change')
 book=str(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'))
 assert before[book]=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81' and dual['local_and_server_workbook_sha256']==after[book]
 unchanged(after)
 put(H/'authorized_GLM41_only_G696_dual_publication_transition_review.json',dict(passed=True,original_true3_job_SHA256=sha(J3/'job.json'),original_protected_canonical_SHA256=before,current_protected_canonical_SHA256=after,publication_receipt=str(PUB/'publication_receipt.json'),publication_receipt_SHA256=sha(PUB/'publication_receipt.json'),full_dual_audit=str(PUB/'full_target_dual_publication_audit.json'),full_dual_audit_SHA256=sha(PUB/'full_target_dual_publication_audit.json'),unchanged_all_original_Meta_D_and_inputs_dtype_budget_math=True,only_four_authorized_canonical_hashes_rebound=True,no_NN_replay_or_original_source_edit=True))
 return after

def wait_exact_GLM41_publication_and_old_noNN_waiter_exit():
 while not (PUB/'full_target_dual_publication_audit.json').is_file() or not (RESERVE/'actual_completion.json').is_file():
  if (RESERVE/'actual_completion.json').is_file():
   r=js(RESERVE/'actual_completion.json')
   if r.get('phase')=='original200_measurement_not_qualified_no_publication_lock_released':
    put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=0,phase='GLM41_not_published_original_Meta_queue_keeps_original183_binding',actual_NEW_model_forwards=0,no_G_or_canonical_write=True));return False
   assert r.get('actual_returncode')==0,('GLM41 publication reservation failed',r)
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_GLM41_full200_dual_publication_not_NN',actual_NEW_model_forwards=0,max_concurrent_models=1));time.sleep(2)
 assert js(RESERVE/'actual_completion.json')['actual_returncode']==0
 while not (OLDWAIT/'actual_completion.json').is_file() or (Path('/proc')/'1620589').exists():
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_original_Meta_noNN_waiter_natural_reference_guard_exit',actual_NEW_model_forwards=0,max_concurrent_models=1));time.sleep(2)
 old=js(OLDWAIT/'actual_completion.json')
 assert old['actual_returncode']==1 and not OLDJ.exists()
 assert 'unchanged' in old['error'] and any(str(D/n) in old['error'] for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json'])
 return True

'''
assert s.count('def main():')==1;s=s.replace('def main():',helper+'def main():')
needle="assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'"
assert s.count(needle)==1;s=s.replace(needle,needle+"\n if not wait_exact_GLM41_publication_and_old_noNN_waiter_exit():return 0")
needle="job=js(J3/'job.json');unchanged(job['protected_canonical_hashes']);assert quiet()"
assert s.count(needle)==1;s=s.replace(needle,"job=js(J3/'job.json');job['protected_canonical_hashes']=verify_authorized_GLM41_dual_transition(job);unchanged(job['protected_canonical_hashes']);assert quiet()")
needle="p=js(oldproto);p2=copy.deepcopy(p)"
assert s.count(needle)==1;s=s.replace(needle,needle+"\n assert p['reference_workbook_SHA256']=='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'\n p2['reference_workbook_SHA256']=job['protected_canonical_hashes'][p['reference_workbook']]\n transition=H/'authorized_GLM41_only_G696_dual_publication_transition_review.json'\n p2['static_evidence_SHA256'][str(transition)]=sha(transition)")
ast.parse(s);(V/name).write_text(s,encoding='utf-8',newline='\n')
cfg=json.loads((V/'MetaMllama_completed_genuine3_shaped_storage_capacity_wait_v3_first200_serial_20261120.json').read_bytes())
old='scripts/'+parent.name;assert cfg['exact_source_SHA256'].pop(old)=='e14a7f393ee6b357862925fc2abc9673618cd9285e52576d3e2469e707059391'
cfg['exact_source_SHA256']['scripts/'+name]=sha(s.encode())
cfg['after_qualified_GLM41_verified184_dual_publication_only']=True
(V/cfgname).write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
(V/'MetaMllama_after_GLM41_verified184_v4_full_parent_diff_20261120.diff').write_text(''.join(difflib.unified_diff(raw.decode().splitlines(keepends=True),s.splitlines(keepends=True),fromfile=parent.name,tofile=name)),encoding='utf-8')
print(json.dumps(dict(source=name,SHA256=sha(s.encode()),config=cfgname,config_SHA256=sha((V/cfgname).read_bytes()),same_MODEL_NF_UTILITY_and_generation_fields=True,only_legal_GLM41_four_hash_transition_and_private_stage_names_added=True,no_NN_or_G_started=True)))

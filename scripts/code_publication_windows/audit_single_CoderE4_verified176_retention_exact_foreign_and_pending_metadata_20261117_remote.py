import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117';I=B/'single_CoderE4_verified176_retention_for_Deadbydawn_exact_inputs_actual_plan_launch_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
plan_sha='c4ea74b93a90e4f2246caba2590556ea46e86c7ffa2a4ea5609e74781f2b3ddb';assert sha(H/'plan.json')==plan_sha
pub=B/'CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117';assert js(pub/'full_target_dual_publication_audit.json')['passed'] and js(pub/'dual_publication_completed.json')['completed']
tool=R/'scripts/audit_single_CoderE4_retention_all_verified176_foreign_metadata_20261117.py';assert sha(tool)=='b72158a7a9edda8d562a78b542139e62780070e94462198ebffde83b8cce257d'
with (I/'foreign_audit_stdout.json').open('xb') as out,(I/'foreign_audit_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',str(tool),plan_sha],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
assert p.returncode==0,(I/'foreign_audit_stderr.log').read_text()
plan=js(H/'plan.json');candidates={x['path'] for x in plan['remove']};documents=[];matches=[]
def strings(v):
 if isinstance(v,str):yield v
 elif isinstance(v,dict):
  for k,x in v.items():yield str(k);yield from strings(x)
 elif isinstance(v,list):
  for x in v:yield from strings(x)
for folder in [B/'DonutInvoice_existing_complete200_v6_present_raw_full_numeric_independent_diagnostic_real_FLOCK_20261117',B/'DeadbydawnGemma4E2B_own_original_BF16_true3_v2_own_constructor_binding_real_FLOCK_20261117',B/'PrithivGemma4E4B_own_original_BF16_true3_v2_own_config_real_FLOCK_20261116']:
 for f in folder.rglob('*.json'):
  if not f.is_file() or f.stat().st_size>8*1024*1024:continue
  v=js(f);documents.append(dict(path=str(f),SHA256=sha(f)));matches.extend(dict(document=str(f),candidate=s) for s in strings(v) if s in candidates)
assert not matches,matches;review=H/'all_current_foreign_evidence_metadata_review.json';v=js(review);assert v['passed'] and not v['foreign_candidate_or_unresolved_home_references'];v['additional_current_pending_control_and_new_numeric_review_metadata_checked']=documents;v['additional_pending_foreign_candidate_references']=matches;review.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
result=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,plan_SHA256=plan_sha,foreign_review_SHA256=sha(review),current_accepted_models_checked=v['current_completed_models_checked'],metadata_document_count=len(v['active_report_lineage_config_and_registered_or_legacy_proof_metadata_checked']),additional_pending_documents=len(documents),foreign_references=[],no_arrays_deleted=True);(I/'foreign_audit_actual_completion.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps(result,ensure_ascii=False))

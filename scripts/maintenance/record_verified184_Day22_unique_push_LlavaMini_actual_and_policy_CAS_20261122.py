"""Preserve exact code receipt/source transition and current serial model state."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
def js(path):return json.loads(Path(path).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()));temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(temp,path)
def main():
    plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan)
    pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(before)
    H=Path(ptr['handoff_directory']);parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_pending_SHA256']
    canonical=dict(ptr['canonical_SHA256']);assert all(sha(path)==digest for path,digest in canonical.items())
    rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert all(sha(path)==row['SHA256'] for path,row in rows.items())
    receipt=q['code_receipt'];assert receipt['passed'] and receipt['status']=='pushed_and_remote_verified' and receipt['day']=='2026-11-22' and receipt['commit']==receipt['remote_commit']=='43a149bb69510288451c2348159d6e7baa0d8d8e' and receipt['one_new_Code_Day22_commit'] and receipt['reviewed_plan_SHA256']=='93b931d17235cbeeac79d32e620eb6fb31434f4d63ce6e58d3410ccbf8a447e0'
    policy=R/'configs/publication/code_only.json';old=policy.read_bytes();new=base64.b64decode(q['code_policy_after_base64'],validate=True)
    assert hashlib.sha256(old).hexdigest()==receipt['policy_before_SHA256']=='c8a705028535eace9acb88fb60fbe45bf6afa3b974a9a3566201d44698786f05'
    assert hashlib.sha256(new).hexdigest()==receipt['policy_after_SHA256']=='39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5'
    a=json.loads(old);b=json.loads(new);assert len(b['files'])==2670 and set(a['files'])<=set(b['files']) and a['repository']==b['repository']=='https://github.com/lyinguo/910b.git' and a['branch']==b['branch']=='codex/910b-source-layout-20261114'
    assert sha(R/'scripts/publish_code_daily.py')=='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
    job_home=B/'LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122';assert sha(job_home/'job.json')=='99994d88480b6b9fef627de652cbf1837b05080650389d5243676e3a2787f9b7';job=js(job_home/'job.json')
    assert 'configs/publication/code_only.json' not in job['spec']['source_sha256'] and str(policy) not in job['protected_canonical_hashes'] and all(sha(R/name)==digest for name,digest in job['spec']['source_sha256'].items())
    out=C/'daily_20261122_actual_reviewed_commit_remote_receipts';out.mkdir(exist_ok=False);(out/'previous_reviewed_code_policy.json').write_bytes(old)
    note=H/'late_actual_notes'/q['round_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
    for row in q['local_original_records']:
        data=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(data).hexdigest()==row['SHA256'];path=note/'local_original_records'/row['name'];assert path.resolve().is_relative_to(note.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        if row['name'].startswith('daily_20261122_'):
            dest=out/row['name'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    put(out/'daily_20261122_receipt.json',receipt)
    assert pointer.read_bytes()==before and policy.read_bytes()==old and all(sha(path)==digest for path,digest in canonical.items())
    temp=policy.with_name(policy.name+'.tmp.'+str(os.getpid()));temp.write_bytes(new);assert policy.read_bytes()==old;os.replace(temp,policy)
    primary=list(q['new_primary_paths'])+[str(Path(__file__).resolve().relative_to(R)),'configs/publication/code_only.json']
    for name in ['LlavaMini_own_original_FP16_2048_true3_v2_actual_reserved_devices_real_FLOCK_20261122','LlavaMini_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v2_actual_reserved_devices_real_FLOCK_20261122']:
        path=B/name/'job.json'
        if path.is_file():primary+=list(js(path)['spec']['source_sha256'])
    for name in sorted(set(primary)):
        path=R/name;assert path.is_file() and path.resolve().is_relative_to(R) and (name=='run.py' or name.startswith(('scripts/','configs/','universal_eval/','tests/')))
        digest=sha(path)
        if str(path) in rows and name!='configs/publication/code_only.json':assert rows[str(path)]['SHA256']==digest,name
        if str(path) not in rows or rows[str(path)]['SHA256']!=digest:
            if str(path) in rows:assert name=='configs/publication/code_only.json' and rows[str(path)]['SHA256']==receipt['policy_before_SHA256'] and digest==receipt['policy_after_SHA256']
            dest=note/'primary_source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());rows[str(path)]=dict(path=str(path),relative=name,SHA256=digest,bytes=path.stat().st_size,snapshot=str(dest),role='Own serial adaptation/complete numeric verification/true consumer configuration/test/read-only real status or reviewed publication transition source')
    reader=R/'scripts/maintenance/read_LlavaMini_current_serial_v4_relative_verifier_FD_actual_20261122_remote.py';result=subprocess.run(['python3','-B',str(reader)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True);assert result.returncode==0,result.stderr.decode(errors='replace');actual=json.loads(result.stdout)
    state=js(D/'state.json');counts={name:sum(row['status']==name for row in state['models'].values()) for name in ['done','needs_adaptation','skipped_no_complete_weights']};assert counts==dict(done=184,needs_adaptation=206,skipped_no_complete_weights=310)
    processes=actual['processes'];fullNF=any(any(Path(arg).name.startswith('verify_llava_mini_') for arg in p['argv']) for p in processes);model=any(any(Path(arg).name=='run.py' for arg in p['argv']) for p in processes)
    actual.update(canonical_SHA256=canonical,canonical_counts=counts,current_model_process_active=model,current_full_numerical_verifier_active=fullNF,model_process_does_not_imply_continuously_submitted_NPU_kernels=True,code_receipt=str(out/'daily_20261122_receipt.json'),code_receipt_SHA256=sha(out/'daily_20261122_receipt.json'),code_policy_exact_transition=dict(before=receipt['policy_before_SHA256'],after=receipt['policy_after_SHA256']))
    put(note/'current_serial_fullNF_and_publication_actual.json',actual)
    full=q['full_handoff_template'].replace('CURRENT_ACTUAL_JSON',json.dumps(actual,ensure_ascii=False)).replace('CURRENT_SOURCE_COUNT',str(len(rows)))
    fullpath=note/q['full_note_name'];fullpath.write_text(full,encoding='utf8')
    pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=now(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit=receipt['commit'],no_second_same_day_commit=True,reviewed_code_policy_exact_transition=dict(before=receipt['policy_before_SHA256'],after=receipt['policy_after_SHA256'],receipt=str(out/'daily_20261122_receipt.json'),receipt_SHA256=sha(out/'daily_20261122_receipt.json')),next_daily_review='Day23 every useful postcutoff source/actual config consumer/test/Windows control/dependency/fresh SHA/full diff/outgoing/appropriate tests/publisher actual reviewed planSHA; no empty commit'))
    updates=dict(UTC=now(),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_primary_source_count=len(rows),latest_actual_note=str(fullpath),latest_actual_note_SHA256=sha(fullpath),latest_full_actual_summary=str(fullpath),latest_full_actual_summary_SHA256=sha(fullpath),latest_actual_snapshot=str(note/'current_serial_fullNF_and_publication_actual.json'),latest_actual_snapshot_SHA256=sha(note/'current_serial_fullNF_and_publication_actual.json'),current_actual_stage='verified184_Day22_unique_code_pushed_own_LlavaMini_true3_or_fullNF_serial_see_real_actual',current_formal_model_stage='Own FP16 LLaVA-Mini genuine3/6 completed; original full numerical acceptance and actual capacity/exit supersede this snapshot',current_verified_count=184,numeric_within5pct=169,numeric_outside5pct=15,complete_weights_pending_formal=206,confirmed_missing=310,complete_original_protocol_qualified=70,current_model_NN_or_NF_active=(model or fullNF),current_actual_model_NN_or_NF_active=(model or fullNF),current_NN_or_NF_active=(model or fullNF),current_actual_model_process_or_NF_active=(model or fullNF),current_full_numerical_verifier_active=fullNF,current_model_process_active=model,current_stop_earliest_proven_Beijing=None if model or fullNF else 'Unknown: inspect actual controller completion and latest full numerical exit',last_formal_recovery_Beijing='2026-11-22T01:13:50+08:00',code_daily_unique_commit=receipt['commit'],code_daily_day='2026-11-22',code_daily_receipt=str(out/'daily_20261122_receipt.json'),code_daily_receipt_SHA256=sha(out/'daily_20261122_receipt.json'),reviewed_code_publication_policy_before_SHA256=receipt['policy_before_SHA256'],reviewed_code_publication_policy_after_SHA256=receipt['policy_after_SHA256'],current_retention_ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'))
    assert pointer.read_bytes()==before and all(sha(path)==digest for path,digest in canonical.items()) and all(sha(path)==row['SHA256'] for path,row in rows.items()) and all(sha(R/name)==digest for name,digest in job['spec']['source_sha256'].items())
    closing=js(H/'closing_addendum.json');closing.update(updates);put(H/'closing_addendum.json',closing);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
    final=dict(passed=True,commit=receipt['commit'],receipt=str(out/'daily_20261122_receipt.json'),receipt_SHA256=sha(out/'daily_20261122_receipt.json'),pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fullpath),full_actual_note_SHA256=sha(fullpath),actual_snapshot=str(note/'current_serial_fullNF_and_publication_actual.json'),current_model_process_active=model,current_full_numerical_verifier_active=fullNF,all_model_jobs_frozen_sources_canonical_workbooks_state_reg_retention_unchanged=True,only_reviewed_real_code_publication_consumer_policy_and_evidence_handoff_updated=True)
    put(out/'actual_R_policy_and_handoff_receipt.json',final);put(note/'round_preservation_receipt.json',final);print(json.dumps(final,ensure_ascii=False))
if __name__=='__main__':main()

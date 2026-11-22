"""Preserve official readback and its useful Windows controls; no live TOML edit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,os,tomllib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()=='bdcb6b689782b72e0f1d898d9d29d8ca9fd06008198bc2bb045fbdaa7c337ba5';ptr=json.loads(before);H=Path(ptr['handoff_directory'])
 parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']=='f1045e334164597c9eaea670457adeddb1b1ccc401f224ac3396d8f363629e45';assert ptr['latest_full_actual_summary_SHA256']=='18adf22645bebf8df7661e48109b8b80d51918e503173953ea30f7086f850712'
 canonical=ptr['canonical_SHA256'];assert all(sha(p)==h for p,h in canonical.items())
 note=H/'late_actual_notes/verified184_Gemma4MLX8_v74_official_actual_readback_v75_20261122';note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes());values={}
 for row in q['records']:
  raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256'];path=note/row['name'];assert path.resolve().is_relative_to(note.resolve());path.write_bytes(raw);values[row['name']]=raw
 actual=tomllib.loads(values['official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.toml'].decode());requested=json.loads(values['official_910b_verified184_Llava_completed_Gemma4MLX8_serial_v74_requested_args_20261122.json']);receipt=json.loads(values['official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.json'])
 assert receipt['passed'] and receipt['official_tool_only_no_live_TOML_direct_edit'] and receipt['actual_TOML_SHA256']=='56e1915ca6d6a0c5c5840ccc59a205920cd776ec0a6b867c00a1080dfae8f935' and receipt['actual_prompt_SHA256']=='2e57580f8f82430b866933bc99e60445302ea5ccdfcca84086e24bc05ff67b80'
 assert actual['prompt']==requested['prompt'] and hashlib.sha256(actual['prompt'].encode()).hexdigest()==receipt['actual_prompt_SHA256'] and actual['id']==requested['id']=='910b' and actual['kind']=='heartbeat' and actual['status']=='ACTIVE' and actual['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300' and actual['name']=='跟进910B模型评测与规范检查' and actual['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
 data=js(parent);rows={r['path']:r for r in data['all_current_primary_source_and_true_consumer_configs']}
 for name in q['new_primary_paths']+[str(Path(__file__).relative_to(R))]:
  p=R/name;assert p.resolve().is_relative_to(R) and name.startswith('scripts/') and p.is_file();digest=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==digest
  else:
   dest=note/'primary_source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=name,SHA256=digest,bytes=p.stat().st_size,snapshot=str(dest),role='Official existing heartbeat builder/exact actual readback/preservation true consumer control source')
 assert all(sha(p)==r['SHA256'] for p,r in rows.items()) and all(sha(p)==h for p,h in canonical.items()) and pointer.read_bytes()==before
 pending=C/'pending_after_daily_20261122_verified184_Gemma4MLX8_serial_official_readback_all_primary_v75_20261122.json';assert not pending.exists();data.update(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[p] for p in sorted(rows)],official_910b_actual_readback=str(note/'official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.json'));put(pending,data)
 update=dict(UTC=datetime.now(timezone.utc).isoformat(),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_primary_source_count=len(rows),official_910b_latest_actual_readback=str(note/'official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.json'),official_910b_latest_actual_readback_SHA256=sha(note/'official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.json'),official_910b_latest_actual_TOML_SHA256=receipt['actual_TOML_SHA256'],official_910b_latest_actual_prompt_SHA256=receipt['actual_prompt_SHA256'],official_910b_ACTIVE_original_name_thread_300minute_preserved=True)
 closing=js(H/'closing_addendum.json');closing.update(update);put(H/'closing_addendum.json',closing);ptr.update(update);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 result=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),official_actual_readback=str(note/'official_910b_verified184_Gemma4MLX8_v74_actual_readback_20261122.json'),all_evaluation_model_jobs_workbooks_state_reg_policies_retention_unchanged=True,latest_full_actual_summary_unchanged=ptr['latest_full_actual_summary'],latest_full_actual_summary_SHA256=ptr['latest_full_actual_summary_SHA256']);put(note/'official_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

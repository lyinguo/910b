import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated';J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119';H=B/'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v4_approval_schema_20261119'
js=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maybe(p):
 if not Path(p).is_file():return None
 try:return js(p)
 except ValueError:return dict(live_partial_JSON=True)
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from batch_910b_support import workbook_rows
job=js(J/'job.json');assert sha(J/'job.json')=='a1bf90557a326523990a6004396e2dbe336e85d72139b216fe00f5337340c0fd'
done=maybe(J/'completion.json');controller=maybe(H/'actual_completion.json');procs=project_processes();locks=runner_lock_owners();control=Path('/proc/1565741')
controller_alive=control.exists() and 'continue_huihui_completed3_verified4972_lossless_storage_serial_fresh3_fixed200_v4_approval_schema_20261119.py' in (control/'cmdline').read_bytes().replace(b'\0',b' ').decode()
out=dict(UTC=datetime.now(timezone.utc).isoformat(),job=str(J),controller=str(H),job_SHA256=sha(J/'job.json'),job_metadata=job,completion=done,controller_completion=controller,actual_procs=procs,FLOCK=locks,own_controller_still_alive=controller_alive,ready_for_publication_review=False)
if done and not done.get('live_partial_JSON'):
 accepted=done.get('accepted',[])
 if accepted and accepted[-1]['sample_count']==200:
  run=accepted[-1];assert sha(run['proof'])==run['proof_sha256'];out.update(formal_proof=js(run['proof']),formal_proof_SHA256=sha(run['proof']))
  approval=js(job['spec']['alignment_approval_review']);out.update(approval=approval,capacity_review=js(approval['capacity_review']),true3_proof=js(approval['true3_proof']),true3_proof_SHA256=sha(approval['true3_proof']))
  book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,sheet,_=workbook_rows(book);row,=[x for x in rows if x['model_id']==MID];ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  with zipfile.ZipFile(book) as z:row['current_D']=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
  prior=B/'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117';out.update(current_row=row,prior180_export_SHA256=sha(prior/'verified180_export_complete_metadata.json'),prior180_stat_SHA256=sha(prior/'active180_full_report_stat_baseline.json'))
  out['ready_for_publication_review']=bool(controller and controller.get('actual_returncode')==0 and done['actual_returncode']==done['verifier_returncode']==0 and done['model_returncode'] in [0,1] and not controller_alive and not locks and not [p for p in procs if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']])
print(json.dumps(out,ensure_ascii=False))

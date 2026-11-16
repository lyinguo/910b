import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'OSAtlas4B_original_probability_verified170_blank_G_publication_20261116';O=B/'GLMOCR_pair_common_baseline_verified169_G_publication_20261116'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=js(H/'full_target_dual_publication_audit.json');assert audit['passed'];ex=js(H/'verified170_export.json');assert ex['measured_count']==len(ex['records'])==170 and ex['numeric_within5pct']==155
stats={};old=js(O/'active169_full_report_stat_baseline.json')['reports'];changes=[]
for rec in ex['records']:
 stats[rec['model_id']]={}
 for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']:
  p=Path(rec['accepted_report'])/n;s=p.stat();v=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns];stats[rec['model_id']][n]=v
  if rec['model_id'] in old and list(old[rec['model_id']][n])!=v:changes.append([rec['model_id'],n])
assert not changes and len(old)==169
ep=H/'verified170_export_complete_metadata.json';sp=H/'active170_full_report_stat_baseline.json';assert not ep.exists() and not sp.exists()
ex.update(report_stat_baseline=str(sp),target_full_audit=dict(path=str(H/'full_target_dual_publication_audit.json'),SHA256=sha(H/'full_target_dual_publication_audit.json')),server_book_SHA256=audit['local_and_server_book_SHA256'],current_original_A100_protocol_qualified=56,original_protocol_qualified=56,blank_count=530,missing_complete_weights=310)
ep.write_text(json.dumps(ex,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');sp.write_text(json.dumps(dict(export_SHA256=sha(ep),reports=stats,prior169_report_stats_exactly_unchanged=True,stat_only_cache_invalidation_not_new_acceptance=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
F=B/'FalconOCR_own_original512_source_dtype_CPU_frontend200_v2_json_set_recording_20261116';f={}
for n in ['actual_completion.json','review.json','progress.json','own_source_and_saved_loading_dtype_diagnostic.json']:
 p=F/n
 if p.exists():
  q=js(p);f[n]=q if n!='own_source_and_saved_loading_dtype_diagnostic.json' else {k:v for k,v in q.items() if k!='source'}
print(json.dumps(dict(export=ex,export_path=str(ep),export_SHA256=sha(ep),stat_baseline=str(sp),stat_baseline_SHA256=sha(sp),old169_stats_all_unchanged=True,Falcon_CPU_frontend=f),ensure_ascii=False))

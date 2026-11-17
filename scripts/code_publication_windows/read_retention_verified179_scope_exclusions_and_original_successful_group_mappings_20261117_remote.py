import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,hashlib
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/authorized_verified179_accepted_current_nonexample_retention_user_continue_Prithiv_20261117'
sys.path.insert(0,str(R/'scripts'))
import retain_verified179_accepted_current_nonexample_matrices_user_request_20261117 as m
cfg=m.js(R/'configs/retention/verified179_accepted_nonexample_matrices_for_Prithiv_user_request_20261117.json');state=m.js(D/'state.json');reg=m.js(D/'registry.json');rows=[]
for mid in cfg['allowed_model_ids']:
 e=state['models'][mid];p=m.proof_for(mid,e,reg);q=m.js(p);ex=m.js(Path(e['report_dir'])/'operator_lineage.json').get('actual_execution',{});home=m.own_home(ex);groups,trace=m.sample_group_index(ex);paths=[(Path(n),h) for n,h in q.get('external_evidence_sha256',{}).items() if n.endswith('.npz') and home and Path(n).is_relative_to(home)]
 rows.append(dict(model_id=mid,proof=str(p),passed=q.get('passed'),count=q.get('sample_count'),historical=q.get('historical_record_reuse'),error_indices=q.get('error_indices'),home=str(home),report_changed=[n for n,h in q['report_sha256'].items() if m.sha(Path(e['report_dir'])/n)!=h],groups_count=len(groups),path_count=len(paths),path_examples=[dict(name=str(p.relative_to(home)),exists=p.exists(),indices=sorted(m.indices_for(p.relative_to(home),groups) or [])) for p,h in paths[:3]],missing_series=[str(p) for p,h in paths if not p.exists() and m.indices_for(p.relative_to(home),groups)],trace=str(trace)))
print(json.dumps(dict(skipped=m.js(H/'discovery.json')['skipped'],rows=rows),ensure_ascii=False))

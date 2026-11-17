import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,re
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';S=R/'workspace/diagnostics/model_runs';js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();state=js(D/'state.json');reg=js(D/'registry.json');scopes=[];skipped=[]
keys=['evidence_root','per_sample_evidence','sample_trace','weight_evidence','weight_loading_evidence','full_numeric_group_index','final_trace_manifest','runtime_evidence','actual_batch_manifest','execution_trace','trace_path']
for mid,e in state['models'].items():
 if e['status']!='done':continue
 rr=reg[mid];pf=rr.get('accepted_formal200_proof')
 if not pf:
  matches=[a for a in e.get('attempts',[]) if a.get('count')==200 and a.get('report_dir')==e['report_dir'] and isinstance(a.get('independent_verification'),str) and Path(a['independent_verification']).is_file()]
  if matches:pf=matches[-1]['independent_verification']
 if not pf or not Path(pf).is_file():skipped.append(dict(model_id=mid,reason='no_identified_current_accepted_full200_proof'));continue
 pp=js(pf)
 if not pp.get('passed') or pp.get('sample_count')!=200 or pp.get('historical_record_reuse') or pp.get('error_indices'):skipped.append(dict(model_id=mid,reason='not_current_successful_complete200_no_error_acceptance'));continue
 if rr.get('accepted_formal200_proof')==pf:assert sha(pf)==rr['accepted_formal200_proof_sha256']
 folder=Path(e['report_dir']);assert pp.get('report_dir',str(folder))==str(folder);line=js(folder/'operator_lineage.json').get('actual_execution',{});homes=[]
 for key in keys:
  name=line.get(key)
  if isinstance(name,str) and Path(name).is_absolute() and Path(name).is_relative_to(S):
   home=S/Path(name).relative_to(S).parts[0]
   if home not in homes:homes.append(home)
 if len(homes)!=1:skipped.append(dict(model_id=mid,reason='formal_owner_not_unambiguous_from_current_lineage',homes=list(map(str,homes))));continue
 home=homes[0];files=[]
 for name,h in pp.get('external_evidence_sha256',{}).items():
  p=Path(name)
  if p.suffix=='.npz' and p.is_relative_to(home) and p.exists():files.append(dict(path=name,SHA256=h,bytes=p.stat().st_size,relative=str(p.relative_to(home))))
 if not files:continue
 rels=[Path(f['relative']) for f in files];kind=None
 if all(len(p.parts)==1 and re.fullmatch(r'sample_\d{5}\.npz',p.name) for p in rels):kind='sample_serial'
 elif all(len(p.parts)==1 and re.fullmatch(r'full_native_group_\d{2}\.npz',p.name) for p in rels):kind='native_batch10_grouped'
 elif all(len(p.parts)==3 and p.parts[0]=='groups' and p.parts[1].isdigit() and p.name=='actual_visual_positions_masks_full_vocab.npz' for p in rels):kind='groups_native_serial_or_actual_groups'
 elif all(len(p.parts)==2 and p.parts[0] in ['full_vocab_and_visual','visual_features','native_samples'] and re.fullmatch(r'\d+\.npz',p.name) for p in rels):kind=rels[0].parts[0]
 else:
  # Non-series diagnostics are always excluded; candidate series is reviewed separately.
  counts=Counter((str(p.parent),p.name) for p in rels)
  skipped.append(dict(model_id=mid,reason='requires_exact_series_classifier_review',home=str(home),npz_files=len(files),npz_bytes=sum(f['bytes'] for f in files),relative_examples=[f['relative'] for f in files[:6]]));continue
 scopes.append(dict(model_id=mid,home=str(home),kind=kind,proof=str(pf),proof_SHA256=sha(pf),registry_proof_binding=rr.get('accepted_formal200_proof')==pf,report_dir=str(folder),existing_npz_count=len(files),existing_npz_bytes=sum(f['bytes'] for f in files),first_relative_files=[f['relative'] for f in files[:5]],series_files=files,lineage_scalar_paths={k:line[k] for k in keys if isinstance(line.get(k),str)},actual_group_count=pp.get('actual_generation_group_count'),actual_group_sizes=pp.get('actual_generation_group_sizes')))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),scopes=scopes,skipped=skipped,no_file_mutation_or_matrix_value_read=True),ensure_ascii=False))

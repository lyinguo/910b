"""Full new-run CANN/raw/source closure; reuse just-verified primary NPZ hashes, never inference."""
import pathlib,json,hashlib,datetime
R=pathlib.Path('/root/projects/comperror');D=R/'workspace/diagnostics'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def audit_raw_and_snapshots(report,primary):
 report=pathlib.Path(report).resolve();line=json.loads((report/'operator_lineage.json').read_text())['actual_execution'];root=pathlib.Path(line['weight_loading_evidence']).parent.resolve();assert root.is_relative_to(D/'model_runs')
 config=json.loads((report/'eval_report.json').read_text())['pipeline_config'];protocol_path=pathlib.Path(config['model']['kwargs']['protocol_file']);assert sha(protocol_path)==config['model']['kwargs']['protocol_sha256'];protocol=json.loads(protocol_path.read_text())
 items={};stats={};reused=0;fresh=0
 for path in sorted(root.rglob('*')):
  if not path.is_file() or path.name in ['independent_report_verification.json','independent_full_raw_and_source_manifest.json']:continue
  key=str(path);relative=path.relative_to(root).as_posix();st=path.stat()
  if key in primary:
   digest=primary[key];reused+=1
  else:
   digest=sha(path);fresh+=1
  if relative.startswith('groups/'):assert key in primary,'Uncovered primary sample evidence '+key
  assert len(digest)==64;items[key]=digest;stats[key]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
 assert items and any('export_only_prof_dir' in key or '/PROF_' in key for key in items),'Native raw CANN artifact tree missing'
 snapshot=root/'source_snapshot'
 assert sha(snapshot/'original_protocol_review.json')==config['model']['kwargs']['protocol_sha256']
 for source,digest in dict(protocol['runtime_sha256'],**protocol['historical_sources']).items():
  copy=snapshot/pathlib.Path(source).relative_to('/');assert items[str(copy)]==digest and sha(source)==digest
 for key,before in stats.items():
  st=pathlib.Path(key).stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==before,'Concurrent evidence mutation '+key
 result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'model_id':protocol['model_id'],'report':str(report),'evidence_root':str(root),'file_SHA256':items,'file_stat_at_validation':stats,'fresh_uncovered_raw_and_source_files_hashed':fresh,'primary_already_independently_verified_arrays_trace_and_group_hashes_reused':reused,'reuse_is_only_current_run_proof_evidence_not_inference_or_model_acceptance':True,'all_frozen_source_snapshots_equal_actual_sources':True,'all_raw_native_CANN_files_included_not_only_exported_chrome_trace':True}
 target=root/'independent_full_raw_and_source_manifest.json'
 if target.exists():
  previous=json.loads(target.read_text());assert previous['passed'] and previous['file_SHA256']==items and previous['file_stat_at_validation']==stats
 else:target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 return {'manifest':str(target),'manifest_SHA256':sha(target),'external_SHA256':items,'file_count':len(items),'newly_hashed':fresh,'primary_hashes_reused':reused}

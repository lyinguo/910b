"""Hash the new run's complete raw CANN and source tree without modifying historical data."""
import pathlib,json,hashlib,datetime

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def audit_all_new_raw_sources(root,previously_verified):
 root=pathlib.Path(root).resolve();assert root.is_relative_to('/root/projects/comperror/workspace/diagnostics/model_runs')
 manifest=root/'independent_all_raw_source_manifest.json'
 excluded={manifest}|{root/name for name in ['independent_report_verification.json','independent_report_verification_v2.json','independent_report_verification_v3.json']}
 files=sorted(q for q in root.rglob('*') if q.is_file() and q not in excluded)
 assert files and any('source_snapshot' in q.parts for q in files) and any('PROF_' in str(q) for q in files),'Full raw CANN/source closure required, not only Chrome JSON'
 records={}
 for q in files:
  st=q.stat();digest=previously_verified.get(str(q)) or sha(q)
  records[str(q)]={'SHA256':digest,'bytes':st.st_size,'stat':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]}
  after=q.stat();assert [after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns]==records[str(q)]['stat']
 if manifest.exists():
  existing=json.loads(manifest.read_text());assert existing['records']==records,'Changed completed evidence must be investigated, not silently overwritten'
 else:manifest.write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'new own GRMR run raw CANN and exact source snapshots only','records':records},ensure_ascii=False,indent=2)+'\n')
 return {'manifest':str(manifest),'manifest_SHA256':sha(manifest),'file_count':len(files),'all_file_sha256':{str(q):records[str(q)]['SHA256'] for q in files}}

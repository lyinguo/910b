import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shutil,os,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
o={'UTC':datetime.now(timezone.utc).isoformat(),'commands':{},'actual_procs':project_processes(),'actual_FLOCK':runner_lock_owners(),'device_nodes':[str(p) for p in Path('/dev').glob('davinci*')],'cpu_count':os.cpu_count(),'allowed_CPU_count':len(os.sched_getaffinity(0)),'memory_info':[l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))],'project_disk':dict(zip(['total','used','free'],shutil.disk_usage(D))),'instructions':{},'sources':{},'small_policies':{}}
for cmd in [['npu-smi','info'],['npu-smi','info','-l'],['npu-smi','info','-t','board','-i','4'],['df','-B1',str(D)],['findmnt','-T',str(D)],['npu-smi','info','--help']]:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);o['commands'][' '.join(cmd)]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for f in [Path('/AGENTS.md'),Path('/root/AGENTS.md'),Path('/root/projects/AGENTS.md'),R/'AGENTS.md',R/'scripts/AGENTS.md',R/'universal_eval/AGENTS.md',R/'universal_eval/model/AGENTS.md',R/'configs/AGENTS.md']:
 if f.exists():o['instructions'][str(f)]=f.read_text()
for n in ['scripts/A100_alignment_evidence_runner_20261103.py','scripts/batch_910b.py']:
 f=R/n;o['sources'][n]={'SHA256':sha(f),'text':f.read_text()}
assert o['sources']['scripts/A100_alignment_evidence_runner_20261103.py']['SHA256']=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
for f in [D/'PAUSE',D/'POLICY.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json']:o['small_policies'][str(f)]={'SHA256':sha(f),'value':js(f)}
home=B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117';done=js(home/'completion.json');line=js(Path(done['accepted'][0]['report_dir'])/'operator_lineage.json')['actual_execution'];idx=js(line['full_numeric_group_index']);dedup=[]
for ref in idx['groups']:
 g=js(ref['path']);refs={}
 def walk(v):
  if isinstance(v,dict):
   if {'array','shape','dtype','SHA256'}<=set(v):refs[v['array']]={k:v[k] for k in ['shape','dtype','SHA256']}
   for a in v.values():walk(a)
  elif isinstance(v,list):
   for a in v:walk(a)
 walk(g);unique={};counts={}
 with zipfile.ZipFile(g['archive']) as z:
  files=z.infolist();missing=[]
  for f in files:
   name=f.filename.removesuffix('.npy');v=refs.get(name)
   if not v:missing.append(name);continue
   key=(tuple(v['shape']),v['dtype'],v['SHA256']);unique.setdefault(key,f.file_size);counts[key]=counts.get(key,0)+1
  dedup.append(dict(group=g['group_index'],archive=g['archive'],archive_bytes=Path(g['archive']).stat().st_size,uncompressed_bytes=sum(f.file_size for f in files),member_count=len(files),all_referenced_array_count=len(refs),unreferenced_members=missing,unique_same_shape_dtype_full_value_SHA_count=len(unique),unique_uncompressed_bytes=sum(unique.values()),potential_exact_value_dedup_savings=sum(f.file_size for f in files)-sum(unique.values()),largest_repeated_arrays=sorted([dict(shape=list(k[0]),dtype=k[1],SHA256=k[2],one_member_bytes=unique[k],repetitions=n) for k,n in counts.items() if n>1],key=lambda v:v['one_member_bytes']*v['repetitions'],reverse=True)[:12],metadata_estimate_only_no_array_modified_or_deleted=True))
o['Merve_small3_readonly_lossless_storage_duplicate_inventory']=dedup
print(json.dumps(o,ensure_ascii=False))

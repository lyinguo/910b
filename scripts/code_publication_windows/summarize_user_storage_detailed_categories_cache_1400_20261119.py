import json
from pathlib import Path
P=Path(__file__).resolve().parent;a=json.loads((P/'R_user_storage_detailed_categories_cache_1355_20261119.json').read_bytes());b=json.loads((P/'R_user_actual_memory_disk_cache_attribution_1325_20261119.json').read_bytes())
def rows(s):
 r=[]
 for line in s.splitlines():
  x,p=line.split('\t',1);r.append((int(x),p))
 return sorted(r,reverse=True)
out={k:a[k] for k in ['UTC','project_realpath','project_path_is_symlink','diagnostic_categories','errors','uv_cache_runtime_process_refs','evidence_path','evidence_SHA256']}
out['top_du']={k:rows(v)[:35] for k,v in a['du_complete'].items()};out['large_files']=a['large_files'][:25]
out['cache_summary']={k:dict(files=len(a[k]),allocated_bytes=sum(z['allocated_bytes'] for z in a[k]),logical_bytes=sum(z.get('logical_bytes',z.get('bytes',0)) for z in a[k]),sample=a[k][:5]) for k in ['owned_python_bytecode_cache_files','temporary_suffix_files','package_HTTP_cache_files']}
out['other_dirs']={z['argv'][-1]:rows(z['stdout'])[:15] for z in b['read_only_inventory'] if z['argv'][0]=='du' and z['argv'][-1]!='/root/projects/comperror'}
print(json.dumps(out,ensure_ascii=False,indent=2))

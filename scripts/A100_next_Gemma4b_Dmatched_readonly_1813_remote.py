import hashlib,json,pathlib
A=pathlib.Path('/mnt/afs/comp_error');targets={'AiAsistent/gemma-3-4b-it-Cognitive-Liberty':.745,'CalmState/gemma-3-4b-polyglot-r1':.765,'heretic-org/gemma-3-4b-it-heretic':.8};out=[]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for mid,D in targets.items():
 root=A/'workspace/reports'/mid.replace('/','_');x={'model_id':mid,'local_D':D,'actual_A100_readonly_only':True,'report_candidates':[],'matching_reports':[],'actual_adapter_sources':{}}
 for f in sorted(root.glob('**/eval_report.json')):
  try:r=json.loads(f.read_text())
  except Exception:continue
  pc=r.get('pipeline_config',{});m=r.get('metrics',{});rows=r.get('detailed_results',[]);x['report_candidates'].append({'path':str(f),'metrics':m,'count':len(rows),'model':pc.get('model',{}),'algorithm':pc.get('algorithm',{})})
  if len(rows)==200 and any(isinstance(v,(float,int)) and abs(float(v)-D)<1e-12 for v in m.values()):
   log=f.parent/'run.log';x['matching_reports'].append({'path':str(f),'report_sha256':sha(f),'report':r,'run_log':log.read_text() if log.is_file() else None,'run_log_sha256':sha(log) if log.is_file() else None})
 for rec in x['matching_reports']:
  rel=rec['report']['pipeline_config']['model'].get('adapter','').rsplit(':',1)[0];p=A/rel
  if p.is_file() and p.is_relative_to(A):x['actual_adapter_sources'][rel]={'sha256':sha(p),'text':p.read_text()}
 out.append(x)
print(json.dumps(out,ensure_ascii=False))

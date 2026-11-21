"""Use the exact signed groups/0001 layout; retain both earlier write-before guards."""
from pathlib import Path
import ast,json,hashlib,base64
V=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
old='retain_verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_20261121.py';new='retain_verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_20261121.py';c0='verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_20261121.json';c1='verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_20261121.json'
s=(V/old).read_text(encoding='utf8');t=ast.parse(s);fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='exact_file');src=ast.get_source_segment(s,fn)
replacement=src.replace("p.parent == home and p.resolve() == p and (p.name == 'sample_%05d.npz' % row['index'])", "p == home / 'groups' / ('%04d' % (row['index']+1)) / 'actual_visual_positions_cache_full_vocab.npz' and p.resolve() == p")
assert replacement!=src;s=s.replace(src,replacement).replace('verified184_Meta_fullNF4_correct_nonexample117_v2_current_ledger_Bee_capacity_recovery_20261121','verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121').replace(c0,c1)
needle="helpers = base.old_helpers(cfg)\n        state, reg = (js(D / 'state.json'), js(D / 'registry.json'))"
assert needle in s
# Match plan/apply ownership maps for this separately completed baselinegap measurement.
needle2="foreign, _ = base.sources_and_dependencies(reg, accepted, {r['path']: r for r in p['remove']}, helpers)"
assert needle2 in s;s=s.replace(needle2,"accepted[MID] = dict(proof=p['formal_proof'], report_dir=js(p['formal_proof'])['report_dir'])\n        "+needle2)
needle3="sc, = [x for x in discovery['scopes'] if x['model_id'] == MID]"
assert needle3 in s;s=s.replace(needle3,needle3+"\n    assert sc['group_index_map']=={str(i+1):[i] for i in range(200)}")
ast.parse(s);(V/new).write_text(s,encoding='utf8',newline='\n')
q=json.loads((V/c0).read_bytes());q['parent_user_authorization']=q['user_authorization'];q['user_authorization']='/root/projects/comperror/workspace/diagnostics/batch_910b/user_decisions/Meta117_current_repeated_user_cleanup_20261121/decision.json'
(V/c1).write_text(json.dumps(q,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
records=[dict(path='scripts/maintenance/'+new,SHA256=sha(s.encode()),base64=base64.b64encode(s.encode()).decode()),dict(path='configs/retention/'+c1,SHA256=sha((V/c1).read_bytes()),base64=base64.b64encode((V/c1).read_bytes()).decode())]
t='DATA='+repr(records)+'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,base64,json,importlib.util
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
for x in DATA:
 p=R/x['path'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'] and not p.exists()
 with p.open('xb') as f:f.write(b)
q=json.loads((R/DATA[1]['path']).read_bytes());a=Path(q['user_authorization']);assert not a.exists();a.parent.mkdir(parents=True,exist_ok=True);a.write_text(json.dumps(dict(origin='actual_user_chat_instruction',thread_id='01a18bbd-8040-7d02-9004-beef491d836a',recorded_UTC=datetime.now(timezone.utc).isoformat(),user_verbatim=['那就再检查服务器上已验收的中间矩阵和临时文件，完整报告、源码、配置、日志和必要数值例证保留，不再需要的清理掉，腾够空间马上串行继续跑，别一直等着。','最好是将，没有用的中间结果定期进行清理，适配的代码，和我的需要的结果肯定是必须的！其他的这个相关项目的资料，都不重要！','再阻塞的时候抓紧时间将那些中间结果进行清理，不要再等了，这个任务已经很久了'],scope='Only exact already fully numerically accepted correct nonexample Meta NPZ; all wrong/invalid/complete examples/small/failed/unverified/reports/source/config/logs preserved; no reference or score change; original retention policy unchanged'),ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
p=R/DATA[0]['path'];sp=importlib.util.spec_from_file_location('Meta117_v3_retention_plan',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.H.mkdir(exist_ok=False);print(json.dumps(m.plan(),ensure_ascii=False))
'''
ast.parse(t);(V/'plan_Meta117_retention_v3_actual_group_layout_20261121_remote.py').write_text(t,encoding='utf8',newline='\n');print(json.dumps({'prepared':True,'source_SHA256':records[0]['SHA256'],'config_SHA256':records[1]['SHA256'],'exact_actual_1based_single_sample_group_binding':True,'no_deletion_yet':True}))

#!/usr/bin/env python3
"""Strict offline 910B evaluation queue. Every actual run uses run.py --config.

The registry is a reviewed model-adaptation manifest, not a replacement model
runner. Rows with weights but no reviewed NPU adaptation remain pending. They
are never misreported as missing weights or completed evaluations.
"""
import argparse,collections,datetime,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback
import yaml
from batch_910b_support import OUTPUT,REQUIRED,atomic_json,digest,validate_config,validate_report,validate_completed_exit,audit_completed_scores,workbook_rows,write_score,workbook_score_representation

ROOT=pathlib.Path(__file__).resolve().parents[1]
DIAG=ROOT/'workspace/diagnostics/batch_910b'
BOOK=pathlib.Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
DIAG.mkdir(parents=True,exist_ok=True)
class MissingDataset(FileNotFoundError):pass

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf8'))
def event(kind,**data):
    with (DIAG/'events.jsonl').open('a',encoding='utf8') as handle:
        handle.write(json.dumps({'time':now(),'event':kind,**data},ensure_ascii=False)+'\n')
def save(state):
    state['updated_at']=now();atomic_json(DIAG/'state.json',state)
    counts=collections.Counter(r['status'] for r in state['models'].values())
    atomic_json(DIAG/'status.json',{'pid':os.getpid(),'phase':state.get('phase'),'updated_at':state['updated_at'],'workbook':str(BOOK),'total':len(state['models']),'counts':dict(counts),'current':state.get('current'),'all_finished':state.get('all_finished',False),'reason':state.get('reason')})

def resource_inventory():
    return {r['model_id']:r for r in read(DIAG/'weights_inventory.json')['results']}

def weight_paths(record,check_files=False):
    valid=[];problems=[]
    for directory in record.get('directories',[]):
        if directory.get('incomplete') or directory.get('invalid'):
            problems.append(directory['path']);continue
        files=[w for w in directory['weights'] if w['valid']]
        if files and (not check_files or all((pathlib.Path(directory['path'])/w['file']).is_file() and (pathlib.Path(directory['path'])/w['file']).stat().st_size==w['bytes'] for w in files)):valid.append(directory['path'])
    return valid,problems

def initialize():
    rows,_,_=workbook_rows(BOOK)
    ids=[r['model_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate model IDs in workbook; review before evaluation')
    weights=resource_inventory()
    state=read(DIAG/'state.json') if (DIAG/'state.json').exists() else {'models':{},'started_at':now()}
    if state['models'] and set(state['models'])!=set(ids):raise RuntimeError('Workbook model set changed; review before resuming evaluation')
    moved=[]
    for row in rows:
        mid=row['model_id'];entry=state['models'].setdefault(mid,dict(row,status='needs_adaptation',attempts=[]))
        if entry['task']!=row['task']:raise RuntimeError('Workbook task changed concurrently: '+mid)
        if entry['row']!=row['row']:
            moved.append({'model_id':mid,'old_row':entry['row'],'current_row':row['row']});entry.setdefault('original_workbook_row',entry['row']);entry['row']=row['row']
        entry['score_in_workbook']=row['score']
        if entry['status']=='done':
            representation=workbook_score_representation(entry['score'],row['score'])
            if representation is None:raise RuntimeError('Completed score changed beyond exact nine-decimal Excel serialization: '+mid)
            entry['workbook_score_representation']=representation;entry['published_score_in_workbook']=float(row['score'])
            continue
        if row['score'] is not None:entry['status']='existing_value_needs_verification';continue
        if mid not in weights:entry.update(status='needs_inventory',reason='Model not covered by weight scan');continue
        paths,problems=weight_paths(weights[mid]);entry['weight_paths']=paths
        if not paths:entry.update(status='skipped_no_complete_weights',reason='No complete existing weights; no download requested or attempted',weight_problems=problems)
    if moved:event('workbook_row_mapping_refreshed_by_unique_model_id',moved_count=len(moved),mappings=moved)
    state['phase']='initialized';state['all_finished']=False;save(state)
    return state

def check_sources(spec):
    required={spec['config'],spec['verifier'],'run.py','universal_eval/orchestrator/pipeline.py'}
    if not required.issubset(spec.get('source_sha256',{})):raise ValueError('Configuration, verifier and shared entrypoint hashes are mandatory')
    if not spec.get('npu_devices') or any(not isinstance(i,int) or not 0<=i<16 for i in spec['npu_devices']):raise ValueError('Explicit valid NPU devices required')
    for rel,value in spec['source_sha256'].items():
        path=(ROOT/rel).resolve()
        if not (path==ROOT/'run.py' or any(path.is_relative_to(ROOT/d) for d in ('universal_eval','scripts','configs'))):raise ValueError('Reviewed executable source must be in source/config directories, never reports or diagnostics')
        if digest(path)!=value:raise RuntimeError('Reviewed source changed concurrently: '+rel)

def config_for(spec,entry,count):
    source=ROOT/spec['config']
    cfg=yaml.safe_load(source.read_text())
    validate_config(cfg,entry['model_id'])
    if cfg['task_type']!=entry['task']:raise ValueError('Workbook task and reviewed YAML task differ')
    if cfg['model']['model_path'] not in entry['weight_paths']:raise ValueError('YAML does not use the model’s exact existing weight directory')
    dataset=pathlib.Path(cfg['data']['dataset_path']).resolve()
    allowed=(ROOT/'workspace/datasets').resolve()
    if not dataset.is_relative_to(allowed):raise ValueError('Dataset must use the existing target dataset directory')
    if not dataset.is_file():raise MissingDataset('Existing target dataset required: '+str(dataset))
    if cfg['model'].get('kwargs',{}).get('npu_devices')!=spec['npu_devices']:raise ValueError('Reviewed device reservation and YAML devices differ')
    if count==200:return source,cfg
    cfg['sample_count']=count;cfg['sample_range']='0:'+str(count-1)
    cfg['name']=cfg.get('name',entry['model_id'])+'-smoke'
    smoke=source.with_name(source.stem+'-smoke.yaml')
    body=json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'
    if smoke.exists() and smoke.read_text()!=body:raise RuntimeError('Refusing to overwrite a different smoke YAML')
    if not smoke.exists():smoke.write_text(body,encoding='utf8')
    return smoke,cfg

def device_idle(spec):
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError('Cannot inspect NPU occupancy')
    import re
    return all(re.search(r'No running processes found in NPU\s+'+str(i)+r'\b',result.stdout) for i in spec['npu_devices'])

def execute_phase(state,entry,spec,count):
    check_sources(spec)
    config,cfg=config_for(spec,entry,count)
    folder=ROOT/'workspace/reports'/entry['model_id'].replace('/','_')/entry['task']
    before=set(folder.iterdir()) if folder.exists() else set()
    attempt=DIAG/'attempts'/entry['model_id'].replace('/','_')/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(count))
    attempt.mkdir(parents=True,exist_ok=False)
    for rel in spec['source_sha256']:
        path=ROOT/rel;snapshot=attempt/'source_snapshot'/rel
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(path.read_bytes())
    (attempt/'effective_config.yaml').write_bytes(config.read_bytes())
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',MODELSCOPE_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',EVAL_OFFLINE_ONLY='1')
    command=['python3','run.py','--config',str(config.relative_to(ROOT))]
    started=time.time();timeout=spec.get('formal_timeout_seconds',28800) if count==200 else spec.get('smoke_timeout_seconds',7200)
    with (attempt/'console.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    entry.update(status='running_formal' if count==200 else 'running_smoke',child_pid=process.pid,active_config=str(config),active_attempt=str(attempt))
    entry['attempts'].append({'count':count,'command':command,'pid':process.pid,'started_at':now(),'diagnostics':str(attempt)})
    state.update(current=entry['model_id'],phase=entry['status']);save(state);event('phase_started',model_id=entry['model_id'],count=count,pid=process.pid,command=command)
    while process.poll() is None:
        if time.time()-started>timeout:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise TimeoutError('Owned model process exceeded reviewed timeout')
        save(state);time.sleep(10)
    entry.pop('child_pid',None)
    entry['attempts'][-1]['returncode']=process.returncode
    if process.returncode not in (0,1):raise RuntimeError('run.py failed; see '+str(attempt/'console.log'))
    new=[p for p in folder.iterdir() if p not in before and p.is_dir()]
    if len(new)!=1:raise RuntimeError('Expected exactly one fresh model/task/timestamp report')
    report=new[0]
    if not (report/'eval_report.json').is_file():raise RuntimeError('run.py did not produce a complete evaluation report; see '+str(attempt/'console.log'))
    grade=validate_completed_exit(process.returncode,report)
    verify_command=['python3',spec['verifier'],str(report),str(count)]
    checked=subprocess.run(verify_command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=spec.get('verify_timeout_seconds',1800))
    (attempt/'verification_stdout.log').write_text(checked.stdout,encoding='utf8')
    (attempt/'verification_stderr.log').write_text(checked.stderr,encoding='utf8')
    if checked.returncode:raise RuntimeError('Independent verification failed; see '+str(attempt))
    proof=json.loads(checked.stdout)
    check_sources(spec)
    score=validate_report(ROOT,report,entry['model_id'],entry['task'],count,proof,fresh_after=started)
    atomic_json(attempt/'accepted_verification.json',proof)
    entry['attempts'][-1].update(report_dir=str(report),independent_verification=str(attempt/'accepted_verification.json'),score=score,completed_at=now())
    event('phase_verified',model_id=entry['model_id'],count=count,score=score,report_dir=str(report));save(state)
    return score,report,proof

def process_one(state,entry,spec):
    check_sources(spec)
    if spec.get('reuse'):
        historical=spec['reuse'];proof=read(historical['proof'])
        score=validate_report(ROOT,historical['report_dir'],entry['model_id'],entry['task'],200,proof)
        if entry['score_in_workbook'] is None or abs(float(entry['score_in_workbook'])-score)>1e-12:raise RuntimeError('Existing first-model score/report mismatch')
        entry.update(status='done',score=score,report_dir=historical['report_dir'],historical_reuse=True,completed_at=now())
        event('verified_existing_result_reused',model_id=entry['model_id'],score=score,report_dir=historical['report_dir'])
        save(state);return
    if entry['score_in_workbook'] is not None:raise RuntimeError('Existing score without approved reuse evidence')
    paths,_=weight_paths(resource_inventory()[entry['model_id']],check_files=True)
    if not paths:
        entry.update(status='skipped_no_complete_weights',reason='Existing weights disappeared or changed before execution');save(state);return
    entry['weight_paths']=paths
    execute_phase(state,entry,spec,3)
    score,report,proof=execute_phase(state,entry,spec,200)
    # Every write is independent, atomic and keyed by model_id. Missing resources
    # never write zero/NaN, and a failed verification cannot reach this point.
    receipt=write_score(BOOK,entry['row'],entry['model_id'],score,DIAG/'workbook_backups',expected_task=entry['task'])
    entry['row']=receipt['row']
    entry.update(status='done',score=score,report_dir=str(report),completed_at=now(),workbook_write=receipt)
    entry['reason']=None;entry['last_error']=None
    atomic_json(DIAG/'receipts'/(entry['model_id'].replace('/','_')+'.json'),entry)
    event('model_completed',model_id=entry['model_id'],score=score,cell='G'+str(entry['row']));save(state)

def run():
    lock=(DIAG/'runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    state=initialize();event('controller_started',pid=os.getpid(),models=len(state['models']))
    while True:
        registry=read(DIAG/'registry.json')
        progress=False;recovery_blocked=False
        for mid,entry in state['models'].items():
            if entry['status'] in ('done','skipped_no_complete_weights','skipped_no_dataset'):continue
            if entry.get('child_pid'):
                p=pathlib.Path('/proc')/str(entry['child_pid'])/'stat'
                if p.exists() and p.read_text().split()[2]!='Z':
                    state.update(phase='recovery_wait_for_owned_process',current=mid);save(state);recovery_blocked=True;break
                entry.update(status='needs_adaptation',reason='Interrupted attempt retained; review before retry');entry.pop('child_pid',None)
            spec=registry.get(mid)
            if not spec or spec.get('approved') is not True:continue
            revision=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
            if entry.get('failed_revision')==revision:continue
            if not spec.get('reuse') and not device_idle(spec):
                state.update(phase='waiting_for_free_devices',current=mid);save(state);continue
            if (DIAG/'PAUSE').exists():state.update(phase='paused_by_control_file');save(state);time.sleep(30);break
            progress=True
            try:
                entry.pop('failed_revision',None)
                process_one(state,entry,spec)
            except MissingDataset as error:
                entry.update(status='skipped_no_dataset',reason=str(error),failed_revision=revision)
                event('resource_missing',model_id=mid,error=str(error));save(state)
            except Exception as error:
                entry.update(status='needs_adaptation',reason=str(error),failed_revision=revision)
                if entry.get('child_pid'):entry.pop('child_pid',None)
                failure=traceback.format_exc();entry['last_error']=failure
                event('model_needs_repair',model_id=mid,error=failure);save(state)
        if recovery_blocked:time.sleep(30);continue
        pending=[e for e in state['models'].values() if e['status'] not in ('done','skipped_no_complete_weights','skipped_no_dataset')]
        if not pending:
            state.update(phase='final_resource_and_report_audit',current=None);save(state)
            scanned=subprocess.run(['python3','scripts/inventory_910b_weights.py'],cwd=ROOT,capture_output=True,text=True,timeout=600)
            if scanned.returncode:raise RuntimeError('Final existing-weight inventory failed')
            fresh=json.loads(scanned.stdout)
            if fresh.get('errors'):raise RuntimeError('Final weight scan has unreadable paths')
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            atomic_json(DIAG/('weights_final_'+stamp+'.json'),fresh)
            refreshed={r['model_id']:r for r in fresh['results']}
            resources_appeared=False
            for mid,entry in state['models'].items():
                if entry['status']=='skipped_no_complete_weights':
                    paths,_=weight_paths(refreshed[mid])
                    if paths:entry.update(status='needs_adaptation',weight_paths=paths,reason='Complete weights now exist');resources_appeared=True
                if entry['status']=='skipped_no_dataset' and mid in registry:
                    cfg=yaml.safe_load((ROOT/registry[mid]['config']).read_text())
                    if pathlib.Path(cfg['data']['dataset_path']).is_file():entry.update(status='needs_adaptation',reason='Dataset now exists');entry.pop('failed_revision',None);resources_appeared=True
            if resources_appeared:
                atomic_json(DIAG/('weights_previous_'+stamp+'.json'),read(DIAG/'weights_inventory.json'))
                atomic_json(DIAG/'weights_inventory.json',fresh);save(state);continue
            audit=audit_completed_scores(ROOT,BOOK,state['models'],registry)
            atomic_json(DIAG/'final_audit.json',audit)
            state.update(phase='all_finished',all_finished=True,current=None,completed_at=now());save(state);event('all_models_finished',counts=dict(collections.Counter(e['status'] for e in state['models'].values())))
            return
        state.update(phase='waiting_for_adaptation',current=None,all_finished=False);save(state)
        if not progress:time.sleep(30)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--plan',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        print(json.dumps(read(DIAG/'status.json'),ensure_ascii=False,indent=2));return
    if args.plan:
        state=initialize();print(json.dumps({'total':len(state['models']),'counts':dict(collections.Counter(e['status'] for e in state['models'].values()))},ensure_ascii=False));return
    if args.run:run();return
    parser.print_help()

if __name__=='__main__':main()

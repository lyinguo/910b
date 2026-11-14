"""Workbook-only cell updates and strict report validation for the 910B queue."""
import copy,hashlib,io,json,math,os,pathlib,re,tempfile,zipfile
import xml.etree.ElementTree as ET

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
N={'s':NS};Q=lambda n:'{'+NS+'}'+n
REQUIRED={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
OUTPUT={'reports_dir':'workspace/reports','layout':'model_task_timestamp','evidence_dir':'workspace/diagnostics/model_runs','timezone':'Asia/Hong_Kong','verbose':True}

def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def workbook_rows(path):
    raw=pathlib.Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        first=workbook.find(Q('sheets'))[0]
        relid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        relationships=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target=next(x.get('Target') for x in relationships if x.get('Id')==relid)
        sheet=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        xml=ET.fromstring(z.read(sheet))
        def value(c):
            if c.get('t')=='s':return strings[int(c.find(Q('v')).text)]
            if c.get('t')=='inlineStr':return ''.join(c.find(Q('is')).itertext())
            v=c.find(Q('v'));return None if v is None else v.text
        rows=[]
        for row in xml.find(Q('sheetData')):
            number=int(row.get('r'));cells={c.get('r'):value(c) for c in row}
            if number==1:
                if cells.get('C1')!='model_id' or str(cells.get('G1')).lower()!='910b':
                    raise ValueError('Expected model_id in column C and 910b in column G')
            elif isinstance(cells.get('C'+str(number)),str) and '/' in cells['C'+str(number)]:
                rows.append({'row':number,'model_id':cells['C'+str(number)].strip(),'task':cells.get('B'+str(number)),'score':cells.get('G'+str(number))})
    return rows,sheet,hashlib.sha256(raw).hexdigest()

def workbook_score_representation(actual,published):
    """Keep full verified metrics; accept only exact Excel values or their observed nine-decimal serialization."""
    if published is None:return None
    try:expected=float(actual);value=float(published)
    except (TypeError,ValueError):return None
    if not math.isfinite(expected) or not math.isfinite(value):return None
    if abs(value-expected)<=1e-12:return 'exact'
    if abs(value-round(expected,9))<=1e-12:return 'serialized_9_decimal_places'
    return None


def write_score(book,row,model_id,score,diagnostics,expected_task=None):
    """Patch one numeric G cell located by unique current C/model_id; row is a historical hint. Preserve all other ZIP members byte-for-byte, current row ordering, task checks, existing-value protection and concurrent-hash checks.

    The remote Python environment has neither Artifact Tool nor spreadsheet
    libraries. Native XML patching avoids flattening formulas and annotations.
    """
    if not isinstance(score,(int,float)) or not math.isfinite(score):raise ValueError('Nonfinite score')
    book=pathlib.Path(book);diagnostics=pathlib.Path(diagnostics)
    rows,sheet,expected_hash=workbook_rows(book)
    requested_row=row
    matches=[r for r in rows if r['model_id']==model_id]
    if len(matches)!=1:raise RuntimeError('Expected exactly one current workbook row for model_id: '+model_id)
    record=matches[0];row=record['row']
    if expected_task is not None and record['task']!=expected_task:raise RuntimeError('Workbook task changed concurrently: '+model_id)
    if record['score'] is not None:
        representation=workbook_score_representation(score,record['score'])
        if representation is not None:return {'reused_existing_value':True,'row':row,'cell':'G'+str(row),'requested_row':requested_row,'row_relocated':row!=requested_row,'model_id':model_id,'published_score':float(record['score']),'score_representation':representation}
        raise RuntimeError('Existing score differs; refusing to overwrite it')
    original=book.read_bytes()
    if hashlib.sha256(original).hexdigest()!=expected_hash:raise RuntimeError('Concurrent workbook change')
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        infos=archive.infolist();parts={i.filename:archive.read(i.filename) for i in infos}
    xml=parts[sheet].decode('utf8')
    row_pattern=r'(<(?P<prefix>[\w.-]+:)?row\b[^>]*\br="'+str(row)+r'"[^>]*>)(.*?)(</(?:[\w.-]+:)?row>)'
    found=re.search(row_pattern,xml,re.S)
    if found is None:raise ValueError('Workbook row not found')
    prefix=found.group('prefix') or '';body=found.group(3)
    cell_pattern=r'<(?:[\w.-]+:)?c\b[^>]*\br="([A-Z]+)'+str(row)+r'"[^>]*(?:/>|>.*?</(?:[\w.-]+:)?c>)'
    cells=list(re.finditer(cell_pattern,body,re.S))
    old=next((m for m in cells if m.group(1)=='G'),None)
    template=old or next((m for m in cells if m.group(1)=='F'),None)
    attributes=''
    if template:
        style=re.search(r'\bs="(\d+)"',template.group())
        if style:attributes=' s="'+style.group(1)+'"'
    new=f'<{prefix}c r="G{row}"{attributes} t="n"><{prefix}v>{score!r}</{prefix}v></{prefix}c>'
    if old:
        if re.search(r'<(?:[\w.-]+:)?f\b',old.group()):raise RuntimeError('Refusing to replace a formula')
        body=body[:old.start()]+new+body[old.end():]
    else:
        position=next((m.start() for m in cells if (len(m.group(1)),m.group(1))>(1,'G')),len(body))
        body=body[:position]+new+body[position:]
    parts[sheet]=(xml[:found.start(3)]+body+xml[found.end(3):]).encode('utf8')
    diagnostics.mkdir(parents=True,exist_ok=True)
    backup=diagnostics/('workbook_before_'+expected_hash+'.xlsx')
    if not backup.exists():backup.write_bytes(original)
    fd,name=tempfile.mkstemp(prefix=book.name+'.',suffix='.tmp',dir=book.parent);os.close(fd)
    candidate=pathlib.Path(name)
    try:
        with zipfile.ZipFile(candidate,'w') as archive:
            for info in infos:archive.writestr(info,parts[info.filename])
        new_rows,new_sheet,_=workbook_rows(candidate)
        expected_rows=copy.deepcopy(rows)
        next(r for r in expected_rows if r['row']==row)['score']=repr(score)
        if new_rows!=expected_rows:raise RuntimeError('Unexpected workbook row/value change')
        with zipfile.ZipFile(candidate) as archive:
            for info in infos:
                if info.filename!=sheet and archive.read(info.filename)!=parts[info.filename]:raise RuntimeError('Unrelated workbook part changed')
        if digest(book)!=expected_hash:raise RuntimeError('Workbook changed immediately before commit')
        os.replace(candidate,book)
    finally:
        if candidate.exists():candidate.unlink()
    return {'cell':'G'+str(row),'row':row,'requested_row':requested_row,'row_relocated':row!=requested_row,'model_id':model_id,'score':score,'before_sha256':expected_hash,'after_sha256':digest(book)}

def validate_config(cfg,model_id):
    if cfg.get('model',{}).get('name')!=model_id:raise ValueError('Wrong model name in YAML')
    if cfg['model'].get('download') is not False or cfg.get('data',{}).get('download') is not False:raise ValueError('Downloads must be explicitly disabled')
    if cfg.get('output')!=OUTPUT:raise ValueError('Mandatory standard output configuration missing/changed')
    if cfg.get('sample_range') not in ('0:2','0:199'):raise ValueError('Fixed sample range required')
    if cfg.get('data',{}).get('filter_empty') is not False:raise ValueError('Do not silently filter the fixed sample set')

def validate_completed_exit(returncode,folder):
    folder=pathlib.Path(folder)
    if returncode not in (0,1):raise ValueError('Runtime process failed with code '+str(returncode))
    if not folder.is_dir() or {p.name for p in folder.iterdir()}!=REQUIRED:raise ValueError('Completed report required even when return code is 1')
    report=json.loads((folder/'eval_report.json').read_text())
    if returncode==1 and report.get('summary',{}).get('passed') is not False:raise ValueError('Return code 1 is not explained by the baseline grade')
    return report.get('summary',{}).get('passed')

def validate_report(root,folder,model_id,task,count,proof,fresh_after=None):
    root=pathlib.Path(root).resolve();folder=pathlib.Path(folder).resolve()
    expected_parent=root/'workspace/reports'/model_id.replace('/','_')/task
    if folder.parent!=expected_parent or not re.fullmatch(r'\d{8}_\d{6}',folder.name):raise ValueError('Nonstandard report directory')
    if {p.name for p in folder.iterdir()}!=REQUIRED:raise ValueError('Report must contain exactly five standard files')
    if fresh_after is not None and (folder/'eval_report.json').stat().st_mtime<fresh_after-2:raise ValueError('Historical report cannot be claimed as a new run')
    report=json.loads((folder/'eval_report.json').read_text())
    validate_config(report['pipeline_config'],model_id)
    if report['pipeline_config']['task_type']!=task:raise ValueError('Report task mismatch')
    if report['pipeline_config']['sample_range']!='0:'+str(count-1):raise ValueError('Wrong report sample range')
    inputs=[json.loads(x) for x in (folder/'sample_inputs.jsonl').read_text().splitlines()]
    outputs=[json.loads(x) for x in (folder/'sample_outputs.jsonl').read_text().splitlines()]
    details=report.get('detailed_results',[])
    if not (len(inputs)==len(outputs)==len(details)==count):raise ValueError('Missing/truncated sample records')
    for records in (inputs,outputs,details):
        if [r.get('index') for r in records]!=list(range(count)):raise ValueError('Duplicate/missing sample indices')
    if not isinstance(report.get('operator_lineage'),dict) or not report['operator_lineage']:raise ValueError('Execution lineage missing')
    if report.get('statistics',{}).get('total_samples')!=count:raise ValueError('Statistics sample count mismatch')
    for a,b,c in zip(inputs,outputs,details):
        if a.get('reference')!=b.get('reference') or b.get('reference')!=c.get('reference'):raise ValueError('References differ between report files')
        if b.get('prediction')!=c.get('prediction'):raise ValueError('Predictions differ between report files')
        if 'raw_output' not in b or 'error' not in b:raise ValueError('Full raw output/error fields required')
    if not proof.get('passed') or proof.get('sample_count')!=count:raise ValueError('Independent verifier did not pass')
    if pathlib.Path(proof['report_dir']).resolve()!=folder:raise ValueError('Verifier used a different report')
    metric=report['summary']['primary_metric'];score=report['metrics'].get(metric)
    independent=proof.get(metric,proof.get('metrics',{}).get(metric))
    if not isinstance(score,(int,float)) or independent is None or abs(score-independent)>1e-10:raise ValueError('Independent score mismatch')
    if proof.get('NPU_profiler_kernel_events',0)<=0:raise ValueError('Actual NPU kernel execution not verified')
    for name,value in proof.get('report_sha256',{}).items():
        if name not in REQUIRED or digest(folder/name)!=value:raise ValueError('Report changed after independent verification')
    if set(proof.get('report_sha256',{}))!=REQUIRED:raise ValueError('Independent verifier must hash all report files')
    return float(score)

def audit_completed_scores(root,book,models,registry):
    rows,_,_=workbook_rows(book)
    by_id={r['model_id']:r for r in rows}
    if len(rows)!=len(by_id) or set(by_id)!=set(models):raise ValueError('Workbook model scope or uniqueness changed; review before completion')
    checked=0
    for mid,entry in models.items():
        row=by_id[mid]
        if row['task']!=entry['task']:raise ValueError('Workbook task changed')
        if entry['status']!='done':
            if row['score'] is not None:raise ValueError('Skipped/pending row unexpectedly contains a score')
            continue
        if workbook_score_representation(entry['score'],row['score']) is None:raise ValueError('Completed published score missing or changed beyond exact nine-decimal serialization')
        proof_path=registry[mid]['reuse']['proof'] if entry.get('historical_reuse') else entry['attempts'][-1]['independent_verification']
        proof=json.loads(pathlib.Path(proof_path).read_text())
        score=validate_report(root,entry['report_dir'],mid,entry['task'],200,proof)
        if abs(score-entry['score'])>1e-12:raise ValueError('Final report/full verified state score mismatch')
        checked+=1
    return {'passed':True,'verified_completed_models':checked,'workbook_sha256':digest(book),'preserved_9dp_serializations':[{'model_id':mid,'current_cell':'G'+str(by_id[mid]['row']),'full_verified_metric':entry['score'],'published_score':float(by_id[mid]['score'])} for mid,entry in models.items() if entry['status']=='done' and workbook_score_representation(entry['score'],by_id[mid]['score'])=='serialized_9_decimal_places']}

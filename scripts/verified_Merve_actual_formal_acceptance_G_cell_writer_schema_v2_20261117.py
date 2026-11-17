"""Conditional direct A100 publication: actual user authorization, real fresh3+200 evidence, current D-relative5%, exact one-G XML patch. Shared core unchanged."""
import pathlib,json,copy,hashlib,io,math,os,re,tempfile,zipfile,decimal
from batch_910b_support import workbook_rows,digest
BOOK=pathlib.Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')

def check_formal_acceptance_metadata(completed,p,job,source):
 assert completed['phase']=='completed_formal_independent_acceptance_of_original_NPU200' and completed['actual_returncode']==0 and completed['verifier_returncode']==0 and completed['actual_new_model_forwards_this_phase']==0
 assert p['passed'] and p['sample_count']==200 and p['actual_NPU_inference'] and not p['historical_record_reuse'] and p['formal_alignment_within_5pct']
 assert p['formal_independent_acceptance_phase_not_diagnostic_only'] and p['no_new_NPU_or_learned_CPU_inference_in_this_acceptance_phase'] and p['all200_input_pixels_exact_original_native_media_decode_verified'] and p['original_full_numeric_model_checks_unchanged']
 assert job['model_id']=='merve/paligemma_vqav2' and job['execution_kind']=='formal_independent_acceptance_of_completed_original_NPU200_only' and job['stages']==[3,200] and job['spec']['approved'] and not job['write_workbook']
 assert source['model_returncode']==0 and source['verifier_returncode']==1 and source['actual_returncode']==1
 assert [x['sample_count'] for x in completed['accepted']]==[3,200]
 return True

def authorize_actual_merve_formal_acceptance(completed,m,model_id):
 assert model_id=='merve/paligemma_vqav2'
 p=json.loads(pathlib.Path(m['proof']).read_text());ctx=pathlib.Path(p['current_execution_job']).resolve();assert ctx.is_relative_to('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103');assert digest(ctx)==p['current_execution_job_SHA256'];job=json.loads(ctx.read_text());assert pathlib.Path(m['job_completion']).resolve()==ctx.parent/'completion.json'
 source_home=pathlib.Path(job['source_original_NPU200_job']);assert str(source_home)==p['actual_original_NPU200_model_job'];source=json.loads((source_home/'completion.json').read_text());assert digest(source_home/'job.json')==job['source_original_NPU200_job_SHA256'] and digest(source_home/'completion.json')==job['source_original_NPU200_completion_SHA256'];check_formal_acceptance_metadata(completed,p,job,source)
 assert completed['model_returncode_source_original_job']==str(source_home/'completion.json') and completed['model_returncode']==source['model_returncode']==p['actual_original_NPU200_model_returncode']==0
 assert "assert matches,'No own original processor branch reproduced actual native input pixels'" in (source_home/'formal200/verification_stderr.log').read_text()
 original_job=json.loads((source_home/'job.json').read_text());assert original_job['spec']['config']==job['spec']['config'] and original_job['spec']['alignment_approval_review_sha256']==job['spec']['alignment_approval_review_sha256'];assert not job['write_workbook']
 for name,h in original_job['spec']['source_sha256'].items():assert digest(pathlib.Path('/root/projects/comperror')/name)==h and job['spec']['source_sha256'][name]==h
 from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
 prior=validate_resume(job,pathlib.Path('/root/projects/comperror'));assert completed['accepted'][0]['proof_sha256']==prior['proof_sha256']
 for name,h in job['spec']['source_sha256'].items():assert digest(pathlib.Path('/root/projects/comperror')/name)==h
 assert digest(pathlib.Path('/root/projects/comperror')/job['spec']['verifier'])==p['independent_verifier_SHA256']
 return True

def authorize(book,model_id,score,decision_path,expected_book_hash,manifest_path):
 d=json.loads(pathlib.Path(decision_path).read_text());m=json.loads(pathlib.Path(manifest_path).read_text());book=pathlib.Path(book).resolve()
 if d.get('test_fixture_only'):
  assert d['origin']=='isolated_test_fixture' and book.is_relative_to(pathlib.Path(d['test_root']).resolve()) and book!=BOOK
 else:
  assert book==BOOK and d['origin']=='actual_user_chat_instruction' and d['overwrite_existing_G_when_verified'] and d['reask_per_model_not_required'] and d['required_sample_count']==200 and d['required_real_smoke_count']==3
 assert expected_book_hash==digest(book),'Concurrent workbook change'
 assert m['model_id']==model_id and float(m['new_score'])==float(score) and m['new_real_NPU_inference'] and not m['historical_record_reuse']
 completed=json.loads(pathlib.Path(m['job_completion']).read_text());assert completed['actual_returncode']==0 and [x['sample_count'] for x in completed['accepted']]==[3,200]
 if completed['phase']!='completed_evidence_only':assert authorize_actual_merve_formal_acceptance(completed,m,model_id)
 for item in completed['accepted']:
  assert digest(item['proof'])==item['proof_sha256'];p=json.loads(pathlib.Path(item['proof']).read_text());assert p['passed'] and p['sample_count']==item['sample_count'] and not p['historical_record_reuse']
 run=completed['accepted'][-1];assert m['proof']==run['proof'] and m['proof_sha256']==run['proof_sha256'] and m['report_dir']==run['report_dir']
 p=json.loads(pathlib.Path(run['proof']).read_text());folder=pathlib.Path(run['report_dir']).resolve();assert folder.is_relative_to('/root/projects/comperror/workspace/reports')
 files={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};assert {q.name for q in folder.iterdir()}==files and set(p['report_sha256'])==files
 for name,h in p['report_sha256'].items():assert digest(folder/name)==h
 r=json.loads((folder/'eval_report.json').read_text());assert r['pipeline_config']['model']['name']==model_id and r['pipeline_config']['task_type']==m['task'] and r['pipeline_config']['sample_count']==200 and r['pipeline_config']['sample_range']=='0:199'
 metric=r['summary']['primary_metric'];assert float(r['metrics'][metric])==float(p['metrics'][metric])==float(score) and p['formal_alignment_within_5pct']
 if not d.get('test_fixture_only'):
  from local_A100_reference_gate_20261103 import check_local_A100_reference
  m['local_A100_full_precision_gate']=check_local_A100_reference(model_id,m['task'],score)
 return m

def write_verified_A100_G(book,row,model_id,score,diagnostics,expected_task=None,*,decision_path,expected_book_hash,manifest_path):
    """Patch one numeric G cell located by unique current C/model_id; row is a historical hint. Preserve all other ZIP members byte-for-byte, current row ordering, task checks, existing-value protection and concurrent-hash checks.

    The remote Python environment has neither Artifact Tool nor spreadsheet
    libraries. Native XML patching avoids flattening formulas and annotations.
    """
    manifest=authorize(book,model_id,score,decision_path,expected_book_hash,manifest_path)
    assert expected_task==manifest['task']
    if not isinstance(score,(int,float)) or not math.isfinite(score):raise ValueError('Nonfinite score')
    book=pathlib.Path(book);diagnostics=pathlib.Path(diagnostics)
    rows,sheet,expected_hash=workbook_rows(book)
    requested_row=row
    matches=[r for r in rows if r['model_id']==model_id]
    if len(matches)!=1:raise RuntimeError('Expected exactly one current workbook row for model_id: '+model_id)
    record=matches[0];row=record['row']
    if expected_task is not None and record['task']!=expected_task:raise RuntimeError('Workbook task changed concurrently: '+model_id)
    expected_old=manifest['expected_old_score']
    if (record['score'] is None)!=(expected_old is None) or (expected_old is not None and decimal.Decimal(record['score'])!=decimal.Decimal(str(expected_old))):raise RuntimeError('Old G differs from prepared manifest')
    import xml.etree.ElementTree as E
    with zipfile.ZipFile(book) as z:
        root=E.fromstring(z.read(sheet));ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};dcell=root.find('.//s:c[@r="D'+str(row)+'"]',ns);baseline=dcell.findtext('s:v',namespaces=ns)
    base=decimal.Decimal(baseline);new_score=decimal.Decimal(str(score))
    if not base.is_finite() or base==0 or abs(new_score-base)/abs(base)>decimal.Decimal('.05'):raise RuntimeError('New score fails current A100 D-relative5%')
    old_value=record['score']
    original=book.read_bytes()
    if hashlib.sha256(original).hexdigest()!=expected_hash:raise RuntimeError('Concurrent workbook change')
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        infos=archive.infolist();parts={i.filename:archive.read(i.filename) for i in infos};archive_comment=archive.comment
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
            archive.comment=archive_comment
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
    return {'cell':'G'+str(row),'row':row,'requested_row':requested_row,'row_relocated':row!=requested_row,'model_id':model_id,'score':score,'old_score':old_value,'conditional_verified_A100_G_publication':True,'old_nonempty_G_replaced':old_value is not None,'authorization_sha256':digest(decision_path),'before_sha256':expected_hash,'after_sha256':digest(book)}

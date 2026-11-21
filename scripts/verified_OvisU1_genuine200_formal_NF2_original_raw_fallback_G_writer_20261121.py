"""Authorize only the original failed Ovis NN200 plus its actual complete NF2 acceptance.

The original job remains failed. The original raw one-G ZIP write function is
byte-for-byte identical to the frozen writer; only this explicit authorization
knows the separately completed formal acceptance, with zero new NN forwards.
"""
import pathlib,json,copy,hashlib,io,math,os,re,tempfile,zipfile,decimal
from batch_910b_support import workbook_rows,digest
BOOK=pathlib.Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
R=pathlib.Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='AIDC-AI/Ovis-U1-3B'
J=B/'OvisU1_completed_storage_fullNF_passed_exact23_capacity_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261121'
NF=B/'OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121'
def read(p):return json.loads(pathlib.Path(p).read_bytes())

def full_numeric_head_verified(proof):
 checks=proof.get('numeric_sample_checks',[]);count=proof.get('sample_count')
 limits={'LM_all512_head_CPU_BF16_reference_relative_RMSE':.02,'visual_full_head_relative_RMSE':.02,'visual_softmax_relative_RMSE':.002,'visual_embedding_relative_RMSE':.02}
 return count in [3,200] and len(checks)==count and {x['index'] for x in checks}==set(range(count)) and proof.get('actual_forward_count')==count*512 and all(x.get('all28_KV_and_all_masks_positions_preserved') is True and x.get('CPU_references_diagnostic_only_not_grade') is True and all(math.isfinite(x[k]) and 0<=x[k]<lim for k,lim in limits.items()) for x in checks)

def check_formal_metadata(old,acceptance,completion,proof,fresh):
 assert old['phase']=='failed_evidence_preserved' and old['actual_returncode']==old['verifier_returncode']==1 and old['model_returncode']==0 and old['sample_count']==200
 assert [x['sample_count'] for x in old['accepted']]==[3] and old['accepted'][0]['proof_sha256']=='a3422cc1cf1b45002e910374e655d832941702741a9cfe7ec9064c6d987f7443'
 assert acceptance['passed'] is True and acceptance['actual_NEW_model_forwards']==0 and acceptance['original_job_remains_failed'] is True
 assert acceptance['original_model_returncode']==0 and acceptance['original_verifier_returncode']==1 and acceptance['current_verifier_returncode']==0
 assert acceptance['original_sample_count']==200 and acceptance['original_native_forwards']==102400 and acceptance['score']==.895
 assert acceptance['own_approved_NEWfresh3_proof_SHA256']=='a3422cc1cf1b45002e910374e655d832941702741a9cfe7ec9064c6d987f7443'
 assert acceptance['original_full_five_reports_unchanged'] and acceptance['all_own1060_source_and_all200_full_numeric_head_KV_inputs_raw_ref_error_denominators_verified'] and acceptance['only_original_raw_prediction_fallback_expected_metadata_corrected']
 assert acceptance['original_protocol_qualified'] is True and acceptance['no_G_or_canonical_write'] is True
 assert completion['actual_returncode']==0 and completion['actual_NEW_model_forwards']==0 and completion['no_G_or_canonical_write']
 for p,count in [(fresh,3),(proof,200)]:
  assert p['passed'] and p['sample_count']==count and p['actual_NPU_inference'] and p['NEW_real_NPU_inference'] and not p['historical_record_reuse']
  assert p['source_tensor_count']==1060 and p['active_physical_source_tensor_count']==487 and p['original_unexecuted_generator_source_tensor_count']==573
  assert p['all_own_source_actual_NPU_SHA256_verified'] and p['all_native_KV_values_losslessly_reconstructed'] and p['all_full_inputs_tokens_raw_references_errors_preserved'] and full_numeric_head_verified(p)
  assert p['error_indices']==p['empty_indices']==[] and p['truncated_indices']==list(range(count)) and not p['A100_historical_engine_or_tensor_capture_claimed']
 assert proof['metrics']['Accuracy']==.895 and proof['invalid_indices']==[198,199] and proof['formal_alignment_within_5pct'] and proof['original_protocol_final_qualification_complete']
 assert set(proof['relative_error_to_current_A100_D'])=={'server','local','preserved_original'} and all(decimal.Decimal(x)<=decimal.Decimal('.05') for x in proof['relative_error_to_current_A100_D'].values())
 assert proof['independent_verifier_SHA256']=='60c0d21fd51f9ad8e712890330e9b7c887d72e8154ace6a14b11ab5f6fc20052'
 return True

def verified_evidence(manifest):
 assert pathlib.Path(manifest['job_completion']).resolve()==J/'completion.json' and pathlib.Path(manifest['formal_acceptance']).resolve()==NF/'formal_acceptance.json'
 assert digest(J/'job.json')=='11f0c684e4dfe6f39bff484c2386dca3e7371f778230faa6ca2a65ca64affc7d'
 job=read(J/'job.json');old=read(J/'completion.json');a=read(NF/'formal_acceptance.json');c=read(NF/'actual_completion.json')
 assert digest(NF/'formal_acceptance.json')==manifest['formal_acceptance_SHA256']==c['formal_acceptance_SHA256']
 assert c['formal_acceptance']==str(NF/'formal_acceptance.json') and a['original_genuine_NEW_NPU200_job']==str(J) and a['original_job_SHA256']==digest(J/'job.json')
 assert manifest['proof']==a['proof']==str(NF/'verification_stdout.json') and manifest['proof_sha256']==a['proof_SHA256']==digest(manifest['proof'])
 p=read(manifest['proof']);f=old['accepted'][0];assert digest(f['proof'])==f['proof_sha256']=='a3422cc1cf1b45002e910374e655d832941702741a9cfe7ec9064c6d987f7443';fresh=read(f['proof'])
 assert manifest['report_dir']==old['report_dir']==p['report_dir'] and pathlib.Path(p['report_dir']).name=='20261121_093507'
 check_formal_metadata(old,a,c,p,fresh)
 assert job['stages']==[3,200] and job['model_id']==MID and job['spec']['approved'] and not job['write_workbook']
 for n,h in job['spec']['source_sha256'].items():
  assert not pathlib.Path(n).is_absolute() and '..' not in pathlib.Path(n).parts and digest(R/n)==h
 assert digest(R/'scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py')==p['independent_verifier_SHA256']
 for proof in [fresh,p]:
  folder=pathlib.Path(proof['report_dir']);files={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
  assert folder.resolve().is_relative_to(R/'workspace/reports') and {x.name for x in folder.iterdir()}==set(proof['report_sha256'])==files
  for n,h in proof['report_sha256'].items():assert digest(folder/n)==h
 r=read(pathlib.Path(p['report_dir'])/'eval_report.json');cfg=r['pipeline_config'];assert cfg['model']['name']==MID and cfg['task_type']==manifest['task']=='image-text-to-text' and cfg['sample_count']==200 and cfg['sample_range']=='0:199' and cfg['algorithm']['batch_size']==4
 assert r['metrics']['Accuracy']==p['metrics']['Accuracy']==manifest['new_score']==.895 and r['statistics']['total_samples']==200 and r['statistics']['error_count']==0
 return job,old,a,c,p,fresh

def authorize(book,model_id,score,decision_path,expected_book_hash,manifest_path):
 d=read(decision_path);m=read(manifest_path);book=pathlib.Path(book).resolve()
 if d.get('test_fixture_only'):
  assert d['origin']=='isolated_test_fixture' and book.is_relative_to(pathlib.Path(d['test_root']).resolve()) and book!=BOOK
 else:
  assert book==BOOK and d['origin']=='actual_user_chat_instruction' and d['overwrite_existing_G_when_verified'] and d['reask_per_model_not_required'] and d['required_sample_count']==200 and d['required_real_smoke_count']==3
 assert expected_book_hash==digest(book),'Concurrent workbook change'
 assert model_id==m['model_id']==MID and float(m['new_score'])==float(score)==.895 and m['new_real_NPU_inference'] and not m['historical_record_reuse']
 verified_evidence(m)
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

from pathlib import Path
from decimal import Decimal
import hashlib,json,shutil,zipfile,xml.etree.ElementTree as E
V=Path(__file__).resolve().parent
P=Path('D:/code-lv/.tmp/batch_910b_20261030')
packet=json.loads(Path(__import__('sys').argv[1]).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
def item(suffix):
 a=[v for n,v in packet['files'].items() if n.endswith(suffix) and 'complete_text' in v]
 assert len(a)==1,(suffix,len(a))
 return json.loads(a[0]['complete_text']),a[0]['SHA256']
MID='zai-org/GLM-4.1V-9B-Thinking'
JNAME='GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
CNAME='GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120'
job,jh=item(JNAME+'/job.json');done,_=item(JNAME+'/completion.json');supervisor,_=item(CNAME+'/actual_completion.json')
pf,pfsha=item(JNAME+'/formal200/verification_stdout.json')
assert jh=='756c7f9221f17a39e85f00cf064d640d197fa9ec42e1af0482eac85f17c7d163'
assert done['actual_returncode']==done['model_returncode']==done['verifier_returncode']==supervisor['actual_returncode']==0
assert [x['sample_count'] for x in done['accepted']]==[3,200]
assert done['accepted'][-1]['proof_sha256']==pfsha
assert pf['passed'] and pf['sample_count']==200 and pf['actual_NPU_inference'] and not pf['historical_record_reuse']
assert pf['original_protocol_final_qualification_complete'] and pf['formal_alignment_within_5pct']
assert set(pf['relative_error_to_current_A100_D'])=={'server','local','preserved_original'} and all(Decimal(v)<=Decimal('.05') for v in pf['relative_error_to_current_A100_D'].values())
assert pf['independent_verifier_SHA256']=='48db53f3f64f7149eb11cdcb1278b8136c17c9a7d432fc82b7c914fb027b1203'
true,trueh=item('GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120/fresh3/verification_stdout.json')
assert trueh=='48d8a3ad392c380c134d1185a800e25039689d4cf92dd820bb460db8a1ca9478' and true['passed']
BEFORE='65eb41b210b31b61630c6706e02b36e7a91afe8fe4d9415daabd8b3600513d81'
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
assert sha(book.read_bytes())==BEFORE
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 first=E.fromstring(z.read('xl/workbook.xml')).find('s:sheets/s:sheet',ns)
 rid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
 target=next(x.get('Target') for x in E.fromstring(z.read('xl/_rels/workbook.xml.rels')) if x.get('Id')==rid)
 sheet=target.lstrip('/') if target.startswith('/') else 'xl/'+target
 ss=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
 values={}
 for c in E.fromstring(z.read(sheet)).findall('.//s:c',ns):
  v=c.findtext('s:v',namespaces=ns)
  values[c.get('r')]=ss[int(v)] if c.get('t')=='s' else ''.join(c.find('s:is',ns).itertext()) if c.get('t')=='inlineStr' else v
ids={v:int(k[1:]) for k,v in values.items() if k.startswith('C') and k!='C1' and v}
assert len(ids)==700 and ids[MID]==696 and values['B696']=='image-text-to-text' and Decimal(values['D696'])==Decimal('.75') and values.get('G696') in (None,'')
baseline=json.loads((P/'AutoGLMPhone9BSingle_verified183_incremental_real_results_export_20261120.json').read_bytes())
assert baseline['measured_count']==len(baseline['records'])==183
for rec in baseline['records']:
 r=ids[rec['model_id']]
 assert values[f'B{r}']==rec['task'] and Decimal(values[f'D{r}'])==Decimal(rec['current_D']) and Decimal(values[f'G{r}'])==Decimal(rec['current_G'])
H=V/'GLM41Thinking_local_result_sync_verified184_20261120';H.mkdir(exist_ok=False)
backup=H/'original_user_workbook.xlsx';shutil.copy2(book,backup);st=book.stat()
root='/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
score=pf['metrics']['Accuracy']
q=dict(target=str(book),source_backup=str(backup),expected_before_sha256=BEFORE,sheet_name=first.get('name'),sheet_xml=sheet,last_model_row=max(ids.values()),changes=[dict(model_id=MID,task='image-text-to-text',row=696,cell='G696',old_value=None,new_value=score,A100_full_precision_cached_D=values['D696'],relative_error=pf['relative_error_to_current_A100_D']['server'],new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir=pf['report_dir'],formal_proof_sha256=pfsha)],real_measured_count=184,numeric_within5pct=169,numeric_outside5pct=15,complete_weights_pending_formal=206,confirmed_missing=310,original_protocol_qualified=70,model_blank_count=516,complete_numeric_verifier_passed=True,all_prior183_C_B_D_G_checked=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),models={MID:dict(job=root+'/'+JNAME,controller=root+'/'+CNAME,job_SHA256=jh,true3_score=true['metrics']['Accuracy'],true3_proof_SHA256=trueh,expected_forward_count=pf['actual_forward_count'],expected_kernels=pf['NPU_profiler_kernel_events'],expected_empty=pf['empty_indices'],expected_invalid=pf['invalid_indices'],expected_truncated=pf['truncated_indices'])},test_template='scripts/tests/verified_Gemma4E2B_manifest_task_writer_11_cases_template_20261116.py',test_template_SHA256='8e595053cedf0ff3fc46b1f1a76fde787ed268ac190bc037e7521882e048e318')
(H/'plan.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(ready=True,local_home=str(H),formal_proof_SHA256=pfsha,Accuracy=score,three_D_errors=pf['relative_error_to_current_A100_D'],actual_forward_count=pf['actual_forward_count'],no_production_write=True)))

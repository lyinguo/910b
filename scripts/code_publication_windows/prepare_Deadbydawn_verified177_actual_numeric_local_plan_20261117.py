import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import json,hashlib,zipfile,shutil,xml.etree.ElementTree as ET
P=Path(__file__).parent
x=json.loads((P/'R_Deadbydawn_v3_full_evidence_for_publication_final_20261117.json').read_bytes())
assert x.get('completion') and x.get('controller_completion'),'Actual full NF and runner/controller completion still required'
done=x['completion'];proof=x['formal_proof'];MID='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit';score=proof['metrics']['Accuracy']
assert proof['passed'] and proof['sample_count']==200 and proof['actual_NPU_inference'] and proof['original_protocol_final_qualification_complete'] and proof['formal_alignment_within_5pct'] and not proof['error_indices']
assert done['actual_returncode']==done['model_returncode']==done['verifier_returncode']==x['controller_completion']['actual_returncode']==0
assert proof['all1951_own_source_actual_NPU_SHA256_verified'] and proof['all2011_own_source_SHA_verified_with60_original_unused_sharedKV'] and proof['all_own_source_complete_no_foreign_or_placeholder_tensors'] and proof['all_full_head_rows_independently_recomputed'] and proof['all_native_KV_values_losslessly_reconstructed']
assert all(Decimal(v)<=Decimal('.05') for v in proof['relative_error_to_current_A100_D'].values())
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(book)=='6d73108e181887c9dfd95b7c851981c1ac316238705fe37e8c5bf59c15dc9a82';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
 wb=ET.fromstring(z.read('xl/workbook.xml'));first=wb.find('s:sheets/s:sheet',ns);sheetname=first.get('name');rid=first.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id');rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(n.get('Target') for n in rels if n.get('Id')==rid);sheetxml=target.lstrip('/') if target.startswith('/') else 'xl/'+target
 strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else [];vals={}
 for c in ET.fromstring(z.read(sheetxml)).findall('.//s:c',ns):
  v=c.findtext('s:v',namespaces=ns);vals[c.get('r')]=strings[int(v)] if c.get('t')=='s' else ''.join(c.find('s:is',ns).itertext()) if c.get('t')=='inlineStr' else v
ids={v:int(k[1:]) for k,v in vals.items() if k.startswith('C') and k!='C1' and v};assert len(ids)==700 and ids[MID]==239 and not vals.get('G239') and vals['B239']=='image-text-to-text' and Decimal(vals['D239'])==Decimal('.7850')
old=json.loads((P/'CoderGemma4E4B_verified176_incremental_real_results_export_20261117.json').read_bytes());assert len(old['records'])==old['measured_count']==176
for r in old['records']:
 i=ids[r['model_id']];assert vals[f'B{i}']==r['task'] and Decimal(vals[f'D{i}'])==Decimal(r['current_D']) and Decimal(vals[f'G{i}'])==Decimal(r['current_G'])
H=P/'DeadbydawnGemma4E2B_local_result_sync_verified177_20261117';H.mkdir(exist_ok=False);backup=H/'original_user_workbook.xlsx';shutil.copy2(book,backup);st=book.stat()
p=dict(target=str(book),source_backup=str(backup),expected_before_sha256=sha(book),sheet_name=sheetname,sheet_xml=sheetxml,last_model_row=max(ids.values()),changes=[dict(model_id=MID,task='image-text-to-text',row=239,cell='G239',old_value=None,new_value=score,A100_full_precision_cached_D=vals['D239'],relative_error=str(abs(Decimal(vals['D239'])-Decimal(str(score)))/abs(Decimal(vals['D239']))),new_real_NPU200=True,A100_reference_from_user_confirmed_D=True,report_dir=done['accepted'][-1]['report_dir'],formal_proof_sha256=x['formal_proof_SHA256'])],real_measured_count=177,numeric_within5pct=162,numeric_outside5pct=15,complete_weights_pending_formal=213,confirmed_missing=310,original_protocol_qualified=63,model_blank_count=523,all_prior176_C_B_D_G_checked=True,complete_numeric_verifier_passed=True,local_stat_before=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),live_model_NF_or_FLOCK_at_evidence_read=x['actual_procs'],actual_FLOCK_at_evidence_read=x['FLOCK'],this_preview_is_not_production_publication_and_still_requires_actual_no_NN_NF_FLOCK_gap=True)
(H/'plan.json').write_bytes(json.dumps(p,ensure_ascii=False,indent=2).encode())
s=(P/'coder_gemma4_e4b_workbook_cell_verified176_20261117.mjs').read_text(encoding='utf8').replace('B210:G214','B237:G241');(P/'deadbydawn_gemma4_e2b_workbook_cell_verified177_20261117.mjs').write_bytes(s.encode())
print(json.dumps(dict(actual_proof_ready=True,score=score,formal_forward_count=proof['actual_forward_count'],formal_proof_SHA256=x['formal_proof_SHA256'],plan=str(H/'plan.json'),empty=proof['empty_indices'],invalid=proof['invalid_indices'],truncated=proof['truncated_indices'],no_production_write_yet=True)))

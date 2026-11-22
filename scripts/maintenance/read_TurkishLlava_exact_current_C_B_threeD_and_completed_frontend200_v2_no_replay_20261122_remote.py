"""Resolve actual unique workbook identity after complete own input proof; no replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,zipfile,xml.etree.ElementTree as E
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from batch_910b_support import workbook_rows
MID='ytu-ce-cosmos/Turkish-LLaVA-v0.1';book=R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx';F=B/'TurkishLlava_own_saved_Transformers_auto_FP16_frontend200_meta_no_NN_v2_exact_own_template_20261122'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
rows,sheet,_=workbook_rows(book);matches=[r for r in rows if r['model_id']==MID];assert len(matches)==1;ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:root=E.fromstring(z.read(sheet));cells={c:root.find('.//s:c[@r="'+c+str(matches[0]['row'])+'"]',ns).findtext('s:v',namespaces=ns) for c in ['D','G'] if root.find('.//s:c[@r="'+c+str(matches[0]['row'])+'"]',ns) is not None}
binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');current=Path(binding['current_user_workbook_D_reference_snapshot']);original=Path(binding['local_A100_reference_snapshot']);assert sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256'] and sha(original)==binding['local_A100_reference_snapshot_sha256'];front=js(F/'own_original_frontend200.json');assert len(front)==200 and [r['index'] for r in front]==list(range(200));headers=js(F/'own_weight_files_and_headers.json');meta=js(F/'own_original_legacy_mapping_meta_review.json')
print(json.dumps(dict(model_id=MID,workbook_SHA256=sha(book),current_unique_C_row=matches[0],actual_current_D_and_G_cells=cells,current_local_D=js(current)['models'][MID],preserved_original_D=js(original)['models'][MID],state=js(D/'state.json')['models'][MID],own_frontend200_SHA256=sha(F/'own_original_frontend200.json'),own_frontend200_bytes=(F/'own_original_frontend200.json').stat().st_size,completed_actual_progress=js(F/'progress.json'),late_metadata_failure=js(F/'actual_completion.json'),all200_own_independent_complete_pixels_and_tokens_already_verified=all(r['independent_own_Rust_tokens_and_TV2_uint8_numpy_F32_all_values_exact'] for r in front),own_image_IDs=sorted({r['image_token_id'] for r in front}),own_first_full_text=front[0]['text'],own_header_summary={k:v for k,v in headers.items() if k!='headers'},own_original_mapping_meta_review=meta,no_model_NN_or_CPU_frontend200_replay=True,no_workbook_source_or_current_GLM_mutation=True),ensure_ascii=False))

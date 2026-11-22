"""Meaningful isolated real-workbook tests; never change production files."""
import pathlib,json,copy,zipfile,io,re,hashlib,shutil
from verified_A100_G_cell_writer_20261103 import write_verified_A100_G
from batch_910b_support import workbook_rows,digest
R=pathlib.Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';base=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';home=base/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122/unsloth--gemma-4-E4B-it-MLX-8bit';root=home/'tests';root.mkdir(exist_ok=False);book=pathlib.Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');original=book.read_bytes();manifest=json.loads((home/'verified_candidate.json').read_text());decision=json.loads(pathlib.Path(manifest['authorization']).read_text());decision.update(origin='isolated_test_fixture',test_fixture_only=True,test_root=str(root));auth=root/'fixture_authorization.json';auth.write_text(json.dumps(decision));passed=[]
def patch(data,cell,replacement):
 with zipfile.ZipFile(io.BytesIO(data)) as z:infos=z.infolist();parts={i.filename:z.read(i.filename) for i in infos};comment=z.comment
 rows,sheet,_=workbook_rows(book);text=parts[sheet].decode();pattern=r'<(?:[\w.-]+:)?c\b[^>]*\br="'+cell+r'"[^>]*(?:/>|>.*?</(?:[\w.-]+:)?c>)';text,n=re.subn(pattern,lambda _:replacement,text,count=1,flags=re.S);assert n==1;parts[sheet]=text.encode();out=io.BytesIO()
 with zipfile.ZipFile(out,'w') as z:
  z.comment=comment
  for i in infos:z.writestr(i,parts[i.filename])
 return out.getvalue()
rows,sheet,_=workbook_rows(book);row=next(x for x in rows if x['model_id']==manifest['model_id'])['row']
def case(name,mutate=None,wrong_model=None,score=None,manifest_edit=None,expected_task=manifest['task'],stale=False,success=False):
 score=manifest['new_score'] if score is None else score
 folder=root/name;folder.mkdir();target=folder/'book.xlsx';target.write_bytes(mutate(original) if mutate else original);m=copy.deepcopy(manifest)
 if manifest_edit:manifest_edit(m)
 mp=folder/'manifest.json';mp.write_text(json.dumps(m));expected=digest(target)
 if stale:target.write_bytes(target.read_bytes()+b'\n')
 before=target.read_bytes()
 try:result=write_verified_A100_G(target,999,wrong_model or manifest['model_id'],score,folder/'backup',expected_task=expected_task,decision_path=auth,expected_book_hash=expected,manifest_path=mp)
 except (AssertionError,RuntimeError,ValueError,FileNotFoundError) as e:
  assert not success,(name,str(e));assert target.read_bytes()==before;passed.append({'name':name,'unsafe_write_rejected_before_mutation':True,'reason':str(e)});return
 assert success,name;assert result['row']==row and result['row_relocated'] and not result['old_nonempty_G_replaced']
 with zipfile.ZipFile(io.BytesIO(before)) as old,zipfile.ZipFile(target) as new:
  assert old.namelist()==new.namelist() and [x for x in old.namelist() if old.read(x)!=new.read(x)]==[sheet]
 passed.append({'name':name,'safe_unique_C_relocation_blank_G_update_and_other_ZIP_members_preserved':True})
case('valid_unique_C_blank_only_G',success=True)
case('wrong_model',wrong_model='wrong/model')
case('unreviewed_score',score=.336)
case('wrong_task',expected_task='image-to-text')
case('stale_workbook_hash',stale=True)
case('wrong_old_score',manifest_edit=lambda m:m.update(expected_old_score='0.123'))
other=next(x for x in rows if x['model_id']!=manifest['model_id'] and x.get('row',0)>1)['row']
case('duplicate_unique_C',mutate=lambda b:patch(b,'C'+str(other),'<c r="C'+str(other)+'" t="inlineStr"><is><t>'+manifest['model_id']+'</t></is></c>'))
case('G_formula_guard',mutate=lambda b:patch(b,'G'+str(row),'<c r="G'+str(row)+'"><f>1+1</f><v>0.155</v></c>'))
case('A100_zero_undefined_guard',mutate=lambda b:patch(b,'D'+str(row),'<c r="D'+str(row)+'" t="n"><v>0</v></c>'))
case('current_A100_5pct_guard',mutate=lambda b:patch(b,'D'+str(row),'<c r="D'+str(row)+'" t="n"><v>0.1</v></c>'))
case('proof_hash_manifest_guard',manifest_edit=lambda m:m.update(proof_sha256='0'*64))
assert book.read_bytes()==original
print(json.dumps({'passed':True,'test_count':len(passed),'production_workbook_unchanged':True,'tests':passed},ensure_ascii=False,indent=2))

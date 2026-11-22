"""Bind genuine completed own200 acceptance to the reviewed original-lock dual transaction."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from decimal import Decimal
import ast,base64,hashlib,json,zipfile,xml.etree.ElementTree as ET
V=Path(__file__).resolve().parent;BASE=Path('D:/code-lv/.tmp/cd22s1');L=V/'Gemma4E4BMLX8_local_result_sync_verified185_20261122';MID='unsloth/gemma-4-E4B-it-MLX-8bit'
R='/root/projects/comperror';B=R+'/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B+'/Gemma4E4BMLX8_verified185_publication_review_inputs_v1_20261122';H=B+'/Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122'
sha=lambda b:hashlib.sha256(b).hexdigest()
x=json.loads((V/'R_Gemma4MLX8_verified185_exact_completed_publication_inputs_20261122.json').read_bytes());p=x['formal_proof'];q=json.loads((L/'plan.json').read_bytes());roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller'}
assert not x['FLOCK'] and not [v for v in x['actual_procs'] if v['role'] in roles]
assert x['completion']['actual_returncode']==x['completion']['model_returncode']==x['completion']['verifier_returncode']==x['controller_completion']['actual_returncode']==0
assert p['passed'] and p['sample_count']==len(p['numeric_sample_checks'])==200 and p['metrics']=={'Accuracy':.84} and p['actual_forward_count']==13383 and p['NPU_profiler_kernel_events']==6698
assert p['actual_NPU_inference'] and not p['historical_record_reuse'] and p['original_protocol_final_qualification_complete'] and p['formal_alignment_within_5pct'] and all(Decimal(v)<=Decimal('.05') for v in p['relative_error_to_current_A100_D'].values())
assert p['all2130_own_source_actual_NPU_and_unused_SHA256_verified'] and p['all54_unused_shared_KV_source_preserved'] and p['all_native_KV_values_losslessly_reconstructed'] and p['all_full_head_rows_independently_recomputed'] and not any(p[k] for k in ['error_indices','invalid_indices','empty_indices','truncated_indices'])
assert x['current_row']==dict(row=664,model_id=MID,task='image-text-to-text',score=None) and x['formal_proof_SHA256']=='3cf18242f6f6276cfe552ccbb5d2bfd54ea56ad0863785b21097e26404bbad8d'
assert x['prior184_export_SHA256']=='adf185dd40b3bd364df1c425ea790d64673ca6a51224203a67ea8f56843c36b8' and x['prior184_stat_SHA256']=='1549f33251985bdd1c1c35bb3752e4dc992ae9f87be700ae3076c8ee899147ee'
target=Path(q['target']);st=target.stat();assert sha(target.read_bytes())==q['expected_before_sha256']=='0104ccaaab17fef845cfd172a7274c0a7c20faf51f858f21c0aef4e4a71992a1'
assert q['local_stat_before']==dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
values=json.loads((L/'artifact_authored_G_values.json').read_bytes());assert values[662][0]==.84
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(L/'artifact_authoring_candidate.xlsx') as z:
 root=ET.fromstring(z.read(q['sheet_xml']));c=next(c for c in root.findall('.//s:c',ns) if c.get('r')=='G664');assert c.get('t') not in ['s','inlineStr'] and Decimal(c.findtext('s:v',namespaces=ns))==Decimal('.84')
q['changes'][0]['formal_proof_sha256']=x['formal_proof_SHA256'];q.update(complete_numeric_verifier_passed=True,actual_completed_acceptance_and_all_quiet_verified=True)
q['models']={MID:dict(job=x['job'],controller=x['controller'],job_SHA256=x['job_SHA256'],true3_score=1.,true3_proof_SHA256=x['true3_proof_SHA256'],expected_forward_count=13383,expected_kernels=6698,expected_empty=[],expected_invalid=[],expected_truncated=[])}
q.update(test_template='scripts/tests/verified_Gemma4E2B_manifest_task_writer_11_cases_template_20261116.py',test_template_SHA256='8e595053cedf0ff3fc46b1f1a76fde787ed268ac190bc037e7521882e048e318')
artifact_names=['before_664.png','after_664.png','artifact_authoring_candidate.xlsx','artifact_authored_G_values.json','formula_scan.ndjson']
q['artifact_tool_actual_authoring_evidence']={n:dict(path=I+'/ArtifactTool/'+n,SHA256=sha((L/n).read_bytes())) for n in artifact_names}
q['actual_before_after_PNG_visually_viewed']=True
(L/'plan.json').write_bytes(json.dumps(q,ensure_ascii=False,indent=2).encode());qb=json.dumps(q,ensure_ascii=False,indent=2).encode();qh=sha(qb);(L/'reviewed_server_publication_inputs.json').write_bytes(qb)
pub='scripts/publish_Gemma4E4BMLX8_verified185_single_original_lock_dual_transaction_v1_20261122.py';pb=(V/Path(pub).name).read_bytes();ph=sha(pb);assert ph=='567f27b07fde53904eb6916f7519a8a5980a7ee6078967743361e22fd3f6ed77';ast.parse(pb)
old=(BASE/'scripts/run_PrithivGemma4E4B_verified180_dual_publication_actual_completion_v1_20261117.py').read_text(encoding='utf-8')
supervisor=old.replace('PrithivGemma4E4B','Gemma4E4BMLX8').replace('verified180','verified185').replace('20261117','20261122').replace('b6630776c326931926e9c6bbe58da77619875954086929050add7090b7863123',ph).replace('99cc4c99cf49e341171879f9a430fea357782c4cd28fdeddc5bbbd9ab406f66a',qh)
assert H not in supervisor and 'Gemma4E4BMLX8_verified185_publication_review_inputs_v1_20261122' in supervisor;ast.parse(supervisor)
sn='scripts/run_Gemma4E4BMLX8_verified185_dual_publication_actual_completion_v1_20261122.py';(V/Path(sn).name).write_bytes(supervisor.encode())
reader=(BASE/'scripts/code_publication_windows/read_Prithiv_verified180_server_book_for_local_CAS_20261117_remote.py').read_text(encoding='utf-8').replace('PrithivGemma4E4B','Gemma4E4BMLX8').replace('verified180','verified185').replace('20261117','20261122').replace('original_HF_BF16','original_Transformers_BF16').replace("['done']==180","['done']==185").replace('prithivMLmods/gemma-4-E4B-it-Uncensored-MAX',MID)
rn='read_Gemma4MLX8_verified185_server_book_for_local_CAS_20261122_remote.py';ast.parse(reader);(V/rn).write_bytes(reader.encode())
sync=(BASE/'scripts/code_publication_windows/sync_Prithiv_verified180_local_exact_zip_CAS_20261117.py').read_text(encoding='utf-8').replace("P=Path(__file__).parent;H=P/", "P=Path(__file__).parent;H=P/").replace('PrithivGemma4E4B','Gemma4E4BMLX8').replace('verified180','verified185').replace('20261117','20261122').replace('G572','G664').replace("['done']==180","['done']==185").replace('all_prior179','all_prior184')
sync=sync.replace('R_Gemma4E4BMLX8_verified185_server_book_for_local_CAS_20261122.json','R_Gemma4MLX8_verified185_server_book_for_local_CAS_20261122.json')
cn='sync_Gemma4MLX8_verified185_local_exact_zip_CAS_20261122.py';ast.parse(sync);(V/cn).write_bytes(sync.encode())
entries={pub:pb,sn:supervisor.encode(),'scripts/maintenance/'+rn:reader.encode(),'scripts/code_publication_windows/'+cn:sync.encode(),'scripts/code_publication_windows/'+Path(__file__).name:Path(__file__).read_bytes()}
for n in ['prepare_Gemma4MLX8_verified185_publisher_and_ArtifactTool_authoring_20261122.py','gemma4_MLX8_workbook_cell_verified185_20261122.mjs','build_Gemma4MLX8_verified185_completed_publication_reader_20261122.py']:
 entries['scripts/code_publication_windows/'+n]=(V/n).read_bytes()
payload={r:dict(SHA256=sha(b),base64=base64.b64encode(b).decode()) for r,b in entries.items()}
launch="""import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier','legacy_controller']]
P=json.loads(base64.b64decode(__PAYLOAD__))
for rel,e in P.items():
 raw=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==e['SHA256'];p=R/rel
 assert rel.startswith('scripts/') and '..' not in Path(rel).parts
 if p.suffix=='.py':ast.parse(raw)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=Path(__I__);assert not I.exists();I.mkdir();(I/'ArtifactTool').mkdir()
for n,e in json.loads(base64.b64decode(__ARTIFACTS__)).items():
 raw=base64.b64decode(e['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==e['SHA256'];(I/'ArtifactTool'/n).write_bytes(raw)
qb=base64.b64decode(__Q__,validate=True);assert hashlib.sha256(qb).hexdigest()==__QH__;(I/'reviewed_inputs.json').write_bytes(qb)
for n,h in __CANONICAL__.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:
 p=subprocess.Popen(['python3','-B',str(R/__SN__)],cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(PID=p.pid,reviewed_inputs_SHA256=__QH__,publisher_SHA256=__PH__,supervisor_SHA256=__SH__,source=str(R/__SN__),publication_started_not_claimed_complete=True)
(I/'background_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v))
"""
artifacts={n:dict(SHA256=sha((L/n).read_bytes()),base64=base64.b64encode((L/n).read_bytes()).decode()) for n in artifact_names}
for a,b in {'__PAYLOAD__':repr(base64.b64encode(json.dumps(payload).encode()).decode()),'__I__':repr(I),'__ARTIFACTS__':repr(base64.b64encode(json.dumps(artifacts).encode()).decode()),'__Q__':repr(base64.b64encode(qb).decode()),'__QH__':repr(qh),'__CANONICAL__':repr(x['current_canonical_SHA256']),'__SN__':repr(sn),'__PH__':repr(ph),'__SH__':repr(sha(supervisor.encode()))}.items():launch=launch.replace(a,b)
ast.parse(launch);(V/'launch_Gemma4MLX8_verified185_reviewed_single_original_lock_dual_publication_20261122_remote.py').write_bytes(launch.encode())
(V/'Gemma4MLX8_verified185_transaction_primary_sources_20261122.json').write_text(json.dumps({k:sha(v) for k,v in entries.items()},indent=2),encoding='utf-8')
print(json.dumps(dict(publisher_SHA256=ph,reviewed_inputs_SHA256=qh,supervisor_SHA256=sha(supervisor.encode()),actual_ArtifactTool_author_and_PNG_checked=True,full200_NF0_threeD_pass=True,primary_sources=len(entries),no_production_write_yet=True)))

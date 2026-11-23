"""Only correct the independent conversation reference to the original template."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
V=Path(__file__).resolve().parent
old='prepare_kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123.py'
new='prepare_kanana_own_original_BF16_dedicated_frontend200_v2_original_whitespace_20261123.py'
s=(V/old).read_text(encoding='utf8')
a="text = bos+'\\n'+INTRO+'\\nHuman: <image>Human: '+prompt+'\\nAI: '"
b="text = bos+INTRO+'\\nHuman: <image>\\nHuman: '+prompt+'\\n\\nAI: '"
assert s.count(a)==1;s=s.replace(a,b)
oldhome='Kanana_own_original_BF16_dedicated_frontend200_v1_heartbeat1627_20261123'
newhome='Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123'
assert s.count(oldhome)==1;s=s.replace(oldhome,newhome)
def fs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
before=fs((V/old).read_text(encoding='utf8'));after=fs(s)
assert {k:v for k,v in before.items() if k!='independent_ids'}=={k:v for k,v in after.items() if k!='independent_ids'}
assert len(before)==len(after)
(V/new).write_text(s,encoding='utf8')
builder=(V/'build_Kanana_own_frontend200_binding_and_source_v2_exact_empty_cell_heartbeat1627_20261123.py').read_text(encoding='utf8')
assert builder.count(old)==1;builder=builder.replace(old,new).replace('Kanana_own_frontend200_v1_primary_source_SHA_heartbeat1627_20261123.json','Kanana_own_frontend200_v2_primary_source_SHA_20261123.json').replace('stage_Kanana_own_frontend200_v1_heartbeat1627_20261123_remote.py','stage_Kanana_own_frontend200_v2_original_whitespace_20261123_remote.py')
needle="for b in sources.values():ast.parse(b)"
assert builder.count(needle)==1
builder=builder.replace(needle,"sources['scripts/code_publication_windows/'+"+repr(Path(__file__).name)+"]=(V/"+repr(Path(__file__).name)+").read_bytes()\n"+needle)
bn='build_Kanana_own_frontend200_binding_and_source_v3_original_Jinja_reference_20261123.py'
ast.parse(builder);(V/bn).write_text(builder,encoding='utf8')
review=dict(passed=True,change='Independent reference only: no newline after BOS; original image/text boundary newline; original blank line before generation prefix',
            original_own_processor_or_prompt_or_model_untouched=True,all_other_functions_AST_identical=True,
            old_source_SHA256=hashlib.sha256((V/old).read_bytes()).hexdigest(),new_source_SHA256=hashlib.sha256(s.encode()).hexdigest(),old_failed_stage_preserved=True,no_NN_replay=True)
(V/'Kanana_frontend_v2_reference_only_source_review_20261123.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
r=subprocess.run([sys.executable,'-X','utf8','-B',str(V/bn)],capture_output=True)
(V/'Kanana_frontend_binding_builder_v3_actual_stdout.log').write_bytes(r.stdout)
(V/'Kanana_frontend_binding_builder_v3_actual_stderr.log').write_bytes(r.stderr)
print(r.stdout.decode('utf8'));print(r.stderr.decode('utf8'));raise SystemExit(r.returncode)

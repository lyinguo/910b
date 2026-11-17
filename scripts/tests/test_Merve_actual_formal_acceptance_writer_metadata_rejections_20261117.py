import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from verified_Merve_actual_formal_acceptance_G_cell_writer_schema_v2_20261117 import check_formal_acceptance_metadata
B=Path('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct');H=B/'Merve_completed_original200_formal_independent_acceptance_v4_native_media_reference_no_NN_replay_20261117';completed=json.loads((H/'completion.json').read_bytes());job=json.loads((H/'job.json').read_bytes());p=json.loads(Path(completed['accepted'][-1]['proof']).read_bytes());source=json.loads((Path(job['source_original_NPU200_job'])/'completion.json').read_bytes());assert check_formal_acceptance_metadata(completed,p,job,source);cases=[]
for target,key,value in [('completed','phase','failed'),('completed','verifier_returncode',1),('completed','actual_new_model_forwards_this_phase',1),('proof','actual_NPU_inference',False),('proof','sample_count',3),('proof','formal_alignment_within_5pct',False),('proof','all200_input_pixels_exact_original_native_media_decode_verified',False),('proof','original_full_numeric_model_checks_unchanged',False),('job','model_id','different/model'),('source','model_returncode',1)]:
 values=[copy.deepcopy(x) for x in [completed,p,job,source]];i={'completed':0,'proof':1,'job':2,'source':3}[target];values[i][key]=value;cases.append(values)
for values in cases:
 try:check_formal_acceptance_metadata(*values)
 except AssertionError:pass
 else:raise AssertionError('Invalid independent acceptance admitted')
print(json.dumps(dict(passed=True,test_count=1+len(cases),no_production_or_original_evidence_mutation=True,frozen_shared_writer_unchanged=True)))

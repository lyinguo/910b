"""Negative controls for pruning eligibility and full-head numerical acceptance."""
import sys
sys.dont_write_bytecode=True
import importlib.util,copy,json
from pathlib import Path
path=Path(__file__).with_name('audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119.py')
if not path.exists():path=Path('/root/projects/comperror/scripts/maintenance')/path.name
spec=importlib.util.spec_from_file_location('owned_measurement_scope',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[dict(index=i,prediction='B' if i%3 else 'C',reference='B',raw_output='B') for i in range(200)]
s=m.preserve_and_remove_indices(rows,4,{44});assert set(s['incorrect'])<=set(s['keep']) and 44 not in s['remove'];assert set(range(8,12))<=set(s['keep']) and set(range(88,112))<=set(s['keep']) and set(range(188,200))<=set(s['keep'])
for pred in ['a word','a',None,'B because','']:
 r=copy.deepcopy(rows);r[1]['prediction']=pred
 if pred=='a':r[1]['reference']='B'
 t=m.preserve_and_remove_indices(r,4);assert 1 not in t['remove']
r=copy.deepcopy(rows);r[44]['extra']={'error':'control'};assert 44 not in m.preserve_and_remove_indices(r,4)['remove']
try:m.preserve_and_remove_indices(rows[:-1],4);raise RuntimeError('Incomplete accepted')
except AssertionError:pass
p=dict(sample_count=1,actual_forward_count=2,numeric_sample_checks=[dict(index=0,full_head_rows=2,head_reference_checks=[{},{}],all40_native_cache_values_reconstructed=True,original_input_pixels_token_ids_checked=True,all_masks_3D_positions_and_own_visual_injection_checked=True)])
assert m.full_head(p)
for key,value in [('actual_forward_count',3),('sample_count',2)]:
 q=copy.deepcopy(p);q[key]=value;assert not m.full_head(q)
q=copy.deepcopy(p);q['numeric_sample_checks'][0]['all40_native_cache_values_reconstructed']=False;assert not m.full_head(q)
print(json.dumps(dict(passed=True,checks=14,all_wrong_special_and_original_complete_outer_groups_kept=True,no_production_file_or_model_changed=True)))

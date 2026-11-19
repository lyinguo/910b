"""Prepare a new archive builder; preserve v1's actual path-guard refusal."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parent
old=P/'build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_1138_20261119.py'
s=old.read_text(encoding='utf8')
needle="static[home+'/'+name]=dict(path=v['path'],SHA256=v['SHA256'])"
assert s.count(needle)==1
s=s.replace(needle,"static[Path(home).name+'/'+name]=dict(path=v['path'],SHA256=v['SHA256'])")
needle="for name in ['build_moondream3_original_HF_load_precision_v2_json_sets_1116_20261119.py'"
assert s.count(needle)==1
s=s.replace(needle,"for name in ['build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_1138_20261119.py','prepare_archive_builder_v2_relative_static_paths_1142_20261119.py','build_moondream3_original_HF_load_precision_v2_json_sets_1116_20261119.py'")
s=s.replace("no_deletion_this_cycle=True,daily_code_already_pushed_commit=","original_archive_v1_path_guard_refusal_preserved=dict(original_stdout_file='R_archive_verified180_Moondream_loading_KV_actual_1138_20261119.json',original_stderr_file='R_archive_verified180_Moondream_loading_KV_actual_1138_20261119.json.stderr',scope='Absolute archive label rejected inside() before facts/pointer publication; only copy destination label changed to relative, original source and canonical unchanged'),no_deletion_this_cycle=True,daily_code_already_pushed_commit=")
ast.parse(s)
new=P/'build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_v2_relative_static_1142_20261119.py'
assert not new.exists();new.write_text(s,encoding='utf8',newline='\n')
print(new.name)

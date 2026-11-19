"""Avoid recursive embedding of already archived large transport/plan files."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parent
old=P/'build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_v2_relative_static_1142_20261119.py'
s=old.read_text(encoding='utf8')
needle="raw=path.read_bytes();assert len(raw)<8*1024**2"
assert s.count(needle)==1
s=s.replace(needle,"raw=path.read_bytes()")
s=s.replace("base64=base64.b64encode(raw).decode(),role='Exact local source/readback/result and original failures/warnings; no model report package or learned arrays'","**({'base64':base64.b64encode(raw).decode()} if len(raw)<8*1024**2 else {}),role='Exact local source/readback/result and original failures/warnings; large previously preserved plans/transports retain path/bytes/SHA without recursive embedding'")
s=s.replace("for name in ['build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_1138_20261119.py'","for name in ['build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_v2_relative_static_1142_20261119.py','prepare_archive_builder_v3_bounded_original_packets_1145_20261119.py','build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_1138_20261119.py'")
s=s.replace("original_archive_v1_path_guard_refusal_preserved=","original_local_archive_v2_large_recursive_packet_guard_refusal_preserved=True,original_archive_v1_path_guard_refusal_preserved=")
ast.parse(s)
new=P/'build_archive_verified180_Moondream_actual_NPU_loading_KV_candidate_v3_bounded_originals_1145_20261119.py'
assert not new.exists();new.write_text(s,encoding='utf8',newline='\n')
print(new.name)

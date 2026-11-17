"""Migrate this checkpoint's frozen sources and inspect interfaces without NN."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,traceback,importlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Phi3VForCausalLM/microsoft/Phi-3.5-vision-instruct');S=R/'universal_eval/model/phi35_own_original_source_20261117';H=B/'Phi35_own_frozen_source_primary_migration_and_current_HF_meta_interface_preflight_no_NN_v2_exact_existing_frontend_1916_20261117'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir('/root');sys.path[:0]=[str(R),str(R/'scripts')];H.mkdir(exist_ok=False);originals={};S.mkdir(exist_ok=True)
 if not (S/'__init__.py').exists():(S/'__init__.py').write_bytes(b'')
 for name in ['configuration_phi3_v.py','modeling_phi3_v.py','processing_phi3_v.py']:
  src=W/name;raw=src.read_bytes();ast.parse(raw);dst=S/name
  if dst.exists():assert dst.read_bytes()==raw
  else:dst.write_bytes(raw)
  originals[name]=dict(original_path=str(src),source_primary_path=str(dst),SHA256=sha(src),exact_original_bytes_preserved=sha(src)==sha(dst))
 oldfront=B/'Phi35_own_original_BF16_4096_native_candidate_frontend200_v3_original_dataset_prompt_no_NN_20261117/own_original_chat_frontend200.jsonl';assert sha(oldfront)=='8efba8e2372fde52937506598e89304eb61330d5c65fdd71bd9c77d4220e1d86'
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),model_id='microsoft/Phi-3.5-vision-instruct',source_migration=originals,already_passed_own_frontend200_reused_no_replay=True,frontend200=str(oldfront),frontend_SHA256=sha(oldfront),no_model_forward_no_NPU_or_learned_CPU=True,no_install_download_stub_or_shared_library_patch=True)
 try:
  import torch,transformers
  result['runtime_versions']=dict(torch=torch.__version__,transformers=transformers.__version__)
  config_module=importlib.import_module('universal_eval.model.phi35_own_original_source_20261117.configuration_phi3_v');model_module=importlib.import_module('universal_eval.model.phi35_own_original_source_20261117.modeling_phi3_v');config=config_module.Phi3VConfig.from_json_file(str(W/'config.json'));config._attn_implementation='eager'
  with torch.device('meta'):model=model_module.Phi3VForCausalLM(config)
  records={n:dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for n,t in model.state_dict().items()};assert all(r['device']=='meta' for r in records.values());(H/'native_meta_state_shapes.json').write_bytes(json.dumps(records,ensure_ascii=False,indent=2).encode());result.update(meta_construction_passed=True,meta_state_tensor_count=len(records),no_tensor_values_initialized_or_evaluated=True,own_original_implementation_imported=True)
 except BaseException:result.update(meta_construction_passed=False,interface_error=traceback.format_exc(),required_next_step='Resolve only proven private import/API contract differences, retain originals, then implement independently observed own NPU adapter and full verifier before true3')
 result['actual_diagnostic_returncode']=0;result['ready_for_true_NPU3']=False;(H/'review.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

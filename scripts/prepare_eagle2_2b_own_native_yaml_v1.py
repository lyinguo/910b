"""Bind independently prepared own Eagle2 inputs/source to one standalone NPU YAML."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,copy,importlib,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';MID='nvidia/Eagle2-2B';H=B/'Eagle2_2B_own_native_readiness_v1_1925_20261115';J=B/'Eagle2_2B_own_native_true3_v1_real_FLOCK_1925_20261115'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 import batch_910b as batch
 from batch_910b_support import validate_config
 from A100_alignment_evidence_runner_20261103 import scope
 state=js(D/'state.json');entry=state['models'][MID];assert entry['score'] is None and entry['status']=='needs_adaptation'
 pre=B/'Eagle2_2B_own_frontend200_native_source_preflight_v1_original_Greedy_1920_20261115/review.json';ready=js(pre);assert ready['passed'] and ready['all_200_original_own_image_math_tokens_masks_exact'] and ready['source_tensor_count']==793
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='4acd405375a631143c1852c75265f1df273877a4773b27bc210379007958dad0'
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:
  root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));cells={c.get('r'):c.findtext('s:v',namespaces=ns) for c in root.findall('.//s:c',ns)}
 row=449;serverD=cells['D'+str(row)];assert float(serverD)>0
 snap=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(snap)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
 preserved=js(snap)['models'][MID];local=js(current)['models'][MID];assert preserved['task']==local['task']==entry['task']=='image-text-to-text'
 ds=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';assert sha(ds)==ready['dataset_SHA256']
 boundary='User-confirmed workbook A100 D is the comparison. Saved own Eagle2 YAML/adapter uses BF16, eager, original Greedy outer10-inner1, own fast 448/max12+thumbnail processor/template, natural EOS151645, 256 tokens. Current HF changes Siglip parameter names only; every one of 793 own source tensors maps bijectively. Historical A100 engine/tensor bytes were not captured.'
 rels=['universal_eval/model/adapters/eagle2_2b_own_native_bf16_v1.py','universal_eval/model/adapters/eagle2_1b_own_frontend_v1.py','scripts/verify_eagle2_2b_own_native_bf16_v1.py','scripts/prepare_eagle2_2b_own_native_yaml_v1.py','scripts/prepare_eagle2_2b_own_frontend_native_sources_v1.py','universal_eval/model/eagle2_2b_own_native_20261115/__init__.py','universal_eval/model/eagle2_2b_own_native_20261115/configuration_eagle2_5_vl.py','universal_eval/model/eagle2_2b_own_native_20261115/modeling_eagle2_5_vl.py','universal_eval/model/adapters/internvl35_flash2b_own_frontend_v3_original_TV2_uint8_F32_1518_20261112.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/tinyllava_strict_mc.py','universal_eval/utils/readonly_npz_indexed.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','run.py','universal_eval/orchestrator/pipeline.py']
 for rel in rels:ast.parse((R/rel).read_text())
 sources={rel:sha(R/rel) for rel in rels}
 proto={k:copy.deepcopy(ready[k]) for k in ['model_id','weight_path','own_weight_files','own_assets_SHA256','own_source_manifest','own_source_manifest_SHA256','own_frontend200','own_frontend200_SHA256']}
 proto.update(workbook_D_reference_authorized=True,dataset_path=str(ds),dataset_SHA256=sha(ds),reference_workbook=str(book),reference_workbook_SHA256=sha(book),current_A100_D_values=dict(server=serverD,local=local['A100_D'],preserved_original=preserved['A100_D']),implementation_source_SHA256=sources,static_evidence_SHA256={str(p):sha(p) for p in [pre,snap,current]},historical_source_boundary=boundary,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3)
 protocol=R/'configs/protocols/eagle2_2b_own_native_bf16_v1.json';put(protocol,proto)
 template=js(R/'configs/InternVLChatModel/OpenGVLab--InternVL2-2B-own-native-BF16-2048-v5-actual-kernel-schema-20261115.yaml');template['name']='nvidia--Eagle2-2B-own-native-BF16-256-v1-20261115';template['description']=boundary;template['model']=dict(name=MID,adapter='universal_eval/model/adapters/eagle2_2b_own_native_bf16_v1.py:Eagle2BOwnNativeBF16',model_path=ready['weight_path'],download=False,kwargs=dict(protocol_file=str(protocol),protocol_sha256=sha(protocol),protocol=dict(model_id=MID),npu_devices=[4]));template['baseline']['score']=float(serverD);validate_config(template,MID)
 config=R/'configs/Eagle2_5_VLForConditionalGeneration/nvidia--Eagle2-2B-own-native-BF16-256-v1-20261115.yaml';put(config,template)
 spec=dict(approved=False,review_revision='Eagle2_2B_own_native_BF16_source793_workbookD_20261115',config=str(config.relative_to(R)),verifier='scripts/verify_eagle2_2b_own_native_bf16_v1.py',npu_devices=[4],source_sha256=dict(sources,**{str(config.relative_to(R)):sha(config),str(protocol.relative_to(R)):sha(protocol)}),smoke_timeout_seconds=14400,formal_timeout_seconds=86400,verify_timeout_seconds=21600)
 batch.check_sources(spec)
 for count in [3,200]:
  _,cfg=batch.config_for(spec,entry,count);validate_config(cfg,MID)
 scope(spec,[3],False);importlib.import_module('universal_eval.model.adapters.eagle2_2b_own_native_bf16_v1')
 parent=js(B/'InternVL2_2B_shape_capacity_retention_approved_fresh3_fixed200_v7_real_FLOCK_1535_20261115/job.json');job={k:copy.deepcopy(v) for k,v in parent.items() if k not in ['spec','model_id','stages','created_at','protected_canonical_hashes']};job.update(model_id=MID,spec=spec,stages=[3],created_at=datetime.now(timezone.utc).isoformat(),protected_canonical_hashes={p:sha(p) for p in parent['protected_canonical_hashes']});put(J/'job.json',job)
 result=dict(passed=True,ready_for_original_runner_true3=True,model_id=MID,job=str(J),job_SHA256=sha(J/'job.json'),source_sha256=spec['source_sha256'],standalone_YAML=str(config),protocol=str(protocol),protocol_SHA256=sha(protocol),all_own793_source_and_200_frontends_prepared=True,no_NPU_inference_yet=True,current_A100_D_values=proto['current_A100_D_values'],historical_source_boundary=boundary,protected_canonical_hashes=job['protected_canonical_hashes']);put(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

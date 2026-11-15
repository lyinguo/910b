LOCAL={'current_local_D': '0.8050', 'current_local_book_SHA256': '59fccf6b1a0d3cda2f793756b70e1ac7d05a042e9b789172e6f2d9391432201c', 'row': 432}
"""Prepare own 35-layer E4B-it YAML and genuine NPU3 private job.

Uses this checkpoint's frontend and original saved config. Shared code contains
operators only. A100 comparison uses the user's workbook D; no AA capture claim.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,importlib.util
import yaml
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
MID='mlx-community/gemma-3n-E4B-it-bf16'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma3nForConditionalGeneration')/MID
H=B/'MLXE4BitBF16_own_chat_native_v1_readiness_1210_20261115'
FRONT=B/'MLXE4BitBF16_own_chat_frontend200_assets_1210_20261115'
PARENT=B/'UnslothE2Bit_own_chat_native_approved_fresh3_fixed200_v3_completed_batch_real_FLOCK_1210_20261115'
J=B/'MLXE4BitBF16_own_chat_native_true3_v1_real_FLOCK_1210_20261115'
ADAPTER='universal_eval/model/adapters/mlx_gemma3n_e4bit_bf16_native_v1.py'
VERIFIER='scripts/verify_mlx_gemma3n_e4bit_bf16_native_v1.py'
SELF='scripts/prepare_mlx_gemma3n_e4bit_bf16_native_v1.py'
CFG='configs/Gemma3nForConditionalGeneration/mlx-community--gemma-3n-E4B-it-bf16-native-BF16-chat-2048-v1-20261115.yaml'
PROTO='configs/protocols/mlx_gemma3n_e4bit_bf16_native_v1.json'
LAYOUT='configs/protocols/mlx_gemma3n_e4bit_bf16_own_vision_layout_v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 import batch_910b as batch
 from batch_910b_support import workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 H.mkdir(exist_ok=False)
 completion=js(FRONT/'actual_completion.json');review=js(FRONT/'review.json');assert completion['actual_returncode']==0 and review['passed'] and review['model_id']==MID and review['samples']==200 and review['own_checkpoint_tensor_count']==1676
 assert sha(FRONT/'frontend200.jsonl')==review['frontend_jsonl_SHA256'] and review['all200_independent_F32_pixel_replays_equal']
 front=[json.loads(x) for x in (FRONT/'frontend200.jsonl').read_text().splitlines()];assert [x['index'] for x in front]==list(range(200)) and all(x['own_template_used'] and x['independent_pixel_replay_equal'] and not x['historical_A100_tensor_capture_claimed'] for x in front)
 source=js(FRONT/'source_tensor_headers.json');assert len(source)==1676
 old_yaml=R/'configs/Gemma3nForConditionalGeneration/mlx-community--gemma-3n-E4B-it-bf16.yaml';saved=yaml.safe_load(old_yaml.read_bytes());assert saved['model']['name']==MID and saved['model']['model_path']==str(W) and saved['task_type']=='image-text-to-text' and saved['sample_range']=='0:199' and saved['model']['kwargs']['max_tokens']==2048 and saved['algorithm']['batch_size']==10 and saved['algorithm']['name']=='greedy_inference' and saved['algorithm']['kwargs']['extract_answer']
 protected={str(p):sha(p) for p in [D/'state.json',D/'registry.json',D/'PAUSE',D/'POLICY.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]}
 assert protected['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx']==LOCAL['current_local_book_SHA256']
 rows,sheet,_=workbook_rows(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'));row,=[x for x in rows if x['model_id']==MID];assert row['score'] is None and row['task']=='image-text-to-text'
 import zipfile,xml.etree.ElementTree as E
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx') as z:serverD=E.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 assert serverD==LOCAL['current_local_D']
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');original=Path(binding['local_A100_reference_snapshot']);assert sha(original)=='700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba';originalD=js(original)['models'][MID]['A100_D']
 decision=D/'user_decisions/workbook_A100_D_is_comparison_reference_2130_20261114/decision.json';assert sha(decision)=='22cbb4581a69875f643902001babfeffa694a2275e0f894a447866e356bccf3d'
 template=copy.deepcopy(js(PARENT/'job.json'));parent_cfg=js(R/template['spec']['config']);parentp=js(parent_cfg['model']['kwargs']['protocol_file']);layout=copy.deepcopy(js(parentp['vision_graph_layout']));prefix='model.vision_tower.timm_model.'
 for key,rec in layout['own_state'].items():assert source[prefix+key]['shape']==rec['shape'] and source[prefix+key]['dtype']=='BF16'
 assert all(name in source for group in layout['actual_parameter_aliases'] for name in group)
 layout.update(model_id=MID,scope='Weight-free architecture graph with parameter names and shapes checked against this own E4B-it checkpoint; no learned values copied')
 put(R/LAYOUT,layout)
 code=[ADAPTER,VERIFIER,SELF,'scripts/prepare_mlx_gemma3n_e4bit_bf16_own_frontend200.py','universal_eval/model/adapters/gemma3n_E4B_actual_original_HighTen_BF16_native_v5_complete_failure_records_1605_20261113_adapter.py','universal_eval/model/adapters/gemma3n_E4B_HighTen_own_parameterized_BF16_vision_native_v1_after_reboot_20261113.py','universal_eval/algorithm/gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113.py','universal_eval/algorithm/native_explicit_mc_v2.py','universal_eval/algorithm/inference.py','universal_eval/utils/readonly_npz_indexed.py','scripts/verify_Gemma3n_E4B_actual_original_HighTen_BF16_native_v4_full_empty_raw_schema_20261113.py','run.py','universal_eval/orchestrator/pipeline.py',LAYOUT,str(Path(parentp['vision_graph']).relative_to(R))]
 protocol=dict(model_id=MID,weight_path=str(W),report_task='image-text-to-text',dataset_path=str(R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'),dataset_SHA256=review['dataset_SHA256'],implementation_source_SHA256={n:sha(R/n) for n in code},static_evidence_SHA256={str(p):sha(p) for p in [FRONT/'review.json',FRONT/'frontend200.jsonl',FRONT/'source_tensor_headers.json',decision,original]},own_runtime_assets_SHA256=review['own_runtime_assets_SHA256'],own_weight_files=review['own_weight_shards'],own_frontend200=str(FRONT/'frontend200.jsonl'),vision_graph=parentp['vision_graph'],vision_graph_layout=str(R/LAYOUT),hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,per_sample_lossless_space_estimate_bytes=2*1024**3,current_A100_D_values=dict(server=serverD,local=LOCAL['current_local_D'],preserved_original=originalD),workbook_D_reference_authorized=True,saved_original_configuration_used=True,saved_original_YAML=str(old_yaml),saved_original_YAML_SHA256=sha(old_yaml),preserved_original_snapshot=str(original),preserved_original_snapshot_SHA256=sha(original),reference_workbook='/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx',reference_workbook_SHA256=protected['/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'],historical_source_boundary='Saved own vLLM/Greedy 2048/outer10 configuration and own instruction template reconstructed as native HF SDPA BF16/inner serial1 on 910B. A100 comparison uses the user workbook D. Historical A100 engine bytes/tensors were not captured.')
 put(R/PROTO,protocol);cfg=copy.deepcopy(parent_cfg);cfg.update(name=Path(CFG).stem,description=protocol['historical_source_boundary']);cfg['model'].update(name=MID,adapter=ADAPTER+':MLXGemma3nE4BitBF16',model_path=str(W));cfg['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO),protocol=dict(model_id=MID));cfg['baseline'].update(score=float(serverD));put(R/CFG,cfg)
 spec=copy.deepcopy(template['spec']);spec.update(approved=False,review_revision='20261115_Unsloth_E4B_it_own_chat_native35_workbook_D',config=CFG,verifier=VERIFIER,npu_devices=[4],source_sha256={n:sha(R/n) for n in code+[CFG,PROTO,'universal_eval/algorithm/tinyllava_strict_mc.py']})
 for k in ['alignment_approval_review','alignment_approval_review_sha256','protocol_file','protocol_sha256']:spec.pop(k,None)
 template.update(model_id=MID,spec=spec,stages=[3],protected_canonical_hashes=protected,write_workbook=False,created_at=datetime.now(timezone.utc).isoformat())
 J.mkdir(exist_ok=False);put(J/'job.json',template);scope(spec,[3],False);batch.validate_config(cfg,MID);batch.check_sources(spec);entry=js(D/'state.json')['models'][MID];c3,e3=batch.config_for(spec,entry,3);c200,e200=batch.config_for(spec,entry,200);assert e3['sample_range']=='0:2' and e200['sample_range']=='0:199'
 m=importlib.util.spec_from_file_location('own_e4bit_preflight',R/ADAPTER);module=importlib.util.module_from_spec(m);m.loader.exec_module(module);assert module.MID==MID
 assert protected=={path:sha(path) for path in protected}
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,YAML=CFG,YAML_SHA256=sha(R/CFG),protocol=PROTO,protocol_SHA256=sha(R/PROTO),own_source_tensors=1676,active_expected=1631,unused_original_shared_KV=45,own_chat_inputs=200,true3_job=str(J),job_SHA256=sha(J/'job.json'),source_SHA256=spec['source_sha256'],no_new_NPU_claimed=True,no_G_or_canonical_write=True,ready_for_original_runner_true3=True)
 put(H/'readiness.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

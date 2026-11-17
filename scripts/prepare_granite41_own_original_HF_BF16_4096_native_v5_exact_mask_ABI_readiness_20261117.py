"""Own Granite4.1 native config and source closure, no learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,inspect,copy,zipfile,xml.etree.ElementTree as ET,shutil,importlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='ibm-granite/granite-vision-4.1-4b';H=B/'Granite41_own_original_HF_BF16_4096_native_v5_exact_mask_ABI_readiness_20261117';J=B/'Granite41_own_original_HF_BF16_4096_true3_v3_exact_mask_ABI_real_FLOCK_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import yaml,torch,batch_910b as batch
 from batch_910b_support import validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope
 from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v4_exact_mask_ABI_full_errors_20261117 import Granite41OwnOriginalBF16ExactMaskABI as Granite41OwnOriginalBF16
 import verify_granite41_own_original_HF_BF16_4096_native_v3_exact_mask_ABI_20261117
 assert not inspect.isabstract(Granite41OwnOriginalBF16);H.mkdir(exist_ok=False);assert not J.exists()
 front=B/'Granite41_own_original_HF_BF16_4096_frontend200_source_META_v3_exact_own_wrapper_layout_20261117';review=js(front/'review.json');assert review['passed'] and review['samples']==200 and not review['learned_CPU_or_NPU_model_forward'];assert sha(front/'frontend200.jsonl')==review['frontend_jsonl_SHA256']=='046b7ea34981095db344981965d5bc3fa6ea7cb926e04779151897755d692573'
 saved=Path(review['original_saved_YAML']);assert sha(saved)==review['original_saved_YAML_SHA256'];cfg=yaml.safe_load(saved.read_text());assert cfg['model']['kwargs']==dict(device='auto',max_new_tokens=4096,torch_dtype='bfloat16');assert cfg['algorithm']==dict(name='greedy_inference',batch_size=10,kwargs=dict(extract_answer=True))
 W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Granite4VisionForConditionalGeneration')/MID;assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.suffix in ['.json','.py','.model','.jinja','.tiktoken']};headers=front/'source_tensor_headers.json';mapping=front/'own_source_to_native_META_name_map.json';assert len(js(headers))==len(js(mapping)['mapping'])==1083
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']=='f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d';rows,sheet,_=workbook_rows(book);row,=[x for x in rows if x['model_id']==MID];assert row['score'] is None
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:baseline=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 ds=dict(server=baseline,local=str(js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D']),preserved_original=str(js(binding['local_A100_reference_snapshot'])['models'][MID]['A100_D']));assert all(Decimal(v)!=0 for v in ds.values())
 entry=js(D/'state.json')['models'][MID];assert entry['status']=='needs_adaptation' and entry['score'] is None
 source={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   f=Path(name).resolve()
   if f.is_relative_to(R) and f.suffix=='.py' and f.is_file() and (f==R/'run.py' or any(f.is_relative_to(R/x) for x in ['scripts','universal_eval','configs'])):source[str(f.relative_to(R))]=sha(f)
 for name in ['scripts/prepare_granite41_own_original_HF_BF16_4096_native_v2_real_source_files_readiness_20261117.py','scripts/continue_granite41_true3_capacity_fresh3_200_v4_exact_mask_ABI_20261117.py','run.py','universal_eval/orchestrator/pipeline.py']:source[name]=sha(R/name)
 data=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';assert sha(data)==review['dataset_SHA256']
 static={str(f):sha(f) for f in [front/'review.json',front/'actual_completion.json',front/'frontend200.jsonl',headers,mapping,saved]}
 preload=R/'sitecustomize.py';assert sha(preload)=='e68ec327754e0020c670f3f771fc03310420fb9b55846f6d946acf63e65d1f09';static[str(preload)]=sha(preload)
 for n in ['configuration.py','modeling.py','processing.py','downsampling.py']:assert (W/n).read_bytes()==(R/'universal_eval/model/granite41_own_original_source_20261117'/n).read_bytes()
 p=dict(model_id=MID,weight_path=str(W),report_task=cfg['task_type'],saved_original_configuration_used=True,workbook_D_reference_authorized=True,current_A100_D_values=ds,own_weight_files=review['own_weight_files'],own_source_headers=str(headers),own_source_to_native_name_map=str(mapping),own_runtime_assets_SHA256=assets,implementation_source_SHA256=copy.deepcopy(source),static_evidence_SHA256=static,own_frontend200=str(front/'frontend200.jsonl'),dataset_path=str(data),dataset_SHA256=sha(data),original_max_new_tokens=4096,original_configured_outer_batch10_inner_serial1=True,original_default_chat_template_kwargs_empty=True,hard_reserve_bytes=3*1024**3,log_reserve_bytes=1024**3,precheck_reserve_bytes=512*1024**2,per_sample_lossless_space_estimate_bytes=2*1024**3,historical_source_boundary='User-authorized existing A100 workbook D comparison. Own saved original HF BF16/4096, remote source, data, processor and Greedy configuration retained. Historical A100 engine and value tensors were not captured; no online-A100 rerun prerequisite or MLU/CPU score substitution.',final_original_protocol_qualification_pending_until_full200_threeD=True)
 pp=R/'configs/protocols/granite41_own_original_HF_BF16_4096_native_v5_exact_mask_ABI_20261117.json';put(pp,p)
 cf=R/'configs/Granite4VisionForConditionalGeneration/ibm-granite--granite-vision-4.1-4b-own-original-HF-BF16-4096-native-v5-exact-mask-ABI-20261117.yaml';cfg.update(name=cf.stem,task_type=p['report_task'],sample_count=200,sample_range='0:199');cfg['data']=dict(preprocess=True,apply_template=False,filter_empty=False,download=False,dataset_path=str(data));cfg['model'].update(name=MID,adapter='universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v4_exact_mask_ABI_full_errors_20261117.py:Granite41OwnOriginalBF16ExactMaskABI',model_path=str(W),download=False,kwargs=dict(protocol_file=str(pp),protocol_sha256=sha(pp),protocol=dict(model_id=MID),npu_devices=[4]));cfg['metrics']=dict(override=['Accuracy'],include_failed_predictions=True);cfg['baseline']=dict(score=float(baseline),threshold=.05,primary_metric='Accuracy');cfg['output']=dict(reports_dir='workspace/reports',layout='model_task_timestamp',evidence_dir='workspace/diagnostics/model_runs',timezone='Asia/Hong_Kong',verbose=True);put(cf,cfg)
 template=js(B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117/job.json');job=copy.deepcopy(template);job.update(model_id=MID,created_at=datetime.now(timezone.utc).isoformat(),stages=[3]);source.update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp)});job['spec']=dict(approved=False,config=str(cf.relative_to(R)),verifier='scripts/verify_granite41_own_original_HF_BF16_4096_native_v3_exact_mask_ABI_20261117.py',source_sha256=source,npu_devices=[4],smoke_timeout_seconds=14400,formal_timeout_seconds=172800,verify_timeout_seconds=21600);job['protected_canonical_hashes']={n:sha(n) for n in template['protected_canonical_hashes']}
 batch.check_sources(job['spec']);scope(job['spec'],[3],False)
 for count in [3,200]:_,c=batch.config_for(job['spec'],entry,count);validate_config(c,MID)
 put(J/'job.json',job);value=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,ready_for_original_runner_true3=True,job=str(J),job_SHA256=sha(J/'job.json'),YAML=str(cf),YAML_SHA256=sha(cf),protocol_SHA256=sha(pp),source_count=len(source),current_three_D=ds,real_NPU_mapping_full_numeric_and_five_reports_still_pending=True,no_NPU_model_forward=True);put(H/'readiness.json',value);print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()

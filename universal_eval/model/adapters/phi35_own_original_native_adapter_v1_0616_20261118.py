"""Own Phi-3.5 saved vLLM BF16/Greedy4096/outer10 with read-only observers."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,logging,os,tempfile,time,traceback
from pathlib import Path
from universal_eval.model.interface import IModel

MID='microsoft/Phi-3.5-vision-instruct'
EXTENSION='universal_eval.model.phi35_own_native_value_0616_20261118.worker.OwnPhi35FullValueWorkerExtension'
LOG=logging.getLogger(__name__)

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_bytes())

class OwnPhi35Native(IModel):
 def __init__(self,npu_devices,protocol_file,protocol_sha256,protocol):
  assert protocol=={'model_id':MID} and len(npu_devices)==1
  self.device=int(npu_devices[0]);assert sha(protocol_file)==protocol_sha256
  self.p=read(protocol_file);self.protocol_sha256=protocol_sha256
  assert self.p['model_id']==MID
  self.engine=None;self.rows=0;self.groups=0;self.report_index=0;self.group_manifests=[]
  self.lineage=dict(model_id=MID,hardware='Ascend910B',devices=[self.device],max_new_tokens=4096,LM_layers=32,head_size=32064,original_outer_batch=10,actual_samples=0,actual_model_load_completed=False)
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Original own vLLM Ascend Phi-3.5 BF16 with full-value observation'
 def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')

 def load(self,model_path,**kwargs):
  p=self.p;root=Path(model_path).resolve();R=Path('/root/projects/comperror')
  assert str(root)==p['weight_path'] and p['workbook_D_reference_authorized']
  for name,record in p['own_weight_files'].items():
   file=root/name;assert file.stat().st_size==record['bytes'] and sha(file)==record['SHA256']
  for name,value in p['own_assets_SHA256'].items():assert sha(root/name)==value
  for relative,value in p['implementation_source_SHA256'].items():
   assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
   assert sha(R/relative)==value,relative
  for path,value in p['static_evidence_SHA256'].items():assert sha(path)==value,path
  self.front=[json.loads(s) for s in Path(p['own_frontend200']).read_text().splitlines()]
  assert [row['index'] for row in self.front]==list(range(200))
  os.environ.update(ASCEND_RT_VISIBLE_DEVICES=str(self.device),VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
  import torch,torch_npu
  from vllm import LLM,SamplingParams,__version__ as vllm_version
  from vllm.platforms import current_platform
  assert current_platform.device_type=='npu' and torch.npu.device_count()==1
  evidence_root=R/'workspace/diagnostics/model_runs';evidence_root.mkdir(exist_ok=True,parents=True)
  self.evidence=Path(tempfile.mkdtemp(prefix='Phi35_own_original_BF16_native_full_values_',dir=evidence_root)).resolve()
  self.trace=self.evidence/'requests.jsonl';self.trace.touch(exist_ok=False)
  self.failure=self.evidence/'failures.jsonl';self.failure.touch(exist_ok=False)
  # No max_model_len, backend, prefix-cache, encoder-cache or stopping override:
  # these are the own saved LLM defaults observed by the preceding preflight.
  self.engine=LLM(model=str(root),tensor_parallel_size=1,dtype='auto',gpu_memory_utilization=.9,trust_remote_code=True,allowed_local_media_path=p['media_root'],enforce_eager=True,profiler_config={'profiler':'torch','torch_profiler_dir':str(self.evidence/'profiler'),'max_iterations':2,'ignore_frontend':True},worker_extension_cls=EXTENSION)
  self.sampling=SamplingParams(temperature=0.,top_p=1.,max_tokens=4096)
  cfg=self.engine.llm_engine.vllm_config
  assert cfg.device_config.device_type=='npu' and cfg.model_config.dtype==torch.bfloat16 and cfg.model_config.quantization is None
  assert cfg.parallel_config.tensor_parallel_size==1 and cfg.model_config.max_model_len==4096
  assert cfg.parallel_config.worker_extension_cls==EXTENSION and os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION') in [None,'0']
  header=read(p['own_source_headers'])
  result=self.engine.collective_rpc('own_phi35_install_capture',args=(str(root),header,str(self.evidence)))
  assert len(result)==1 and result[0]['installed']
  yaml_path=Path(sys.argv[sys.argv.index('--config')+1]).resolve();assert yaml_path.is_relative_to(R/'configs')
  self.lineage.update(actual_model_load_completed=True,engine='vLLM Ascend',vllm_version=vllm_version,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,device_type='npu',runtime_dtype=str(cfg.model_config.dtype),tensor_parallel_size=1,effective_max_model_len=cfg.model_config.max_model_len,effective_max_batched_tokens=cfg.scheduler_config.max_num_batched_tokens,cache_block_size=cfg.cache_config.block_size,prefix_caching=cfg.cache_config.enable_prefix_caching,quantization=None,temperature=0.,top_p=1.,max_output_tokens=4096,original_engine_stop_ids=[2,32000],protocol_sha256=self.protocol_sha256,evidence_directory=str(self.evidence),request_trace=str(self.trace),failure_trace=str(self.failure),full_numeric_group_index=str(self.evidence/'full_numeric_group_index.json'),actual_batch_manifest=str(self.evidence/'full_numeric_group_index.json'),own_actual_worker_source_inspection=result[0]['own_source_inspection'],actual_run_YAML=str(yaml_path),actual_run_YAML_SHA256=sha(yaml_path),named_RPC_only_no_arbitrary_callable_deserialization=True,observation_only_worker_extension_cls=EXTENSION,all_original_model_math_backend_dtype_sampling_prompt_batching_preserved_with_observation_extension_only=True,historical_A100_engine_or_tensor_capture_claimed=False,historical_source_boundary=p['historical_source_boundary'],profile_scope='First real original vision and prefill, not all generation forwards')
  self.save_index()
  LOG.info('Own Phi3.5 native model loaded on physical NPU%s, evidence=%s',self.device,self.evidence)

 def save_index(self):
  value=dict(model_id=MID,groups=self.group_manifests,actual_completed_samples=self.rows,actual_completed_outer_groups=self.groups,own_actual_worker_source_inspection=self.lineage['own_actual_worker_source_inspection'])
  path=self.evidence/'full_numeric_group_index.json';tmp=path.with_suffix('.tmp')
  tmp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode());tmp.replace(path)

 def message(self,prompt,media_path,index):
  ref=self.front[index];media=Path(media_path).resolve(strict=True)
  assert prompt==ref['original_model_prompt'] and str(media)==ref['media_path'] and sha(media)==ref['image_SHA256']
  return ref['original_native_message']

 def batch_predict(self,batch):
  assert self.engine is not None and 1<=len(batch)<=10 and self.rows+len(batch)<=200
  messages=[self.message(prompt,media,self.rows+i) for i,(prompt,media) in enumerate(batch)]
  started=time.monotonic();profile=self.groups==0;capture_began=False
  try:
   result=self.engine.collective_rpc('own_phi35_begin_capture',args=(self.groups,len(batch)))
   assert len(result)==1 and result[0]['started'];capture_began=True
   if profile:self.engine.start_profile('own_phi35_first_original_group')
   try:outputs=self.engine.chat(messages,self.sampling,use_tqdm=False)
   finally:
    if profile:self.engine.stop_profile()
   assert len(outputs)==len(batch)
   captured=self.engine.collective_rpc('own_phi35_finish_capture');assert len(captured)==1
   self.group_manifests.append(captured[0]);rows=[]
   for i,(message,result) in enumerate(zip(messages,outputs)):
    ref=self.front[self.rows+i]
    assert result.finished and len(result.outputs)==1
    candidate=result.outputs[0];prompt_ids=list(result.prompt_token_ids)
    assert prompt_ids==ref['actual_native_prompt_token_ids'] and prompt_ids.count(32044)==ref['original_HD_image_token_count']
    rows.append(dict(index=self.rows+i,group_index=self.groups,group_size=len(batch),group_wall_seconds=time.monotonic()-started,prompt=message[0]['content'][1]['text'],media_uri=message[0]['content'][0]['image_url']['url'],request_id=result.request_id,prompt_token_ids=prompt_ids,generated_token_ids=list(candidate.token_ids),finish_reason=candidate.finish_reason,stop_reason=candidate.stop_reason,raw_output=candidate.text,error=None))
   with self.trace.open('a',encoding='utf-8') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   self.rows+=len(rows);self.groups+=1;self.save_index()
   LOG.info('Own Phi35 completed group%s samples%s, genuine captured forward count%s',self.groups,self.rows,captured[0]['actual_native_forward_calls'])
   return [row['raw_output'] for row in rows]
  except BaseException:
   value=dict(index_start=self.rows,group_index=self.groups,group_size=len(batch),capture_began=capture_began,error=traceback.format_exc(),all_requested_prompts=[p for p,m in batch],all_media_paths=[m for p,m in batch],no_missing_sample_accepted=True)
   with self.failure.open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
   LOG.error('%s',value['error']);raise

 def predict(self,prompt,media_path=None):return self.batch_predict([(prompt,media_path)])[0]
 def get_sample_execution(self,prompt,media_path):
  rows=[json.loads(s) for s in self.trace.read_text().splitlines()];row=rows[self.report_index]
  assert row['index']==self.report_index;self.report_index+=1
  return dict(original_native_request=row,full_numeric_group_manifest=self.group_manifests[row['group_index']]['path'],full_numeric_group_manifest_SHA256=self.group_manifests[row['group_index']]['SHA256'])
 def get_execution_lineage(self):return dict(self.lineage,observed_samples=self.rows,observed_groups=self.groups)
 def unload(self):self.engine=None

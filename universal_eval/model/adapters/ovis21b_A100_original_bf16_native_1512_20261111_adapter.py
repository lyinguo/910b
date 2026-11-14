"""Ovis2-1B original BF16/eager LM, own AIMv2 SDPA and original greedy interface."""
import hashlib,json,logging,tempfile,time
from pathlib import Path
from universal_eval.model.interface import IModel
MODEL='AIDC-AI/Ovis2-1B';LOG=logging.getLogger(__name__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):
 q=Path(str(p)+'.own_tmp');q.write_text(json.dumps(x,ensure_ascii=False,default=lambda v:sorted(v) if isinstance(v,set) else str(v)),encoding='utf-8');q.replace(p)
class Ovis21BOriginalNativeBF16(IModel):
 def __init__(self,npu_devices,protocol_file,protocol_sha256,protocol):
  assert protocol=={'model_id':MODEL} and len(npu_devices)==1
  assert sha(protocol_file)==protocol_sha256
  self.protocol=read(protocol_file);self.protocol_sha=protocol_sha256;assert self.protocol['model_id']==MODEL
  self.device_index=int(npu_devices[0]);self.rows=0;self.groups=0;self.model=None
 @property
 def model_name(self):return MODEL
 @property
 def model_type(self):return 'vlm'
 @property
 def inference_framework(self):return 'Transformers native torch_npu'
 def download(self,*args,**kwargs):raise RuntimeError('Use only the existing original own Ovis checkpoint')
 def load(self,model_path,**kwargs):
  import os,torch,torch_npu
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  from universal_eval.model.adapters.ovis21b_own_native_factory_v3_original_tokenizer_1512_20261111 import create,original_predict_replay
  from universal_eval.model.adapters.ovis21b_forward_observer_1512_20261111 import ForwardObserver,digest_tensor
  root=Path(model_path).resolve();assert root==Path(self.protocol['weight_path']).resolve()
  for n,h in self.protocol['resource_sha256'].items():assert sha(root/n)==h,n
  assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') in (None,'') and torch.npu.device_count()>self.device_index
  torch.npu.set_device(self.device_index);device=f'npu:{self.device_index}'
  self.evidence=Path(tempfile.mkdtemp(prefix='ovis21b_original_BF16_native_',dir='workspace/diagnostics/model_runs')).resolve()
  self.model,meta=create(root,self.evidence,self.protocol['resource_sha256'],device=device)
  assert all(p.device.type=='npu' and p.dtype==torch.bfloat16 for p in self.model.parameters())
  assert self.model.get_llm().config._attn_implementation=='eager'
  assert self.model.get_llm().config.vocab_size==151936 and self.model.get_llm().config.num_hidden_layers==24
  assert self.model.config.multimodal_max_length==8192
  original=read(self.protocol['original_A100_deployment'])
  self.replay=original_predict_replay(original['core_source']['universal_eval/model/adapters/ovis_adapter.py'],original['original_configuration']['model']['kwargs'],self.model)
  assert self.replay._answer_suffix==" End your response with 'Final answer: '." and self.replay._max_new_tokens==2048
  expected=read(self.protocol['R_independent_CPU200'])['CPU200'];self.expected=expected
  self.index=None;self.last_input=None;native_preprocess=self.model.preprocess_inputs
  def observe_preprocess(*args,**kw):
   values=native_preprocess(*args,**kw);text,ids,pixels=values
   record={'index':self.index,'prompt':text,'input_ids':ids.tolist(),'input_ids_SHA':digest_tensor(ids),
    'attention_mask':digest_tensor(torch.ne(ids,self.model.get_text_tokenizer().pad_token_id)),
    'pixels':digest_tensor(pixels),'pixels_cast_BF16':digest_tensor(pixels.to(torch.bfloat16))}
   cpu=expected[self.index]
   assert all(record[k]==cpu[k] for k in record),self.index
   self.last_input=record
   return values
  self.model.preprocess_inputs=observe_preprocess
  self.observer=ForwardObserver(self.model.get_llm());self.observer.enabled=True;self.observer.capture_generation=True
  for n,m in self.model.get_visual_tokenizer().named_modules():
   if type(m).__name__=='AIMv2Block':
    self.observer.handles.append(m.register_forward_pre_hook(self.observer.layer_before('visual_tokenizer.'+n),with_kwargs=True))
    self.observer.handles.append(m.register_forward_hook(self.observer.layer_after('visual_tokenizer.'+n),with_kwargs=True))
  self.trace=self.evidence/'requests.jsonl';self.trace.touch(exist_ok=False);self.array_files=[];self.array_keys={}
  write(self.evidence/'source_cast_actual_NPU.json',meta)
  self.lineage={'model_id':MODEL,'device_type':'npu','physical_NPU':self.device_index,'dtype':'torch.bfloat16',
   'engine':'own Ovis native torch_npu','LLM_attention':'eager','visual_attention':'own AIMv2 SDPA',
   'original_outer_batch':10,'actual_inner_batch':1,'max_new_tokens':2048,'multimodal_max_length':8192,
   'own_answer_suffix':self.replay._answer_suffix,'effective_generation_config':self.model.generation_config.to_dict(),
   'source_manifest':str(self.evidence/'source_cast_actual_NPU.json'),'source_manifest_SHA256':sha(self.evidence/'source_cast_actual_NPU.json'),
   'request_trace':str(self.trace),'forwards':str(self.evidence/'forwards.json'),'stages':str(self.evidence/'block_stages.json'),
   'array_index':str(self.evidence/'array_index.json'),'evidence_directory':str(self.evidence),'protocol_sha256':self.protocol_sha,
   'profiler_scope':'first actual image generation only, not full200','dense_block_arrays_scope':'first3 only; full-vocabulary logits retained for every actual later forward',
   'source_tensors':468,'native_cache_layers':24,'whole_vocabulary_size':151936,'quantization':None,
   'source_and_API_compatibility':meta['compatibility_changes'],'no_mock_initializer_or_CPU_answers':True}
 def get_execution_lineage(self):return self.lineage
 def predict(self,prompt,media_path=None):return self.batch_predict([(prompt,media_path)])[0]
 def batch_predict(self,batch):
  import contextlib,torch
  from torch_npu.profiler import profile,ProfilerActivity,tensorboard_trace_handler
  assert self.model is not None and 1<=len(batch)<=10
  result=[]
  for offset,(prompt,media_path) in enumerate(batch):
   i=self.rows+offset;assert i<200
   media=Path(media_path).resolve(strict=True);assert media.is_relative_to(Path(self.protocol['media_root']).resolve())
   assert sha(media)==self.expected[i]['media_SHA256']
   self.index=i;self.observer.index=i;self.observer.capture_dense=i<3;start=time.monotonic()
   ctx=profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],record_shapes=True,with_stack=False,
    profile_memory=False,on_trace_ready=tensorboard_trace_handler(str(self.evidence/'profiler'))) if i==0 else contextlib.nullcontext()
   with ctx:raw=self.replay.predict(prompt,str(media))
   torch.npu.synchronize();ids=self.observer.outputs[-1]['generated_token_ids']
   assert self.observer.outputs[-1]['index']==i and 0<len(ids)<=2048 and all(0<=t<151936 for t in ids)
   eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
   stop=ids[-1] in eos;assert stop or len(ids)==2048
   row={'index':i,'group_index':self.groups,'group_size':len(batch),'inner_model_batch':1,
    'prompt':prompt,'formatted_prompt':self.last_input['prompt'],'prompt_token_ids':self.last_input['input_ids'],
    'generated_token_ids':ids,'raw_output':raw,'media_uri':'file://'+str(media),
    'request_id':f'{self.groups}:{i}','finish_reason':'stop' if stop else 'length','stop_reason':ids[-1] if stop else None,
    'actual_forward_count':len([f for f in self.observer.steps if f['index']==i]),'elapsed_seconds':time.monotonic()-start}
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   result.append(raw);LOG.info('Own Ovis sample %s complete; tokens/forwards %s/%s',i,len(ids),row['actual_forward_count'])
  path=self.evidence/f'group_{self.groups:02d}_arrays.npz';archive=self.observer.save(path);self.array_files.append(archive)
  for k in self.observer.arrays:self.array_keys[k]=str(path)
  self.observer.saved_keys.update(self.observer.arrays);self.observer.arrays.clear()
  write(self.evidence/'forwards.json',self.observer.steps);write(self.evidence/'block_stages.json',self.observer.stage)
  write(self.evidence/'array_index.json',{'files':self.array_files,'keys':self.array_keys})
  self.rows+=len(batch);self.groups+=1;return result
 def unload(self):self.model=None

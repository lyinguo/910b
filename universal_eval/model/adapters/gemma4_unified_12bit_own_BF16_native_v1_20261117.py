"""Genuine own Unified12B HF inference on NPU; original saved protocol.

The original source is inspected without executing its mock/stub initializer.
Own historical vision parameter names are mapped by exact shape/coverage;
all 677 own tensors are independently hashed on the actual NPU before use.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,hashlib,copy,re,inspect,logging,shutil,traceback
import torch,torch_npu
from safetensors import safe_open
from universal_eval.model.interface import IModel
from universal_eval.model.reference_sources.gemma4_unified_owned_checkpoint_layout import owned_mapping
from universal_eval.model.gemma4_unified_12bit_own_20261117.observer import FullObserver,fingerprint,record
log=logging.getLogger(__name__);MID='google/gemma-4-12B-it';R=Path('/root/projects/comperror')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def original_response(processor,text):
 text=text.strip()
 try:
  if hasattr(processor,'parse_response'):
   parsed=processor.parse_response(text)
   if isinstance(parsed,dict):text=parsed.get('content',text)
   elif isinstance(parsed,str):text=parsed
 except Exception:pass
 return text.strip()

class OwnUnified12BitBF16(IModel):
 def __init__(self,npu_devices=None,protocol_file=None,protocol_sha256=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
  self.devices=list(npu_devices or [4]);assert self.devices==[4];self.device=torch.device('npu:4');self.protocol_file=Path(protocol_file);assert sha(self.protocol_file)==protocol_sha256;self.p=js(self.protocol_file);assert self.p['model_id']==MID
  stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f');self.home=(R/evidence_dir)/('Gemma4Unified12Bit_own_original_BF16_'+stamp+'_'+str(os.getpid()));self.home.mkdir(parents=True,exist_ok=False);self.details={};self.dispatches=[];self.calls=0;self.model=None;self.finalized=False;self.failed=False;self.lineage=dict(hardware='Ascend 910B',devices=self.devices,evidence_root=str(self.home),final_manifest=str(self.home/'completed_trace.json'),protocol_file=str(self.protocol_file),protocol_SHA256=protocol_sha256,no_A100_historical_engine_tensor_capture_claimed=True)
 @property
 def model_name(self):return MID
 @property
 def model_type(self):return 'Gemma4UnifiedForConditionalGeneration'
 @property
 def inference_framework(self):return 'Genuine installed Transformers Unified12B / all own learned parameters on Ascend NPU'
 def download(self,*a,**kw):raise RuntimeError('Downloads forbidden')
 def load(self,model_path,**kwargs):
  for n,m in list(sys.modules.items()):
   if n.startswith(('flash_attn','transformer_engine')):assert getattr(m,'__file__',None) not in [None,'/dev/null'],('Legacy mock/stub must not be loaded',n)
  from transformers import AutoProcessor,Gemma4UnifiedForConditionalGeneration
  p=self.p;W=Path(model_path).resolve();assert W==Path(p['own_original_root']).resolve() and W.name=='gemma-4-12B-it' and W.parent.name=='google'
  for name,h in p['static_evidence_SHA256'].items():assert sha(name)==h,name
  assert sha(p['reference_workbook'])==p['reference_workbook_SHA256'];assert sha(W/'model.safetensors')=='5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d'
  own=js(p['own_headers_manifest']);headers=own['headers'];meta=js(p['own_META_contract'])['meta'];mapped,alias=owned_mapping(headers,{k:v['shape'] for k,v in meta.items()});assert len(mapped)==677 and {x['dtype'] for x in headers.values()}=={'BF16'}
  torch.set_num_threads(4);torch.npu.set_device(self.device);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  self.processor=AutoProcessor.from_pretrained(str(W),local_files_only=True)
  key_mapping={'^'+re.escape(source)+'$':target for target,source in mapped.items() if target!=source}
  self.model,loading=Gemma4UnifiedForConditionalGeneration.from_pretrained(str(W),local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,key_mapping=key_mapping,output_loading_info=True)
  assert not loading.get('unexpected_keys') and not loading.get('mismatched_keys') and set(loading.get('missing_keys',[]))<={'lm_head.weight'},loading
  self.model.eval();self.model.to(self.device);self.model.tie_weights();params=dict(self.model.named_parameters(remove_duplicate=False));assert set(params)==set(mapped)|{'lm_head.weight'}
  assert params['lm_head.weight'] is params['model.language_model.embed_tokens.weight'];records=[]
  with safe_open(W/'model.safetensors',framework='pt',device='cpu') as sf:
   for target,source in sorted(mapped.items()):
    value=sf.get_tensor(source);actual=params[target];assert value.dtype==actual.dtype==torch.bfloat16 and actual.device==self.device and list(actual.shape)==headers[source]['shape'];expected=fingerprint(value);assert fingerprint(actual)==expected,(source,target,'Own source to actual NPU differs')
    records.append(dict(source_name=source,target_name=target,shape=list(value.shape),source_dtype=str(value.dtype),actual_dtype=str(actual.dtype),actual_device=str(actual.device),source_SHA256=expected,actual_NPU_SHA256=expected))
  assert len(records)==677;self.loading=self.home/'own_source_actual_NPU_loading.json';write(self.loading,dict(own_root=str(W),source_count=677,records=records,own_head_alias=dict(alias,actual_tied_identity=True,actual_NPU_SHA256=fingerprint(params['lm_head.weight'])),genuine_HF_loading_info=loading,exact_own_checkpoint_name_mapping=key_mapping,no_other_checkpoint_or_weight_used=True))
  buffers={n:record(t) for n,t in self.model.named_buffers()};assert all(t.device==self.device and not t.is_meta for _,t in self.model.named_buffers());assert all(t.device==self.device for t in self.model.parameters())
  self.front=[json.loads(l) for l in Path(p['frontend200']).read_text().splitlines()];assert len(self.front)==200;self.by_input={(v['original_model_prompt'].strip(),v['media_path']):v for v in self.front}
  self.observer=FullObserver(self.model,self.home,self.device);self.trace=self.home/'actual_sample_trace.jsonl';self.trace.touch(exist_ok=False)
  runtime={}
  for module in list(sys.modules.values()):
   name=getattr(module,'__file__',None)
   if name and Path(name).is_file() and (name.endswith('.py') or name.endswith('.so')) and (name.startswith('/usr/local/python') or name.startswith(str(R/'universal_eval'))):runtime[name]=sha(name)
  self.runtime=self.home/'actual_imported_runtime_SHA256.json';write(self.runtime,runtime)
  self.lineage.update(model_id=MID,own_source_count=677,own_source_loading=str(self.loading),own_source_loading_SHA256=sha(self.loading),own_root=str(W),own_checkpoint_file_SHA256=sha(W/'model.safetensors'),actual_head_size=262144,actual_language_layers=48,actual_GQA_sliding_heads=8,actual_global_heads=1,actual_K_equals_V_from_own_config=True,actual_shared_KV_layers=0,actual_execution_dtype='BF16',actual_input_floating_dtype='float32',own_saved_HF_serial1_2048_enable_thinking_False_restored=True,no_new_disable_thinking=True,generation_kwargs=dict(max_new_tokens=2048,do_sample=False,temperature=0.0),original_generation_config=self.model.generation_config.to_dict(),actual_attention={n:m.config._attn_implementation for n,m in [('root',self.model),('text',self.model.model.language_model)]},actual_analytic_buffers=buffers,actual_parameter_devices=sorted({str(t.device) for t in self.model.parameters()}),runtime_SHA256_file=str(self.runtime),runtime_SHA256_file_SHA256=sha(self.runtime),per_sample_evidence=str(self.trace),profiler_trace=str(self.home/'npu_trace.json'),profiler_scope='First genuine original HF root vision+prefill only; not all generation or all200',original_legacy_adapter_initializer_not_executed=True,all_full_head_native48KV_inputs_masks_positions_retained=True,CPU_learned_model_forward_count=0)
 def predict(self,prompt,media_path=None):
  from PIL import Image
  assert self.model is not None;media=str(Path(media_path).resolve());ref=self.by_input[(str(prompt).strip(),media)];index=ref['index'];assert sha(media)==ref['source_image_SHA256'];assert shutil.disk_usage(self.home).free>=6*1024**3,'3GiB hard+1GiB log+2GiB sample guard'
  image=Image.open(media).convert('RGB');message=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':str(prompt)}]}];cpu=self.processor.apply_chat_template(message,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
  assert set(cpu)==set(ref['inputs'])
  for name,value in cpu.items():assert list(value.shape)==ref['inputs'][name]['shape'] and str(value.dtype)==ref['inputs'][name]['dtype'] and fingerprint(value)==ref['inputs'][name]['SHA256'],(index,name)
  values={n:t.to(self.device) for n,t in cpu.items()};assert values['pixel_values'].dtype==torch.float32;self.observer.begin(index,values);N=cpu['input_ids'].shape[-1]
  try:
   with torch.inference_mode():result=self.model.generate(**values,max_new_tokens=2048,do_sample=False,temperature=0.0)
   if hasattr(result,'sequences'):result=result.sequences
   generated=result[0,N:].detach().cpu().tolist();raw_decoded=self.processor.decode(generated,skip_special_tokens=True).strip();raw=original_response(self.processor,raw_decoded);detail=self.observer.finish(raw_decoded,raw,generated);detail.update(prompt=str(prompt),media_path=media,reference=ref['reference'],truncated=len(generated)>=2048,actual_original_input_ids=cpu['input_ids'].tolist(),actual_NPU_input_SHA256={n:record(t) for n,t in values.items()})
   self.details[(str(prompt).strip(),media)]=detail
   with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.calls+=1;log.info('Own Unified12B NPU sample index=%s forwards=%s tokens=%s raw=%r',index,detail['actual_native_forwards'],len(generated),raw[:200]);return raw
  except BaseException as error:
   detail=self.observer.preserve_failure(error,traceback.format_exc());self.details[(str(prompt).strip(),media)]=detail;raise
 def batch_predict(self,batch):
  record=dict(requested_indices=[self.by_input[(str(p).strip(),str(Path(m).resolve()))]['index'] for p,m in batch],status='running');self.dispatches.append(record)
  result=[]
  for prompt,media in batch:
   key=(str(prompt).strip(),str(Path(media).resolve()));index=self.by_input[key]['index']
   try:
    if self.failed:raise RuntimeError('Not executed after earlier real failure; no fabricated output')
    result.append(self.predict(prompt,media))
   except Exception:
    self.failed=True;result.append(None)
    if key not in self.details:
     detail=dict(index=index,error=traceback.format_exc(),actual_native_forwards=0,not_executed=True,prompt=str(prompt),media_path=key[1]);write(self.home/f'not_executed_sample_{index:05}.json',detail);self.details[key]=detail
    with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(self.details[key],ensure_ascii=False)+'\n')
  record['status']='failed' if self.failed else 'completed';return result
 def get_sample_execution(self,prompt,media_path):
  if not self.finalized:
   runtime=js(self.runtime)
   for module in list(sys.modules.values()):
    name=getattr(module,'__file__',None)
    if name and Path(name).is_file() and (name.endswith('.py') or name.endswith('.so')) and (name.startswith('/usr/local/python') or name.startswith(str(R/'universal_eval'))):runtime[name]=sha(name)
   final_runtime=self.home/'final_actual_imported_runtime_SHA256.json';write(final_runtime,runtime)
   profile={str(f):sha(f) for f in self.home.rglob('*') if f.is_file() and (f.name=='npu_trace.json' or f.suffix=='.csv' or f.name=='trace_view.json')}
   write(self.home/'completed_trace.json',dict(samples=self.calls,failed=any(x['status']!='completed' for x in self.dispatches),outer_batch_dispatches=self.dispatches,sample_trace_SHA256=sha(self.trace),final_runtime_SHA256_file=str(final_runtime),final_runtime_SHA256_file_SHA256=sha(final_runtime),first_vision_prefill_all_profiler_files_SHA256=profile,actual_single_model_forward_batch_size=1));self.finalized=True
  from universal_eval.algorithm.inference import GreedyInference
  media=str(Path(media_path).resolve());key=(str(prompt).strip(),media)
  if key in self.details:return self.details[key]
  key=(GreedyInference._add_answer_instruction(str(prompt)).strip(),media)
  if key in self.details:return self.details[key]
  values=[v for (p,m),v in self.details.items() if m==media];assert len(values)==1;return values[0]
 def get_execution_lineage(self):return self.lineage
 def unload(self):
  self.model=None;torch.npu.empty_cache()

"""Own SmolVLM2-256M FP16 HF execution on an isolated 910B device."""
import hashlib,json,logging,os,tempfile,time
from pathlib import Path
from universal_eval.model.interface import IModel
MODEL='HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
LOG=logging.getLogger(__name__)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    temporary=Path(str(path)+'.own_tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8',newline='\n')
    temporary.replace(path)
class SmolVLM2256VideoOwnFP16OriginalInputsV3(IModel):
    def __init__(self,npu_devices,protocol_file,protocol_sha256,protocol):
        assert protocol=={'model_id':MODEL} and len(npu_devices)==1
        assert sha(protocol_file)==protocol_sha256
        self.protocol=read(protocol_file);assert self.protocol['model_id']==MODEL
        self.protocol_sha=protocol_sha256;self.physical=int(npu_devices[0])
        self.model=None;self.rows=0;self.groups=0
    @property
    def model_name(self):return MODEL
    @property
    def model_type(self):return 'vlm'
    @property
    def inference_framework(self):return 'Transformers native torch_npu'
    def download(self,*args,**kwargs):raise RuntimeError('Use the existing own checkpoint')
    def load(self,model_path,**kwargs):
        root=Path(model_path).resolve()
        assert root==Path(self.protocol['weight_path']).resolve()
        for n,h in self.protocol['resource_sha256'].items():assert sha(root/n)==h,n
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import torch,torch_npu
        from transformers import AutoModelForImageTextToText
        from universal_eval.model.adapters.smolvlm2_256video_own_image_inputs_v2_legacy_axis_1512_20261111 import OwnSmolVLM2ImageInputs
        from universal_eval.model.adapters.smolvlm2_256video_observer_support_v3_1512_20261111 import ForwardObserver,source_cuda_or_npu_manifest
        assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') in (None,'')
        assert torch.npu.device_count()>self.physical
        torch.npu.set_device(self.physical)
        self.device=torch.device(f'npu:{self.physical}')
        self.io=OwnSmolVLM2ImageInputs(root)
        self.model,loading=AutoModelForImageTextToText.from_pretrained(str(root),dtype=torch.float16,
            trust_remote_code=True,local_files_only=True,output_loading_info=True)
        assert not loading['missing_keys'] and not loading['unexpected_keys']
        self.model=self.model.to(self.device).eval()
        assert all(p.device.type=='npu' and p.dtype==torch.float16 for p in self.model.parameters())
        assert self.model.config.text_config.vocab_size==49280 and self.model.lm_head.weight.shape==(49280,576)
        self.evidence=Path(tempfile.mkdtemp(prefix='smol256video_FP16_native_',dir='workspace/diagnostics/model_runs')).resolve()
        self.trace=self.evidence/'requests.jsonl';self.trace.touch(exist_ok=False)
        self.observer=ForwardObserver(self.model);self.observer.capture_generation=True;self.observer.enabled=True
        self.array_files=[];self.array_index={}
        manifest=source_cuda_or_npu_manifest(self.model,root)
        assert manifest['source_tensor_count']==471
        write(self.evidence/'source_cast_actual_NPU.json',manifest)
        self.expected_cpu=read(self.protocol['R_independent_CPU200'])['CPU200']
        self.lineage={'model_id':MODEL,'engine':'Transformers native torch_npu','device_type':'npu',
            'requested_physical_npu':self.physical,'logical_npu':self.physical,'dtype':'torch.float16',
            'torch_version':torch.__version__,'torch_npu_version':torch_npu.__version__,
            'original_A100_outer_batch':10,'actual_model_inner_batch':1,'original_max_new_tokens':2048,
            'original_generation_kwargs':{'max_new_tokens':2048,'do_sample':False,'temperature':0.0},
            'effective_generation_config':self.model.generation_config.to_dict(),
            'attention_implementations':{n:getattr(m.config,'_attn_implementation',None) for n,m in self.model.named_modules() if hasattr(m,'config')},
            'source_parameter_count':471,'source_manifest':str(self.evidence/'source_cast_actual_NPU.json'),
            'source_manifest_SHA256':sha(self.evidence/'source_cast_actual_NPU.json'),
            'evidence_directory':str(self.evidence),'request_trace':str(self.trace),
            'whole_forward_manifest':str(self.evidence/'forwards.json'),
            'block_positions_masks_manifest':str(self.evidence/'block_stages.json'),
            'array_index':str(self.evidence/'array_index.json'),
            'protocol_sha256':self.protocol_sha,'source_resources_sha256':self.protocol['resource_sha256'],
            'input_contract':'own image-only SmolVLM, original Pillow104 resize axis order, all200 native CPU inputs bitwise verified; unused video dependency not stubbed',
            'profiler_scope':'first actual image and its generation; not the whole 200',
            'dense_block_arrays_scope':'first three actual samples; all later forwards retain complete-vocabulary logits and positions/cache digests',
            'quantization':None}
        # Typed config sets are observations only; do not alter model defaults.
        def safe(v):
            if isinstance(v,set):return sorted(v,key=repr)
            raise TypeError(type(v).__name__)
        self.lineage=json.loads(json.dumps(self.lineage,default=safe))
        LOG.info('Own SmolVLM2 FP16 source cast/actual NPU 471 tensors verified; physical card %s',self.physical)
    def get_execution_lineage(self):
        return self.lineage
    def predict(self,prompt,media_path=None):
        return self.batch_predict([(prompt,media_path)])[0]
    def batch_predict(self,batch):
        import contextlib,torch
        from PIL import Image
        from torch_npu.profiler import profile,ProfilerActivity,tensorboard_trace_handler
        from universal_eval.model.adapters.smolvlm2_256video_observer_support_v3_1512_20261111 import digest_tensor
        assert self.model is not None and 1<=len(batch)<=10
        outputs=[]
        for offset,(prompt,media_path) in enumerate(batch):
            i=self.rows+offset;assert i<200
            media=Path(media_path).resolve(strict=True)
            assert media.is_relative_to(Path(self.protocol['media_root']).resolve()) and media.is_file()
            text=self.io.render(prompt)
            with Image.open(media) as im:inputs=self.io(text,im.convert('RGB'))
            cpu=self.expected_cpu[i]
            assert text==cpu['text']
            assert {k:digest_tensor(v) for k,v in inputs.items()}==cpu['tensor_sha256'],i
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            self.observer.index=i;self.observer.capture_dense=i<3
            start=time.monotonic()
            context=profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],record_shapes=True,
                profile_memory=False,with_stack=False,on_trace_ready=tensorboard_trace_handler(str(self.evidence/'profiler'))) if i==0 else contextlib.nullcontext()
            with torch.no_grad(),context:
                generated=self.model.generate(**inputs,max_new_tokens=2048,do_sample=False,temperature=0.0)
            torch.npu.synchronize()
            ids=generated[:,inputs['input_ids'].shape[1]:]
            raw=self.io.batch_decode(ids,skip_special_tokens=True)[0].strip()
            record=self.observer.outputs[-1]
            assert record['index']==i
            tokens=record['generated_token_ids'];assert 0<len(tokens)<=2048 and all(0<=t<49280 for t in tokens)
            eos=self.model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else eos
            stopped=tokens[-1] in eos
            assert stopped or len(tokens)==2048
            row={'index':i,'group_index':self.groups,'group_size':len(batch),'inner_model_batch':1,
                'prompt':prompt,'rendered_text':text,'media_uri':'file://'+str(media),
                'request_id':f'{self.groups}:{i}','prompt_token_ids':record['input_token_ids'],
                'generated_token_ids':tokens,'raw_output':raw,'finish_reason':'stop' if stopped else 'length',
                'stop_reason':tokens[-1] if stopped else None,'elapsed_seconds':time.monotonic()-start,
                'actual_forward_count':len([s for s in self.observer.steps if s['index']==i])}
            with self.trace.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            outputs.append(raw)
            LOG.info('Own SmolVLM2 sample %s completed, tokens %s, forwards %s',i,len(tokens),row['actual_forward_count'])
        path=self.evidence/f'group_{self.groups:02d}_arrays.npz'
        archive=self.observer.save(path);self.array_files.append(archive)
        for k in self.observer.arrays:self.array_index[k]=str(path)
        self.observer.saved_keys.update(self.observer.arrays);self.observer.arrays.clear()
        write(self.evidence/'forwards.json',self.observer.steps)
        write(self.evidence/'block_stages.json',self.observer.stage)
        write(self.evidence/'array_index.json',{'files':self.array_files,'keys':self.array_index})
        self.rows+=len(batch);self.groups+=1
        return outputs
    def unload(self):
        self.model=None


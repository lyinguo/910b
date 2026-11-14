"""Native 910B TrOCR with original OCR data, generation configuration and F32 tensors."""
import contextlib,datetime,hashlib,json,logging,os,pathlib,struct,time,traceback
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch,torch_npu
import numpy as np
from PIL import Image
from universal_eval.model.interface import IModel
log=logging.getLogger('universal_eval.model.adapters.trocr_original_910b')

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fingerprint(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def record(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':fingerprint(t)}

def target_key(k):
    if k.startswith('encoder.encoder.layer.'):
        k='encoder.layers.'+k[len('encoder.encoder.layer.'):]
        for a,b in [('.attention.attention.query.','.attention.q_proj.'),('.attention.attention.key.','.attention.k_proj.'),('.attention.attention.value.','.attention.v_proj.'),('.attention.output.dense.','.attention.o_proj.'),('.intermediate.dense.','.mlp.fc1.'),('.output.dense.','.mlp.fc2.')]:k=k.replace(a,b)
    return k

class TrOCROriginal910BAdapter(IModel):
    def __init__(self,npu_devices=None,protocol=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        self.devices=list(npu_devices or []);assert len(self.devices)==1
        self.device=torch.device('npu:'+str(self.devices[0]));self.protocol=dict(protocol or {})
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('TrOCR_original_910b_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False)
        (self.evidence/'visual_features').mkdir();self.trace=self.evidence/'sample_execution.jsonl';self.details={};self.calls=0;self.profile_done=False;self.lineage={}
    @property
    def model_name(self):return getattr(self,'_name','unloaded')
    @property
    def model_type(self):return 'VisionEncoderDecoderModel'
    @property
    def inference_framework(self):return 'Native Transformers / PyTorch + torch_npu F32'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
    def load(self,model_path,**kwargs):
        from transformers import AutoConfig,AutoImageProcessor,RobertaTokenizer,TrOCRProcessor,VisionEncoderDecoderModel,GenerationConfig
        from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
        from safetensors import safe_open
        directory=pathlib.Path(model_path).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
        p=self.protocol;self._name=directory.parent.name+'/'+directory.name;assert p['model_id']==self.model_name
        assert torch.npu.is_available();torch.npu.set_device(self.device)
        for n in ('conv','matmul'):getattr(torch.npu,n).allow_hf32=False
        for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(directory/name)==digest,name
        for path,digest in p['runtime_sha256'].items():assert sha(path)==digest,path
        for path,digest in p['original_source_sha256'].items():assert sha(path)==digest,path
        reference=pathlib.Path(p['reference_report']);assert reference.is_relative_to('/mnt/afs/comp_error/workspace/reports') and sha(reference)==p['reference_report_sha256']
        historical=json.loads(reference.read_text());pc=historical['pipeline_config'];assert pc['model']['name']==self.model_name and pc['metrics']['override']==['CER'] and len(historical['detailed_results'])==200
        assert sha(p['reference_log'])==p['reference_log_sha256']
        import re
        flags=re.findall(r'开始推理循环: 共 200 条样本[^\n]*提取答案=(True|False)',pathlib.Path(p['reference_log']).read_text())
        assert flags and len(set(flags))==1 and (flags[0]=='True')==p['extract_answer']==pc['algorithm']['kwargs']['extract_answer']
        assert pc['algorithm']['kwargs'].get('ocr_extract',False)==p['ocr_extract']
        self.max_new_tokens=int(p['max_new_tokens']);assert pc['model']['kwargs']['max_new_tokens']==self.max_new_tokens
        assert not pc['model']['kwargs'].get('ocr_allowed_charset') and not pc['model']['kwargs'].get('ocr_plate_grammar')
        self.dataset=pathlib.Path(p['dataset_path']).resolve();assert self.dataset.is_relative_to('/root/projects/comperror/workspace/datasets') and sha(self.dataset)==p['dataset_sha256']
        cfg=AutoConfig.from_pretrained(str(directory),local_files_only=True);assert cfg.encoder.model_type in ('vit','deit') and cfg.decoder.model_type=='trocr'
        cfg.encoder._attn_implementation='eager';cfg.decoder._attn_implementation='eager'
        with torch.device('meta'):self.model=VisionEncoderDecoderModel(cfg)
        weightfile=directory/next(iter(p['weights']));wreview=p['weights'][weightfile.name];assert weightfile.stat().st_size==wreview['size']
        if weightfile.suffix=='.safetensors':
            with weightfile.open('rb') as f:head=f.read(8);header=f.read(struct.unpack('<Q',head)[0])
            assert hashlib.sha256(head+header).hexdigest()==wreview['header_sha256']
            handle=safe_open(str(weightfile),framework='pt',device='cpu');keys=list(handle.keys());source_get=handle.get_tensor;source_context=handle
        else:
            assert sha(weightfile)==wreview['sha256']
            try:state=torch.load(weightfile,map_location='cpu',weights_only=True,mmap=True)
            except RuntimeError as error:
                if 'mmap' not in str(error):raise
                state=torch.load(weightfile,map_location='cpu',weights_only=True)
            assert isinstance(state,dict) and all(torch.is_tensor(t) for t in state.values())
            keys=list(state);source_get=state.__getitem__;source_context=contextlib.nullcontext()
        disabled_pooler=False
        if self.model.encoder.pooler is not None and not any(k.startswith('encoder.pooler.') for k in keys):self.model.encoder.pooler=None;disabled_pooler=True
        expected=dict(self.model.named_parameters(remove_duplicate=False));filled=set();loading=[];legacy=[]
        with source_context:
            for key in keys:
                tensor=source_get(key)
                if key=='decoder.model.decoder.embed_positions._float_tensor':
                    assert not cfg.decoder.use_learned_position_embeddings and tensor.numel()==1
                    legacy.append({'source_key':key,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':fingerprint(tensor),'reason':'Old fairseq device/dtype sentinel; native analytic weights are rebuilt from the original formula. Not a learned tensor.'});continue
                target=target_key(key);assert target in expected and target not in filled,key
                assert tensor.dtype==torch.float32 and tensor.shape==expected[target].shape,key
                value=tensor.to(self.device).contiguous();source_sha=fingerprint(tensor);npu_sha=fingerprint(value);assert source_sha==npu_sha
                parent,leaf=target.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(value,requires_grad=False));filled.add(target)
                loading.append({'source_file':weightfile.name,'source_key':key,'target_key':target,'shape':list(tensor.shape),'source_dtype':str(tensor.dtype),'execution_dtype':str(value.dtype),'device':str(value.device),'source_tensor_sha256':source_sha,'NPU_tensor_sha256':npu_sha})
        decoder=self.model.decoder.model.decoder;embed=decoder.embed_tokens.weight;head=self.model.decoder.output_projection.weight
        tied=bool(cfg.decoder.tie_word_embeddings);headkey='decoder.output_projection.weight'
        if tied:
            if headkey in filled:assert fingerprint(head)==fingerprint(embed)
            else:assert set(expected)-filled=={headkey}
            self.model.decoder.output_projection.weight=embed;filled.add(headkey)
        assert filled==set(expected)
        if not tied:assert self.model.decoder.output_projection.weight is not embed
        analytic={}
        for name,module in self.model.named_modules():
            if isinstance(module,TrOCRSinusoidalPositionalEmbedding):
                # Ordinary tensor, not a registered buffer: materialize CPU analytic F32 then move only the result.
                cpu=module.get_embedding(cfg.decoder.max_position_embeddings+module.padding_idx+1,module.embedding_dim,module.padding_idx)
                module.weights=cpu.to(self.device);analytic[name]=record(cpu)
                def materialize(m,args,kw):
                    input_ids=args[0] if args else kw['input_ids'];past=kw.get('past_key_values_length',args[1] if len(args)>1 else 0)
                    needed=m.padding_idx+1+input_ids.shape[-1]+past
                    if m.weights is None or m.weights.shape[0]<needed:m.weights=m.get_embedding(needed,m.embedding_dim,m.padding_idx).to(self.device)
                module.register_forward_pre_hook(materialize,with_kwargs=True)
        assert all(t.device==self.device and t.dtype==torch.float32 for t in self.model.parameters())
        assert all(not t.is_meta and t.device==self.device for t in self.model.buffers())
        self.model.eval();self.processor=TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,backend='pil'),tokenizer=RobertaTokenizer.from_pretrained(str(directory),local_files_only=True))
        assert self.processor.tokenizer.get_vocab()==json.loads((directory/'vocab.json').read_text())
        gen=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if (directory/'generation_config.json').is_file() else GenerationConfig.from_model_config(cfg)
        self.model.generation_config=gen;self.gen,_=self.model._prepare_generation_config(None,max_new_tokens=self.max_new_tokens,do_sample=False,temperature=0.0)
        assert self.gen.to_dict()==p['effective_generation_config'] and self.gen.do_sample is False and self.gen.use_cache is False
        eos=self.gen.eos_token_id;self.eos=[eos] if isinstance(eos,int) else list(eos)
        self._features=None;self._encoder=None;self._cross=None;self._bridges=0;self._position_first=None;self._position_last=None
        def encoder_hook(module,args,value):
            self._encoder=value.last_hidden_state;self._features=self._encoder
            assert self._encoder.device==self.device and self._encoder.dtype==torch.float32 and bool(torch.isfinite(self._encoder).all())
        def projector_hook(module,args,value):self._features=value
        def cross_hook(module,args,kw):
            value=kw.get('key_value_states',args[1] if len(args)>1 else None)
            assert value is not None and value.device==self.device and value.dtype==torch.float32 and self._features is not None
            if self._cross is None:
                assert value.shape[0]%self._features.shape[0]==0
                expected_features=self._features.repeat_interleave(value.shape[0]//self._features.shape[0],dim=0)
                assert torch.equal(value,expected_features) and bool(value.abs().sum()>0)
                self._cross=record(value);self._bridges+=1
        def position_hook(module,args,kw,value):
            ids=args[0] if args else kw['input_ids'];past=kw.get('past_key_values_length',args[1] if len(args)>1 else 0)
            row={'input_ids':ids.cpu().tolist(),'past_key_values_length':past,'output':record(value)}
            if self._position_first is None:self._position_first=row
            self._position_last=row
        self.model.encoder.register_forward_hook(encoder_hook)
        if hasattr(self.model,'enc_to_dec_proj'):self.model.enc_to_dec_proj.register_forward_hook(projector_hook)
        decoder.layers[0].encoder_attn.register_forward_pre_hook(cross_hook,with_kwargs=True)
        decoder.embed_positions.register_forward_hook(position_hook,with_kwargs=True)
        loading_path=self.evidence/'weight_loading.json';loading_path.write_text(json.dumps(loading,indent=2))
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for path,digest in p['original_source_sha256'].items():(snapshot/(pathlib.Path(path).parent.name+'_'+pathlib.Path(path).name)).write_bytes(pathlib.Path(path).read_bytes())
        (snapshot/'historical_reference_report.json').write_bytes(reference.read_bytes());(snapshot/'historical_run.log').write_bytes(pathlib.Path(p['reference_log']).read_bytes())
        self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'source_dtype':'F32','precision':'All original learned F32 checkpoint tensors remain native NPU F32, HF32 disabled. CPU computes only image preprocessing and original analytic sinusoidal constants. Historical MLU auto dtype may differ; no bitwise cross-hardware or low-bit acceleration claim. No learned-layer CPU offload or downloads.','HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ('conv','matmul')},'all_expected_parameters_loaded':True,'weight_tensor_count':len(loading),'weight_loading_evidence':str(loading_path),'model_resource_root':str(directory),'model_resource_sha256':p['resource_sha256'],'runtime_source_sha256':p['runtime_sha256'],'weight_files':p['weights'],'per_sample_evidence':str(self.trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'architecture_source_snapshot':str(snapshot),'protocol_review':p,'effective_generation_config':self.gen.to_dict(),'effective_image_processor':self.processor.image_processor.to_dict(),'eos_token_ids':self.eos,'analytic_positions':analytic,'decoder_embedding_scale':decoder.embed_tokens.embed_scale,'declared_decoder_tie_word_embeddings':tied,'actual_head_embedding_shared':self.model.decoder.output_projection.weight is embed,'head_sha256':fingerprint(self.model.decoder.output_projection.weight),'embedding_sha256':fingerprint(embed),'disabled_absent_unused_encoder_pooler':disabled_pooler,'legacy_unlearned_sentinels':legacy,'key_conversion':'Only original ViT/DeiT encoder.layer attention/intermediate/output module names mapped to current native layers q/k/v/o_proj and mlp.fc1/fc2. Complete source learned tensors preserved, including declared tied or independent decoder head.','protocol':'Table-matched historical full200 OCR/CER pipeline and real runtime log flags; own image-only TrOCR processor, own vocab/merges, unchanged native generation configuration including original beam and cache settings. Text prompt is retained but never tokenized for this original image-only model. No character constraint, plate grammar or label/reference enters the model.','module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
        log.info('Native TrOCR loaded %s exact F32 tensors; original max_new_tokens=%s beams=%s cache=%s',len(loading),self.max_new_tokens,self.gen.num_beams,self.gen.use_cache)
    def predict(self,prompt,media_path=None):
        self.calls+=1;started=time.monotonic();entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
        try:
            media=pathlib.Path(media_path).resolve();assert media.is_relative_to(self.dataset.parent)
            with Image.open(media) as image:prepared=self.processor(images=image.convert('RGB'),return_tensors='pt')
            assert set(prepared)=={'pixel_values'};pixels=prepared['pixel_values'];assert pixels.dtype==torch.float32
            entry.update(source_image_sha256=sha(media),CPU_pixel_values=record(pixels),text_prompt_used_by_model=False)
            inputs={'pixel_values':pixels.to(self.device)}
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profiler:
                            self._cross=None;self._features=None
                            output=self.model(**inputs,decoder_input_ids=torch.tensor([[self.gen.decoder_start_token_id]],device=self.device),use_cache=False,return_dict=True)
                            assert bool(torch.isfinite(output.logits).all());torch.npu.synchronize(self.device)
                        profiler.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                    self.profile_done=True
                self._cross=None;self._features=None;self._encoder=None;self._bridges=0;self._position_first=None;self._position_last=None
                generated=self.model.generate(**inputs,generation_config=self.gen,return_dict_in_generate=True,output_scores=False)
            tokens=generated.sequences[0].cpu().tolist();assert tokens[0]==self.gen.decoder_start_token_id and self._bridges==1
            raw=self.processor.batch_decode(generated.sequences,skip_special_tokens=True)[0].strip()
            new_count=len(tokens)-1;truncated=new_count>=self.max_new_tokens and tokens[-1] not in self.eos
            features=self.evidence/'visual_features'/('%05d.npz'%(self.calls-1));arrays={'encoder_output':self._encoder.detach().cpu().numpy(),'decoder_cross_attention_features':self._features.detach().cpu().numpy()};np.savez_compressed(features,**arrays)
            entry.update(raw_output=raw,generation_sequence_ids=tokens,generation_count=new_count,truncated=truncated,stop_reason='length' if truncated else 'eos' if tokens[-1] in self.eos else 'other',visual_cross_attention_verified=True,visual_feature_archive=str(features),visual_feature_archive_sha256=sha(features),encoder_output=record(self._encoder),decoder_cross_attention_features=record(self._features),actual_cross_attention_input=self._cross,actual_decoder_beam_batch=self._cross['shape'][0],position_first=self._position_first,position_last=self._position_last)
            log.info('TrOCR NPU sample %s generated=%s beams=%s truncated=%s raw=%r',self.calls,new_count,self.gen.num_beams,truncated,raw[:200]);return raw
        except Exception:
            entry['error']=traceback.format_exc();log.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started
            clean=prompt[:-len('\n\nProvide only the final answer.')] if (prompt or '').endswith('\n\nProvide only the final answer.') else prompt
            self.details[(clean.strip(),media_path)]=entry
            with self.trace.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None;self._encoder=None
    def batch_predict(self,batch):return [self.predict(p,m) for p,m in batch]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing genuine execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):self.model=None;torch.npu.empty_cache()

"""Exact existing OCR encoder-decoder checkpoints with native Bert/Roberta/Camembert/Electra on 910B."""
import contextlib,hashlib,json,logging,os,pathlib,re,struct,time,traceback,collections
import torch,torch_npu,numpy as np
from PIL import Image
from universal_eval.model.adapters.trocr_original_910b_adapter import TrOCROriginal910BAdapter,sha,fingerprint,record,target_key
log=logging.getLogger('universal_eval.model.adapters.ocr_decoder_variants_910b')

class OCRDecoderVariants910BAdapterV2(TrOCROriginal910BAdapter):
    @property
    def inference_framework(self):return 'Native Transformers / PyTorch + torch_npu '+self.protocol['source_dtype']
    def load(self,model_path,**kwargs):
        from transformers import AutoConfig,AutoImageProcessor,AutoTokenizer,TrOCRProcessor,VisionEncoderDecoderModel,GenerationConfig
        from safetensors import safe_open
        directory=pathlib.Path(model_path).resolve();assert directory.is_relative_to('/mnt/afs/huggingfaceautoclonemodels/downmodels')
        p=self.protocol;self._name=directory.parent.name+'/'+directory.name;assert p['model_id']==self.model_name
        assert p['source_dtype'] in ('torch.float32','torch.float16');self.dtype=getattr(torch,p['source_dtype'].split('.')[-1])
        assert torch.npu.is_available();torch.npu.set_device(self.device)
        for n in ('conv','matmul'):getattr(torch.npu,n).allow_hf32=False
        for name,digest in p['resource_sha256'].items():assert pathlib.Path(name).name==name and sha(directory/name)==digest,name
        for path,digest in dict(p['runtime_sha256'],**p['original_source_sha256']).items():assert sha(path)==digest,path
        reference=pathlib.Path(p['reference_report']);assert reference.is_relative_to('/mnt/afs/comp_error/workspace/reports') and sha(reference)==p['reference_report_sha256']
        historical=json.loads(reference.read_text());pc=historical['pipeline_config'];assert pc['model']['name']==self.model_name and pc['metrics']['override']==['CER'] and len(historical['detailed_results'])==200
        assert pc['algorithm']['kwargs']=={'extract_answer':True,'ocr_extract':True} and sha(p['reference_log'])==p['reference_log_sha256']
        flags=re.findall(r'开始推理循环: 共 200 条样本[^\n]*提取答案=(True|False)',pathlib.Path(p['reference_log']).read_text());assert flags and set(flags)=={'True'}
        self.max_new_tokens=p['max_new_tokens'];assert self.max_new_tokens==pc['model']['kwargs']['max_new_tokens']==4096
        assert not pc['model']['kwargs'].get('ocr_allowed_charset') and not pc['model']['kwargs'].get('ocr_plate_grammar')
        self.dataset=pathlib.Path(p['dataset_path']).resolve();assert self.dataset.is_relative_to('/root/projects/comperror/workspace/datasets') and sha(self.dataset)==p['dataset_sha256']
        raw=json.loads((directory/'config.json').read_text());cfg=AutoConfig.from_pretrained(str(directory),local_files_only=True)
        assert cfg.encoder.model_type in ('vit','deit') and cfg.decoder.model_type==p['decoder_family'] in ('bert','roberta','camembert','electra')
        cfg.encoder._attn_implementation='eager';cfg.decoder._attn_implementation='eager'
        with torch.device('meta'):self.model=VisionEncoderDecoderModel(cfg)
        weightfile=directory/next(iter(p['weights']));review=p['weights'][weightfile.name];assert weightfile.stat().st_size==review['size']
        if weightfile.suffix=='.safetensors':
            with weightfile.open('rb') as f:length=f.read(8);header=f.read(struct.unpack('<Q',length)[0])
            assert hashlib.sha256(length+header).hexdigest()==review['header_sha256']
            context=safe_open(str(weightfile),framework='pt',device='cpu');keys=list(context.keys());source_get=context.get_tensor
        else:
            assert sha(weightfile)==review['sha256']
            try:state=torch.load(weightfile,map_location='cpu',weights_only=True,mmap=True)
            except RuntimeError as error:
                if 'mmap' not in str(error):raise
                state=torch.load(weightfile,map_location='cpu',weights_only=True)
            assert isinstance(state,dict) and all(torch.is_tensor(t) for t in state.values())
            keys=list(state);source_get=state.__getitem__;context=contextlib.nullcontext()
        disabled=False
        if self.model.encoder.pooler is not None and not any(k.startswith('encoder.pooler.') for k in keys):self.model.encoder.pooler=None;disabled=True
        expected=dict(self.model.named_parameters(remove_duplicate=False));buffers=dict(self.model.named_buffers());groups=collections.defaultdict(list)
        for name,tensor in expected.items():groups[id(tensor)].append(name)
        aliases=sorted([sorted(v) for v in groups.values() if len(v)>1]);assert aliases==sorted(p['alias_groups'])
        loaded=set();loading=[];source_buffers={}
        with context:
            for key in keys:
                tensor=source_get(key);target=target_key(key)
                if target in buffers:
                    assert target not in source_buffers;source_buffers[target]=tensor.clone();continue
                assert target in expected and target not in loaded and tensor.shape==expected[target].shape and tensor.dtype==self.dtype,key
                value=tensor.to(self.device).contiguous();digest=fingerprint(tensor);assert fingerprint(value)==digest
                parent,leaf=target.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(value,requires_grad=False));loaded.add(target)
                loading.append({'source_file':weightfile.name,'source_key':key,'target_key':target,'shape':list(tensor.shape),'source_dtype':str(tensor.dtype),'execution_dtype':str(value.dtype),'device':str(value.device),'source_tensor_sha256':digest,'NPU_tensor_sha256':digest})
        alias_evidence=[]
        for group in aliases:
            present=[k for k in group if k in loaded];assert present
            parent,leaf=present[0].rsplit('.',1);representative=getattr(self.model.get_submodule(parent),leaf)
            for name in present:
                parent,leaf=name.rsplit('.',1);assert fingerprint(getattr(self.model.get_submodule(parent),leaf))==fingerprint(representative)
            for name in group:
                parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,representative);loaded.add(name)
            alias_evidence.append({'names':group,'all_shared':True,'sha256':fingerprint(representative),'present_source_names':present})
        assert loaded==set(expected)
        materialized=[]
        for name,buffer in buffers.items():
            assert buffer.dtype==torch.int64 and name.startswith('decoder.')
            if name.endswith('.position_ids'):cpu=torch.arange(buffer.shape[-1],dtype=torch.int64).reshape(1,-1).expand(buffer.shape)
            elif name.endswith('.token_type_ids'):cpu=torch.zeros(buffer.shape,dtype=torch.int64)
            else:raise AssertionError(('Unreviewed native buffer',name))
            if name in source_buffers:assert torch.equal(cpu,source_buffers[name]) and source_buffers[name].dtype==torch.int64
            value=cpu.to(self.device);parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,value)
            materialized.append({'name':name,'source_checkpoint_buffer_present':name in source_buffers,'canonical_CPU':record(cpu),'actual_NPU':record(value),'device':str(value.device)})
        assert set(source_buffers)<=set(buffers) and all(t.device==self.device and t.dtype==self.dtype for t in self.model.parameters()) and all(b.device==self.device and b.dtype==torch.int64 for b in self.model.buffers())
        self.model.eval();self.processor=TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(str(directory),local_files_only=True,backend='pil'),tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True))
        from tokenizers import Tokenizer
        own_backend=Tokenizer.from_file(str(directory/'tokenizer.json'));own_vocab=own_backend.get_vocab()
        assert self.processor.tokenizer.get_vocab()==own_vocab and len(own_vocab)==p['vocabulary_contract']['tokenizer_size']
        assert cfg.decoder.vocab_size==p['vocabulary_contract']['model_vocab_size'] and all(0<=i<cfg.decoder.vocab_size for i in own_vocab.values())
        core_digest=hashlib.sha256(json.dumps(sorted([(i,t) for t,i in own_vocab.items()]),ensure_ascii=False).encode()).hexdigest()
        assert core_digest==p['vocabulary_contract']['core_vocabulary_sha256']
        unmapped_ids=sorted(set(range(cfg.decoder.vocab_size))-set(own_vocab.values()))

        genfile=directory/'generation_config.json'
        # Passing the raw legacy dict is essential: new AutoConfig omits old beam/penalty/cache fields.
        gen=GenerationConfig.from_pretrained(str(directory),local_files_only=True) if genfile.is_file() else GenerationConfig.from_model_config(raw)
        self.model.generation_config=gen;self.gen,_=self.model._prepare_generation_config(None,max_new_tokens=self.max_new_tokens,do_sample=False,temperature=0.)
        assert self.gen.to_dict()==p['effective_generation_config'] and self.gen.do_sample is False and self.gen.use_cache is True
        eos=self.gen.eos_token_id;self.eos=[eos] if isinstance(eos,int) else list(eos)
        self._encoder=None;self._features=None;self._cross=None;self._bridges=0;self._decoder_call=None;self._decoder_calls=0;self._position_first=None;self._position_last=None
        def encoder_hook(module,args,value):
            self._encoder=value.last_hidden_state;self._features=self._encoder
            assert self._encoder.device==self.device and self._encoder.dtype==self.dtype and bool(torch.isfinite(self._encoder).all())
        def projector_hook(module,args,value):self._features=value
        def decoder_hook(module,args,kw):
            ids=kw.get('input_ids',args[0] if args else None);cache=kw.get('past_key_values');past=cache.get_seq_length() if cache is not None else 0
            assert ids is not None;self._decoder_call={'input_ids':ids.cpu().tolist(),'past_key_values_length':past};self._decoder_calls+=1
        def cross_hook(module,args,kw):
            if self._cross is not None:return
            value=kw.get('encoder_hidden_states',args[1] if len(args)>1 else None);assert value is not None and self._features is not None and value.device==self.device and value.dtype==self.dtype
            assert value.shape[0]%self._features.shape[0]==0;wanted=self._features.repeat_interleave(value.shape[0]//self._features.shape[0],dim=0)
            assert torch.equal(value,wanted) and bool(value.abs().sum()>0);self._cross=record(value);self._bridges+=1
        def position_hook(module,args,kw,value):
            ids=args[0] if args else kw['input'];assert self._decoder_call is not None
            row=dict(self._decoder_call,position_ids=ids.cpu().tolist(),output=record(value))
            if self._position_first is None:self._position_first=row
            self._position_last=row
        self.model.encoder.register_forward_hook(encoder_hook)
        if hasattr(self.model,'enc_to_dec_proj'):self.model.enc_to_dec_proj.register_forward_hook(projector_hook)
        self.model.decoder.register_forward_pre_hook(decoder_hook,with_kwargs=True)
        self.model.get_submodule(p['cross_attention_module']).register_forward_pre_hook(cross_hook,with_kwargs=True)
        self.model.get_submodule(p['position_embedding_module']).register_forward_hook(position_hook,with_kwargs=True)
        loading_path=self.evidence/'weight_loading.json';loading_path.write_text(json.dumps(loading,indent=2))
        (self.evidence/'buffer_materialization.json').write_text(json.dumps(materialized,indent=2));snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for path,digest in p['original_source_sha256'].items():
            destination=snapshot/pathlib.Path(path).relative_to('/');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(pathlib.Path(path).read_bytes())
        (snapshot/'historical_reference_report.json').write_bytes(reference.read_bytes());(snapshot/'historical_run.log').write_bytes(pathlib.Path(p['reference_log']).read_bytes())
        self.lineage={'hardware':'Ascend 910B','devices':self.devices,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'source_dtype':'F16' if self.dtype==torch.float16 else 'F32','precision':'Every original learned tensor remains native NPU '+str(self.dtype)+'. HF32 conv/matmul off, no learned-layer CPU offload or low-bit acceleration. Historical MLU auto dtype can differ; no bitwise cross-hardware claim. CPU preprocessing and canonical integer arange/zero buffers only.','HF32_enabled':{n:bool(getattr(torch.npu,n).allow_hf32) for n in ('conv','matmul')},'all_expected_parameters_loaded':True,'weight_tensor_count':len(loading),'weight_loading_evidence':str(loading_path),'model_resource_root':str(directory),'model_resource_sha256':p['resource_sha256'],'runtime_source_sha256':p['runtime_sha256'],'weight_files':p['weights'],'per_sample_evidence':str(self.trace),'profiler_trace':str(self.evidence/'npu_trace.json'),'architecture_source_snapshot':str(snapshot),'protocol_review':p,'effective_generation_config':self.gen.to_dict(),'generation_config_source':'generation_config.json' if genfile.is_file() else 'Raw own config.json legacy generation fields, via official GenerationConfig.from_model_config(dict)','effective_image_processor':self.processor.image_processor.to_dict(),'eos_token_ids':self.eos,'decoder_family':cfg.decoder.model_type,'vocabulary_contract':dict(p['vocabulary_contract'],original_unmapped_model_ids=unmapped_ids),'parameter_alias_groups':alias_evidence,'buffer_materialization':str(self.evidence/'buffer_materialization.json'),'disabled_absent_unused_encoder_pooler':disabled,'position_embedding_module':p['position_embedding_module'],'cross_attention_module':p['cross_attention_module'],'protocol':'Exact table-matched historical200 OCR/CER plus real extraction runtime log. Own image-only processor and tokenizer, original4096 budget, actual own beams/penalties/ngram/cache and special IDs. Legacy raw generation dictionary is used only when no generation_config file exists. Reference text is never passed to the model, no character constraints or grammar.','module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))}}
        log.info('Native OCR %s loaded %s exact %s tensors; original beams=%s cache=%s budget=%s',cfg.decoder.model_type,len(loading),self.dtype,self.gen.num_beams,self.gen.use_cache,self.max_new_tokens)
    def predict(self,prompt,media_path=None):
        self.calls+=1;started=time.monotonic();entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
        try:
            media=pathlib.Path(media_path).resolve();assert media.is_relative_to(self.dataset.parent)
            with Image.open(media) as image:prepared=self.processor(images=image.convert('RGB'),return_tensors='pt')
            assert set(prepared)=={'pixel_values'};pixels=prepared['pixel_values'];assert pixels.dtype==torch.float32
            entry.update(source_image_sha256=sha(media),CPU_pixel_values=record(pixels),text_prompt_used_by_model=False)
            inputs={'pixel_values':pixels.to(self.device,dtype=self.dtype)}
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profiler:
                            self._cross=None;self._features=None
                            output=self.model(**inputs,decoder_input_ids=torch.tensor([[self.gen.decoder_start_token_id]],device=self.device),use_cache=False,return_dict=True);assert bool(torch.isfinite(output.logits).all());torch.npu.synchronize(self.device)
                        profiler.export_chrome_trace(str(self.evidence/'npu_trace.json'))
                    self.profile_done=True
                self._cross=None;self._features=None;self._encoder=None;self._bridges=0;self._decoder_calls=0;self._position_first=None;self._position_last=None
                generated=self.model.generate(**inputs,generation_config=self.gen,return_dict_in_generate=True,output_scores=False)
            tokens=generated.sequences[0].cpu().tolist();assert tokens[0]==self.gen.decoder_start_token_id and self._bridges==1
            raw=self.processor.batch_decode(generated.sequences,skip_special_tokens=True)[0].strip();new_count=len(tokens)-1;truncated=new_count>=self.max_new_tokens and tokens[-1] not in self.eos
            archive=self.evidence/'visual_features'/('%05d.npz'%(self.calls-1));np.savez_compressed(archive,encoder_output=self._encoder.detach().cpu().numpy(),decoder_cross_attention_features=self._features.detach().cpu().numpy())
            entry.update(raw_output=raw,generation_sequence_ids=tokens,generation_count=new_count,truncated=truncated,stop_reason='length' if truncated else 'eos' if tokens[-1] in self.eos else 'other',visual_cross_attention_verified=True,visual_feature_archive=str(archive),visual_feature_archive_sha256=sha(archive),encoder_output=record(self._encoder),decoder_cross_attention_features=record(self._features),actual_cross_attention_input=self._cross,actual_decoder_beam_batch=self._cross['shape'][0],decoder_forward_calls=self._decoder_calls,position_first=self._position_first,position_last=self._position_last)
            log.info('Native %s NPU sample %s: generated=%s beams=%s cache=%s truncated=%s raw=%r',self.protocol['decoder_family'],self.calls,new_count,self.gen.num_beams,self.gen.use_cache,truncated,raw[:200]);return raw
        except Exception:
            entry['error']=traceback.format_exc();log.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started;clean=prompt[:-len('\n\nProvide only the final answer.')] if (prompt or '').endswith('\n\nProvide only the final answer.') else prompt
            self.details[(clean.strip(),media_path)]=entry
            with self.trace.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._features=None;self._encoder=None

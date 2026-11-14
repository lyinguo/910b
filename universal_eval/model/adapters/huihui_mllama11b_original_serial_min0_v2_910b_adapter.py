"""Original own Mllama BF16/eager generation; native vision and LM on two NPUs.

Only placement metadata, exact device transfers and read-only observers are added.
Native learned forwards, eager attention, cache and generation are inherited.
"""
import os, pathlib, json, datetime, logging, time, traceback, contextlib, collections, functools
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch, torch_npu, numpy as np
from PIL import Image
from transformers import MllamaForConditionalGeneration
from universal_eval.model.interface import IModel
from scripts.mllama11b_original_independent_cpu_910b import sha, fp, tr

log=logging.getLogger('universal_eval.model.adapters.mllama11b_original_910b')

class HuihuiMllamaOriginalPlacedModel(MllamaForConditionalGeneration):
    @property
    def device(self):
        # GenerationMixin routes token/mask tensors to this metadata device.
        # Vision transfers below preserve every input bit and native operation.
        return self.model.language_model.embed_tokens.weight.device

class HuihuiMllama11BOriginalSerialMin0V2_910BAdapter(IModel):
    def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        assert sha(protocol_file)==protocol_sha256
        self.p=json.loads(pathlib.Path(protocol_file).read_text())
        self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha=protocol_sha256
        self.devices=list(npu_devices or [8,9]);assert len(self.devices)==2 and len(set(self.devices))==2
        self.vision_device=torch.device('npu:'+str(self.devices[0]));self.device=torch.device('npu:'+str(self.devices[1]))
        self.group_calls=0;self.details={};self.lineage={};self.profile_done=set()
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('Huihui_Mllama11B_original_serial_910b_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl';self.activation_trace=self.evidence/'native_activation_execution.jsonl'
    @property
    def model_name(self):return self.p['model_id']
    @property
    def model_type(self):return 'MllamaForConditionalGeneration'
    @property
    def inference_framework(self):return 'Native Transformers / torch_npu original BF16 eager true size-grouped greedy4096'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
    def _placement(self,name):
        return self.vision_device if name.startswith(('model.vision_model.','model.multi_modal_projector.')) else self.device
    def load(self,model_path,**kwargs):
        from transformers import AutoConfig,AutoProcessor,GenerationConfig
        from transformers.models.mllama import modeling_mllama as mm
        from safetensors import safe_open
        p=self.p;wp=pathlib.Path(model_path).resolve()
        assert p['model_id']=='huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated' and str(wp)==p['weight_path']
        assert sha(p['CPU200_protocol'])==p['CPU200_protocol_sha256'] and p['original_runtime_precision_explicit_BF16_and_frozen_eager_proven']
        for n,h in p['own_resource_sha256'].items():assert pathlib.Path(n).name==n and sha(wp/n)==h
        for n,h in p['runtime_and_original_source_sha256'].items():assert sha(n)==h
        assert sha(p['dataset_path'])==p['dataset_SHA'] and p['original_outer_batch_size']==1 and p['original_runtime_serial1_proven_by_pipeline_statistics_true_run_log_and_frozen_SequentialA100MLU'] and p['group_count']==len(p['groups'])==200 and p['original_explicit_min_new_tokens']==0 and p['original_frozen_TransformersVLM_predict_explicit_zero_min_new_tokens_preserved']
        assert torch.npu.is_available();torch.set_num_threads(4)
        torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
        for dev in [self.vision_device,self.device]:
            with torch.npu.device(dev):
                for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
        torch.npu.set_device(self.device)
        cfg=AutoConfig.from_pretrained(wp,local_files_only=True)
        assert cfg.architectures==['MllamaForConditionalGeneration'] and not cfg.text_config.tie_word_embeddings
        assert cfg.text_config.num_hidden_layers==40 and cfg.text_config.cross_attention_layers==[3,8,13,18,23,28,33,38]
        assert cfg.vision_config.image_size==560 and cfg.vision_config.max_num_tiles==4 and cfg.vision_config.num_hidden_layers==32 and cfg.vision_config.num_global_layers==8
        cfg._attn_implementation='eager';cfg.text_config._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
        with torch.device('meta'):self.model=HuihuiMllamaOriginalPlacedModel(cfg)
        assert self.model.forward.__func__ is MllamaForConditionalGeneration.forward
        assert self.model.prepare_inputs_for_generation.__func__ is MllamaForConditionalGeneration.prepare_inputs_for_generation
        assert not self.model.is_remote_code() and not getattr(self.model,'_keep_in_fp32_modules',None) and not getattr(self.model,'_keep_in_fp32_modules_strict',None)
        expected={r['source_key']:r for r in p['all906_source_to_native_semantic_name_shape_mapping_verified']}
        params=dict(self.model.named_parameters());assert len(params)==len(expected)==906
        assert {r['target_key'] for r in expected.values()}==set(params)
        loaded=[];filled=set()
        for shard,head in p['all_source_file_sha256'].items():
            q=wp/shard;assert pathlib.Path(shard).name==shard and q.stat().st_size==head['bytes'] and sha(q)==head['sha256']
            with safe_open(str(q),framework='pt',device='cpu') as f:
                for key in f.keys():
                    r=expected[key];target=r['target_key'];x=f.get_tensor(key)
                    assert key not in filled and x.dtype==torch.bfloat16 and list(x.shape)==r['shape']==list(params[target].shape)
                    assert fp(x)==r['full_sha256']==r['original_CPU_BF16_cast_sha256'] and torch.isfinite(x).all()
                    dev=self._placement(target);y=x.to(dev);assert y.dtype==x.dtype and fp(y)==r['full_sha256']
                    parent,leaf=target.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,torch.nn.Parameter(y,requires_grad=False))
                    loaded.append(dict(r,source_sha256=fp(x),original_BF16_cast_sha256=fp(x.bfloat16()),actual_NPU_sha256=fp(y),device=str(dev),runtime_dtype=str(y.dtype),original_cast_changes=0));filled.add(key)
            log.info('Own Mllama full source/cast/NPU SHA loaded %s/906',len(loaded))
        assert filled==set(expected)
        self.model.model.language_model.rotary_emb=mm.MllamaRotaryEmbedding(cfg.text_config,device='cpu').to(self.device)
        assert all(not v.is_meta and v.dtype==torch.bfloat16 and v.device==self._placement(n) for n,v in self.model.named_parameters())
        assert all(v.dtype==torch.float32 and v.device==self.device and not v.is_meta for v in self.model.buffers())
        assert self.model.lm_head.weight is not self.model.model.language_model.embed_tokens.weight
        assert self.model.lm_head.weight.shape==(128256,4096) and self.model.model.language_model.embed_tokens.weight.shape==(128264,4096)
        self.model.eval();self.processor=AutoProcessor.from_pretrained(wp,local_files_only=True);self.tokenizer=self.processor.tokenizer
        assert self.tokenizer.padding_side=='right' and len(self.tokenizer)==128257 and self.processor.image_token_id==128256
        assert json.loads(json.dumps(self.processor.image_processor.to_dict()))==p['native_image_processor']
        self.model.generation_config=GenerationConfig.from_pretrained(wp,local_files_only=True)
        self.generation_config,_=self.model._prepare_generation_config(None,max_new_tokens=4096,min_new_tokens=0,do_sample=False,temperature=0.0)
        assert self.generation_config.to_dict()==p['original_resolved_generation'] and self.generation_config.use_cache and self.generation_config.num_beams==1
        self.activation_modules={n:type(m).__name__ for n,m in self.model.named_modules() if n.endswith(('act_fn','activation_fn'))}
        assert self.activation_modules==p['native_activation_modules'] and len(self.activation_modules)==80
        self._reset();self._install_observers()
        vals=torch.linspace(-8,8,20001,dtype=torch.float32).bfloat16();probe={'input':vals.view(torch.uint8).numpy()}
        for kind in ['none','tanh']:
            probe['CPU_'+kind]=torch.nn.functional.gelu(vals,approximate=kind).view(torch.uint8).numpy()
            for dev in [self.vision_device,self.device]:probe[str(dev)+'_'+kind]=torch.nn.functional.gelu(vals.to(dev),approximate=kind).cpu().view(torch.uint8).numpy()
        pq=self.evidence/'native_GELU_BF16_semantics.npz';np.savez_compressed(pq,**probe)
        wq=self.evidence/'weight_loading.json';wq.write_text(json.dumps(loaded,indent=2))
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir();(snapshot/'original_protocol.json').write_bytes(pathlib.Path(self.protocol_file).read_bytes())
        for n,h in p['runtime_and_original_source_sha256'].items():
            path=pathlib.Path(n);dest=snapshot/path.relative_to('/');assert dest.resolve().is_relative_to(snapshot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());assert sha(dest)==h
        self.lineage={'hardware':'Ascend910B','devices':self.devices,'protocol':'Huihui_Mllama11B_original_BF16_eager_true_serial1_own_single_BOS_free_generation','protocol_file':self.protocol_file,'protocol_sha256':self.protocol_sha,'weight_tensor_count':906,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(wq),'per_sample_evidence':str(self.trace),'per_group_evidence':str(self.evidence/'groups'),'activation_execution_trace':str(self.activation_trace),'activation_modules':self.activation_modules,'profiler_traces':{'vision':str(self.evidence/'vision_npu_trace.json'),'language':str(self.evidence/'language_npu_trace.json')},'native_GELU_probe':{'path':str(pq),'sha256':sha(pq),'CPU_threads':4,'only_diagnostic':True},'original_max_new_tokens':4096,'original_min_new_tokens':0,'original_outer_batch_size':1,'actual_batching':'Original SequentialA100MLU serial1; one native generate per sample, not fallback from grouped generation','generation_config':self.generation_config.to_dict(),'runtime_dtype':'BF16','original_untied_head128256_embedding128264_tokenizer128257_preserved':True,'original_single_BOS128000_preserved':True,'original_eager_attention_all_text_and_vision':True,'image_processor':self.processor.image_processor.to_dict(),'analytic_F32_buffers':{n:tr(v) for n,v in self.model.named_buffers()},'HF32_enabled':{k:bool(getattr(torch.npu,k).allow_hf32) for k in ['conv','matmul']},'module_placements':{n:str(next(m.parameters(recurse=False)).device) for n,m in self.model.named_modules() if n and any(True for _ in m.parameters(recurse=False))},'placement_only_metadata_override':'Subclass device property equals original language input embedding device; inherited native forward/prepare/generate and all learned layers unchanged. Vision inputs moved bit-exact to first NPU and original projector output moved bit-exact to second. Native full-group eager attention remains intact.','precision_disclosure':'Own906 source BF16 tensors with full source/cast/actual NPU SHA, original untied128256 output head and128264 input embedding. Native32+8 vision/projector on first NPU; native40 text including8 cross-only layers and32 self layers plus head on second, all BF16, original F32 analytic frequencies parsed on CPU then exact copied. Own560/max4tiles/single BOS/authentic ByteLevel postprocessor/own right-pad metadata/4096/original SequentialA100MLU serial1/cache/eager unchanged. Own source, template and tokenizer remain distinct from alpindale; no cross-model weights or tokenizer substitution. No CPU learned offload, replacement resources, changed attention, budget reduction or low-bit acceleration claim. CPU controls diagnostic only. Two profilers capture first actual native vision and language prefill, not the entire200.'}
    def _reset(self):
        self._arrays={};self._steps=[];self._step=None;self._counts={n:0 for n in getattr(self,'activation_modules',{})}
        self._vision_calls=0;self._previous_cache=None;self._last_prefix=None;self._projected=None;self._cross_key={};self._cross_value={}
    def _store(self,name,t):
        assert t.device in [self.vision_device,self.device]
        self._arrays[name]=t.detach().cpu().contiguous().clone();return tr(self._arrays[name])
    def _native_profile(self,which,dev,fn,args,kw):
        with torch.npu.device(dev):
            if which in self.profile_done:return fn(*args,**kw)
            with contextlib.chdir(self.evidence):
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
                    out=fn(*args,**kw);torch.npu.synchronize(dev)
                prof.export_chrome_trace(str(self.evidence/(which+'_npu_trace.json')))
            self.profile_done.add(which);return out
    def _install_observers(self):
        vision=self.model.model.vision_model;text=self.model.model.language_model;projector=self.model.model.multi_modal_projector
        original_v=vision.forward;original_l=text.forward;original_prepare=self.model.prepare_inputs_for_generation
        @functools.wraps(original_v)
        def vforward(*args,**kw):
            assert not args
            moved={k:v.to(self.vision_device) if isinstance(v,torch.Tensor) else v for k,v in kw.items()}
            assert all(not isinstance(v,torch.Tensor) or torch.equal(v.cpu(),kw[k].cpu()) for k,v in moved.items())
            self._step['vision_input_device_transfer_bitwise']=True
            return self._native_profile('vision',self.vision_device,original_v,args,moved)
        @functools.wraps(original_l)
        def lforward(*args,**kw):return self._native_profile('language',self.device,original_l,args,kw)
        @functools.wraps(original_prepare)
        def prepare(input_ids,*args,**kw):
            self._last_prefix=input_ids.detach().cpu().clone();return original_prepare(input_ids,*args,**kw)
        vision.forward=vforward;text.forward=lforward;self.model.prepare_inputs_for_generation=prepare
        def model_pre(module,args,kw):
            assert self._step is None and self._last_prefix is not None
            self._step={'step':len(self._steps)+1,'full_prefix':tr(self._last_prefix),'input_ids':kw['input_ids'].cpu().tolist(),'position_ids':None if kw.get('position_ids') is None else kw['position_ids'].cpu().tolist(),'attention_mask2':kw['attention_mask'].cpu().tolist(),'cross_attention_mask4':tr(kw['cross_attention_mask']),'pixels_present':kw.get('pixel_values') is not None,'all40_attention':[]}
            self._store('%04d_original_cross_attention_mask4'%self._step['step'],kw['cross_attention_mask'])
        def model_post(module,args,out):
            step=self._step;n=step['step'];logits=out.logits[:,-1]
            assert logits.shape==(len(self._items),128256) and logits.dtype==torch.bfloat16 and logits.device==self.device and torch.isfinite(logits).all()
            self._store('%04d_full_vocab_logits'%n,logits);step['actual_logits_shape']=list(out.logits.shape)
            cache=out.past_key_values;assert len(cache.layers)==40;step['cache40']=[];now=[]
            for i,c in enumerate(cache.layers):
                k,v=c.keys,c.values;cross=i in self.model.config.text_config.cross_attention_layers
                assert k.device==v.device==self.device and k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape and list(k.shape[:2])==[len(self._items),8] and k.shape[-1]==128
                assert k.shape[-2]==(6404 if cross else len(self._active_ids[0])+n-1)
                prefix=True
                if self._previous_cache is not None:
                    pk,pv=self._previous_cache[i];prefix=torch.equal(k[...,:pk.shape[-2],:],pk) and torch.equal(v[...,:pv.shape[-2],:],pv);assert prefix
                    if cross:assert k.shape==pk.shape and k.data_ptr()==pk.data_ptr() and v.data_ptr()==pv.data_ptr()
                    else:assert k.shape[-2]==pk.shape[-2]+1
                if cross and n==1:
                    assert torch.equal(k,self._cross_key[i]) and torch.equal(v,self._cross_value[i]);step.setdefault('cross_prefill_projection_cache_bitwise',[]).append({'layer':i,'key':tr(k),'value':tr(v),'original_projection_received_in_native_cache_bitwise':True})
                step['cache40'].append({'layer':i,'type':'cross' if cross else 'self','shape':list(k.shape),'dtype':str(k.dtype),'device':str(k.device),'original_prefix_NPU_bitwise_unchanged':prefix});now.append((k.detach(),v.detach()))
            self._cross_key={};self._cross_value={};self._previous_cache=now;assert len(step['all40_attention'])==40
            self._steps.append(step);self._step=None
            if n==1 or n%128==0:
                with (self.evidence/'generation_progress.jsonl').open('a') as f:f.write(json.dumps({'group':self.group_calls,'batch_size':len(self._items),'completed_decoder_forward':n,'generated_before_this_forward':n-1})+'\n')
        def vision_pre(module,args,kw):
            self._store('actual_vision_original_F32_pixels',kw['pixel_values']);self._store('actual_aspect_ratio_ids',kw['aspect_ratio_ids']);self._store('actual_aspect_ratio_mask',kw['aspect_ratio_mask'])
        def patch(module,args,out):
            self._store('actual_patch_input_BF16',args[0]);self._store('actual_patch_embeddings',out)
        def vision_post(module,args,out):
            self._vision_calls+=1;self._store('actual_vision_output',out.last_hidden_state)
        def project(module,args,out):
            assert torch.equal(args[0].cpu(),self._arrays['actual_vision_output']) and out.device==self.vision_device and out.dtype==torch.bfloat16
            self._store('actual_projected_vision',out);moved=out.to(self.device)
            assert moved.dtype==out.dtype and torch.equal(moved.cpu(),out.cpu());self._projected=moved.reshape(-1,1601,4096)
            self._step['projector_device_transfer_bitwise']=True;return moved
        def language_pre(module,args,kw):
            if self._step['step']==1:
                assert kw['cross_attention_states'].device==self.device and torch.equal(kw['cross_attention_states'],self._projected)
                assert list(kw['cross_attention_states'].shape)==[len(self._items)*4,1601,4096]
                self._step['actual_projected_visual_received_bitwise']=True
            else:assert kw['cross_attention_states'] is None
            self._store('%04d_expanded_cross_attention_mask'%self._step['step'],kw['cross_attention_mask'])
            self._store('%04d_full_text_row_mask'%self._step['step'],kw['full_text_row_masked_out_mask'])
        def embedding(module,args,out):
            if self._step['step']==1:self._store('actual_prefill_text_embeds',out)
        def rope(module,args,kw,out):
            positions=kw.get('position_ids',args[1] if len(args)>1 else None)
            self._step['actual_text_position_ids']=positions.cpu().tolist();self._store('%04d_text_cos'%self._step['step'],out[0]);self._store('%04d_text_sin'%self._step['step'],out[1])
        def attention(i,cross):
            def hook(module,args,kw):
                if cross:
                    if self._step['step']==1:
                        assert torch.equal(kw['cross_attention_states'],self._projected)
                    else:assert kw['cross_attention_states'] is None
                mask=kw.get('attention_mask');record=None if mask is None else tr(mask);kind='cross' if cross else 'self'
                if kind+'_mask' not in self._step:
                    self._step[kind+'_mask']=record
                    if mask is not None:self._store('%04d_%s_mask'%(self._step['step'],kind),mask)
                else:assert self._step[kind+'_mask']==record
                self._step['all40_attention'].append({'layer':i,'type':kind,'mask':record,'device':str(next(module.parameters()).device)})
            return hook
        def normkey(i):
            def hook(module,args,out):
                if self._step['step']==1:self._cross_key[i]=out
            return hook
        def valproj(i):
            def hook(module,args,out):
                if self._step['step']==1:self._cross_value[i]=out.view(len(self._items),-1,8,128).transpose(1,2)
            return hook
        def activation(name):
            def hook(module,args,out):
                expected=self.vision_device if '.vision_model.' in name else self.device
                assert out.device==expected and out.dtype==torch.bfloat16;self._counts[name]+=1
            return hook
        def vision_attention(name):
            def hook(module,args,kw):
                mask=kw['attention_mask'];record=tr(mask)
                if 'vision_attention_mask' not in self._step:
                    self._step['vision_attention_mask']=record;self._store('actual_vision_attention_mask',mask)
                else:assert self._step['vision_attention_mask']==record
                self._step.setdefault('all40_vision_attention',[]).append({'name':name,'mask':record,'device':str(next(module.parameters()).device)})
            return hook
        self.model.register_forward_pre_hook(model_pre,with_kwargs=True);self.model.register_forward_hook(model_post)
        vision.register_forward_pre_hook(vision_pre,with_kwargs=True);vision.register_forward_hook(vision_post);vision.patch_embedding.register_forward_hook(patch);projector.register_forward_hook(project)
        text.register_forward_pre_hook(language_pre,with_kwargs=True);text.embed_tokens.register_forward_hook(embedding);text.rotary_emb.register_forward_hook(rope,with_kwargs=True)
        for i,layer in enumerate(text.layers):
            cross=i in text.cross_attention_layers;att=layer.cross_attn if cross else layer.self_attn
            att.register_forward_pre_hook(attention(i,cross),with_kwargs=True)
            if cross:att.k_norm.register_forward_hook(normkey(i));att.v_proj.register_forward_hook(valproj(i))
        for n,m in self.model.named_modules():
            if n in self.activation_modules:m.register_forward_hook(activation(n))
            if n.startswith('model.vision_model.') and n.endswith('.self_attn'):m.register_forward_pre_hook(vision_attention(n),with_kwargs=True)
    def _generate_group(self,items):
        self.group_calls+=1;self._reset();self._items=items;home=self.evidence/'groups'/('%04d'%self.group_calls);home.mkdir()
        rec={'group':self.group_calls,'indices':[x['index'] for x in items],'batch_size':len(items),'error':None};start=time.monotonic()
        try:
            images=[];texts=[]
            for x in items:
                assert sha(x['media_path'])==self.p['input_records'][x['index']]['media_sha256']
                with Image.open(x['media_path']) as im:images.append(im.convert('RGB'))
                texts.append(self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':x['model_prompt']}]}],add_generation_prompt=True,tokenize=False))
            inp=self.processor(text=texts[0] if len(items)==1 else texts,images=[images[0]] if len(items)==1 else [[im] for im in images],return_tensors='pt',padding=True)
            for slot,x in enumerate(items):
                r=self.p['input_records'][x['index']];ids=inp['input_ids'][slot][inp['attention_mask'][slot].bool()].tolist()
                assert texts[slot]==r['rendered'] and ids==r['unpad_ids'] and ids[0]==128000 and ids.count(128000)==1
            rec.update(CPU_inputs={k:tr(v) for k,v in inp.items()},input_ids=inp['input_ids'].tolist(),attention_mask=inp['attention_mask'].tolist(),rendered=texts)
            device_inputs={k:v.to(self.device) for k,v in inp.items()};self._active_ids=device_inputs['input_ids'];length=self._active_ids.shape[-1]
            with torch.inference_mode():result=self.model.generate(**device_inputs,max_new_tokens=4096,min_new_tokens=0,do_sample=False,temperature=0.0)
            generated=result[:,length:].cpu();tokens=generated.tolist();raw=[s.strip() for s in self.processor.batch_decode(generated,skip_special_tokens=True)]
            assert self._vision_calls==1 and self._steps[0]['actual_projected_visual_received_bitwise'] and len(self._steps[0]['cross_prefill_projection_cache_bitwise'])==8
            assert len(self._steps)==generated.shape[1]<=4096
            for n,c in self._counts.items():assert c==(1 if '.vision_model.' in n else len(self._steps)),(n,c)
            q=home/'actual_visual_positions_cache_full_vocab.npz';np.savez_compressed(q,**{k:v.reshape(-1).view(torch.uint8).numpy() for k,v in self._arrays.items()})
            rec.update(steps=self._steps,arrays={k:tr(v) for k,v in self._arrays.items()},archive=str(q),archive_sha256=sha(q),generated_padded_token_ids=tokens,raw_outputs=raw,vision_calls=1,activation_counts=self._counts,elapsed_seconds=time.monotonic()-start)
            jq=home/'batch_execution.json';jq.write_text(json.dumps(rec,ensure_ascii=False,indent=2))
            from universal_eval.algorithm.native_explicit_mc_v2 import explicit_choice
            with self.trace.open('a',encoding='utf8') as f:
                for slot,x in enumerate(items):
                    eos=next((i for i,t in enumerate(tokens[slot]) if t in [128001,128008,128009]),None);count=len(tokens[slot]) if eos is None else eos+1
                    e={'index':x['index'],'attempt':x['index']+1,'prompt':x['prompt'],'actual_model_prompt':x['model_prompt'],'media_path':x['media_path'],'source_image_sha256':self.p['input_records'][x['index']]['media_sha256'],'group':self.group_calls,'group_batch_size':len(items),'group_slot':slot,'group_execution':str(jq),'group_execution_sha256':sha(jq),'generation_token_ids':tokens[slot],'first_EOS_index':eos,'generation_count':count,'raw_output':raw[slot],'error':None,'empty':not raw[slot],'invalid':explicit_choice(raw[slot]) is None,'truncated':count>=4096}
                    self.details[(x['prompt'].strip(),x['media_path'])]=e;f.write(json.dumps(e,ensure_ascii=False)+'\n')
            with self.activation_trace.open('a') as f:f.write(json.dumps({'group':self.group_calls,'indices':rec['indices'],'batch_size':len(items),'decoder_forwards':len(self._steps),'module_call_counts':self._counts,'vision_device':str(self.vision_device),'language_device':str(self.device),'dtype':'torch.bfloat16','error':None})+'\n')
            log.info('Mllama real NPU group%s batch%s indices%s steps%s',self.group_calls,len(items),rec['indices'],len(self._steps));return raw
        except Exception:
            rec['error']=traceback.format_exc();(home/'failed_batch_execution.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));log.error(rec['error']);raise
        finally:self._reset()
    def batch_predict(self,batch):
        assert len(batch)==1
        lookup={(x['model_prompt'],x['media_path']):x for x in self.p['input_records']};groups=collections.OrderedDict();result=[None]*len(batch)
        for slot,(prompt,media) in enumerate(batch):
            key=(prompt,str(pathlib.Path(media).resolve()));assert key in lookup;row=lookup[key]
            with Image.open(media) as im:group=(im.width//64,im.height//64)
            groups.setdefault(group,[]).append(dict(row,media_path=media,slot=slot))
        for items in groups.values():
            outputs=self._generate_group(items)
            for x,out in zip(items,outputs):result[x['slot']]=out
        assert all(isinstance(v,str) for v in result);return result
    def predict(self,prompt,media_path=None):raise RuntimeError('Use original serial1 algorithm batch entry point; no batching change')
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),media_path),{'error':'Missing execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        self.model=None
        for dev in [self.vision_device,self.device]:
            with torch.npu.device(dev):torch.npu.empty_cache()

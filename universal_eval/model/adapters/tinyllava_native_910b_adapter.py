"""Yi-VL family: original BF16 weights, explicit Ascend NPU layer placement.

Reuses the project's existing Yi projection, ChatML and mean continuation
log-probability protocol. No low-bit kernel or quantization is used.
"""
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import traceback
import contextlib

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

import torch
import torch_npu
from universal_eval.model.adapters.yivl34b_adapter import (
    YiVL34BAdapter, YiVL34BProjector, _IMG_TOKEN,
    _strip_eval_instruction, _parse_options,
)

_IMG_TOKEN='[PAD]'
logger=logging.getLogger('universal_eval.model.adapters.tinyllava_native_910b')

def fingerprint(tensor):
    value=tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

def move_tree(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,tuple):return tuple(move_tree(x,device) for x in value)
    if isinstance(value,list):return [move_tree(x,device) for x in value]
    if isinstance(value,dict):return {k:move_tree(v,device) for k,v in value.items()}
    return value

class TinyLlavaNative910BAdapter(YiVL34BAdapter):
    def __init__(self,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        super().__init__(**kwargs)
        self.devices=list(npu_devices or [0,1])
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        self.evidence=Path(evidence_dir).resolve()/('TinyLlava_native_910b_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False)
        self.trace_path=self.evidence/'sample_execution.jsonl'
        self.load_records=[]
        self.calls=0
        self._cached_features=None
        self._profile_done=False
        self._lineage={}
        self._details={};self._eos_ids=[2];self._source_rotary_inv_freq=None

    @property
    def inference_framework(self):return 'PyTorch + torch_npu / Transformers; BF16 continuation scoring'

    def download(self,*args,**kwargs):
        raise RuntimeError('Model downloads are disabled; existing local weights are required')

    def _destination(self,name):
        match=re.match(r'model.language_model.layers\.(\d+)\.',name)
        if match:
            index=min(len(self.devices)-1,int(match[1])*len(self.devices)//self._num_hidden_layers)
            return torch.device('npu:'+str(self.devices[index]))
        if name=='lm_head.weight' or name.startswith('model.language_model.norm.'):
            return torch.device('npu:'+str(self.devices[-1]))
        return torch.device('npu:'+str(self.devices[0]))

    @staticmethod
    def _read_shard(shard):
        return torch.load(shard,map_location='cpu',weights_only=True,mmap=True)

    def load(self,model_path,**kwargs):
        from transformers import AutoTokenizer,AutoImageProcessor,AutoModelForImageTextToText
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        if not torch.npu.is_available():raise RuntimeError('Ascend NPU is unavailable')
        self._model_path=model_path;self._model_name=Path(model_path).parent.name+'/'+Path(model_path).name
        raw=json.loads((Path(model_path)/'config.json').read_text());self._num_hidden_layers=raw['num_hidden_layers']
        if not (Path(model_path)/'pytorch_model.bin').is_file():raise FileNotFoundError(model_path)
        self._tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
        self._img_token_id=self._tokenizer.convert_tokens_to_ids(_IMG_TOKEN)
        assert self._img_token_id not in (None,self._tokenizer.unk_token_id)
        vit_dir=self._find_vit_dir(model_path)
        self._image_processor=AutoImageProcessor.from_pretrained(vit_dir,trust_remote_code=True,local_files_only=True)
        cfg,_=self._build_llava_config(model_path,vit_dir,raw['vocab_size'],self._img_token_id)
        self._vision_layer=cfg.vision_feature_layer;self._vision_select=cfg.vision_feature_select_strategy
        dtype=torch.bfloat16
        with torch.device('meta'):
            self._model=AutoModelForImageTextToText.from_config(cfg,dtype=dtype,attn_implementation='sdpa')
            self._model.model.multi_modal_projector=torch.nn.Identity()
            self._projector=torch.nn.Linear(cfg.vision_config.hidden_size,cfg.text_config.hidden_size)
        params=dict(self._model.named_parameters())
        projector_params=dict(self._projector.named_parameters())
        filled=set();projector_filled=set()
        for shard in self._shards(model_path):
            started=time.monotonic();sd=self._read_shard(shard)
            for key,value in sd.items():
                if key.endswith('.self_attn.rotary_emb.inv_freq'):
                    if self._source_rotary_inv_freq is None:self._source_rotary_inv_freq=value.detach().clone()
                    elif not torch.equal(self._source_rotary_inv_freq,value):raise RuntimeError('Layer-specific rotary buffers differ')
                    self.load_records.append({'source_file':os.path.basename(shard),'source_key':key,'kind':'serialized_rotary_buffer','source_dtype':str(value.dtype),'values':value.float().tolist(),'policy':'Preserved exactly in equivalent shared rotary buffer'})
                    continue
                is_projector=key.startswith('model.mm_projector.')
                name=key[len('model.mm_projector.'):] if is_projector else self._ckpt_to_param(key)
                if name is None:raise RuntimeError('Unexpected skipped source tensor: '+key)
                collection=projector_params if is_projector else params
                if name not in collection:raise RuntimeError('Unmapped source tensor: '+key+' -> '+name)
                original=collection[name]
                assert tuple(original.shape)==tuple(value.shape),(key,original.shape,value.shape)
                device=torch.device('npu:'+str(self.devices[0])) if is_projector else self._destination(name)
                target=value.to(device=device,dtype=dtype).contiguous()
                flat_source=value.reshape(-1)
                positions=sorted(set([0,value.numel()//2,value.numel()-1]))
                expected=flat_source[positions].to(dtype)
                actual=target.reshape(-1)[torch.tensor(positions,device=device)].cpu()
                if not torch.equal(expected,actual):raise RuntimeError('Device weight slice mismatch: '+key)
                owner=self._projector if is_projector else self._model
                parent,leaf=name.rsplit('.',1) if '.' in name else ('',name)
                module=owner.get_submodule(parent) if parent else owner
                setattr(module,leaf,torch.nn.Parameter(target,requires_grad=False))
                (projector_filled if is_projector else filled).add(name)
                self.load_records.append({'source_file':os.path.basename(shard),'source_key':key,'target_key':name,'projector':is_projector,'shape':list(value.shape),'source_dtype':str(value.dtype),'execution_dtype':str(dtype),'device':str(device),'slice_positions':positions,'source_slice':expected.float().tolist(),'device_slice':actual.float().tolist()})
            del sd
            logger.info('NPU loaded shard %s in %.1fs (%s tensors)',os.path.basename(shard),time.monotonic()-started,len(self.load_records))
        assert filled==set(params),sorted(set(params)-filled)
        assert projector_filled==set(projector_params),sorted(set(projector_params)-projector_filled)
        first=torch.device('npu:'+str(self.devices[0]))
        # Rebuild nonpersistent rotary buffers from the exact original config.
        self._model.model.language_model.rotary_emb=LlamaRotaryEmbedding(cfg.text_config,device=first)
        if self._source_rotary_inv_freq is None:raise RuntimeError('Original rotary buffers missing')
        rotary=self._model.model.language_model.rotary_emb
        rotary.inv_freq=self._source_rotary_inv_freq.to(device=first,dtype=torch.float32)
        rotary.original_inv_freq=rotary.inv_freq.clone()

        for name,buffer in list(self._model.named_buffers()):
            if buffer.is_meta:
                if name.endswith('position_ids'):
                    replacement=torch.arange(buffer.numel(),device=first).reshape(buffer.shape)
                    parent,leaf=name.rsplit('.',1);setattr(self._model.get_submodule(parent),leaf,replacement)
                else:raise RuntimeError('Unmaterialized nonparameter buffer: '+name)
        for name,param in self._model.named_parameters():
            assert param.device.type=='npu',(name,param.device)
        def hook(device):
            def dispatch(module,args,kw):return move_tree(args,device),move_tree(kw,device)
            return dispatch
        for i,layer in enumerate(self._model.model.language_model.layers):
            layer.register_forward_pre_hook(hook(self._destination('model.language_model.layers.'+str(i)+'.weight')),with_kwargs=True)
        self._model.model.language_model.norm.register_forward_pre_hook(hook(self._destination('model.language_model.norm.weight')),with_kwargs=True)
        self._model.lm_head.register_forward_pre_hook(hook(self._destination('lm_head.weight')),with_kwargs=True)
        def check_image_bridge(module,args,kw):
            if self._bridge_checked:return
            actual=kw['inputs_embeds'][self._active_ids.to(first)==self._img_token_id]
            expected=self._cached_features[0]
            if actual.shape!=expected.shape or not torch.equal(actual,expected):
                raise RuntimeError('Actual language-model image embeddings differ from projected vision features')
            if not bool(expected.abs().sum()>0):raise RuntimeError('Zero image features')
            self._bridge_checked=True
        self._model.model.language_model.register_forward_pre_hook(check_image_bridge,with_kwargs=True)
        self._model.eval();self._projector.eval();self._patch_image_features()
        self._n_patches=self._num_patches();self._img_token_repeat=_IMG_TOKEN*self._n_patches
        self._model.generation_config.eos_token_id=self._eos_ids
        self._model.generation_config.pad_token_id=0
        self._model.generation_config.do_sample=False
        (self.evidence/'weight_loading.json').write_text(json.dumps(self.load_records,indent=2))
        self._lineage={'hardware':'Ascend 910B','torch':torch.__version__,'torch_npu':torch_npu.__version__,'devices':self.devices,'precision':'Original checkpoint tensors executed as BF16; no low-bit kernel or quantization','protocol':'Mean teacher-forced log-probability of each full option continuation; same prefix and scoring as historical CUDA adapter','weight_tensor_count':len(self.load_records),'all_expected_parameters_loaded':True,'weight_loading_evidence':str(self.evidence/'weight_loading.json'),'per_sample_evidence':str(self.trace_path),'profiler_trace':str(self.evidence/'npu_trace.json'),'module_placements':{name:str(next(module.parameters()).device) for name,module in self._model.named_modules() if name and any(True for _ in module.parameters(recurse=False))}}
        logger.info('TinyLLaVA NPU ready: %s source tensors accounted for (parameters loaded, identical rotary buffers preserved), %s image tokens, evidence=%s',len(self.load_records),self._n_patches,self.evidence)

    def _vision_features(self,pixel_values):
        if self._cached_features is not None:return self._cached_features
        return super()._vision_features(pixel_values)

    def _forward_logits(self,ids,pixel_values,logits_to_keep=0):
        self._active_ids=ids
        def execute():
            output=super(TinyLlavaNative910BAdapter,self)._forward_logits(ids,pixel_values,logits_to_keep)
            if not bool(torch.isfinite(output).all()):raise RuntimeError('Nonfinite NPU logits')
            return output
        if not self._profile_done:
            self._profile_done=True
            # The SDK creates intermediate export_only_prof_dir relative to cwd.
            # Keep that raw diagnostic data with this run's evidence as well.
            with contextlib.chdir(self.evidence):
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                    output=execute()
                    torch.npu.synchronize()
                profile.export_chrome_trace(str(self.evidence/'npu_trace.json'))
            return output
        return execute()

    def _discriminative(self,clean,pixel_values,options):
        # Exact historical prefix and per-continuation normalization. Any failed
        # option is an explicit inference error, never a silently omitted choice.
        prefix=self._chatml_text(clean).rstrip('\n')+'The correct answer is'
        pre=self._tokenize(prefix)
        logits=self._forward_logits(pre,pixel_values,logits_to_keep=1)
        letters=self._letter_scores(logits[-1],[letter for letter,_ in options])
        info={'letter_scores':letters,'option_scores':{},'option_token_counts':{},'decision':'content','truncated':False,'input_ids':pre[0].tolist()}
        best=None;best_score=float('-inf')
        for letter,text in options:
            cont=' '+letter+'. '+text
            n=len(self._tokenizer(cont,add_special_tokens=False)['input_ids'])
            ids=self._tokenize(prefix+cont)
            if n<=0:raise RuntimeError('Empty option continuation')
            logits=self._forward_logits(ids,pixel_values,logits_to_keep=n+1)
            logp=torch.log_softmax(logits,dim=-1)
            tail=ids[0,-n:].to(logp.device)
            value=float(logp[torch.arange(n,device=logp.device),tail].mean())
            info['option_scores'][letter]=value;info['option_token_counts'][letter]=n
            if value>best_score:best,best_score=letter,value
        assert best is not None
        info['letter_fallback']=max(letters,key=letters.get) if letters else ''
        return best,info

    def predict(self,prompt,media_path=None):
        started=time.monotonic();self.calls+=1;self._cached_features=None
        self._bridge_checked=False
        entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None}
        try:
            if not media_path or not os.path.isfile(media_path):raise FileNotFoundError(media_path)
            clean=_strip_eval_instruction(prompt or '').strip();options=_parse_options(clean)
            if not options:raise RuntimeError('The fixed MMBench protocol requires options')
            pixels=self._load_image(media_path)
            with torch.inference_mode():self._cached_features=super()._vision_features(pixels)
            if not bool(torch.isfinite(self._cached_features).all()):raise RuntimeError('Nonfinite image features')
            entry.update(pixel_sha256=fingerprint(pixels),pixel_shape=list(pixels.shape),image_features_sha256=fingerprint(self._cached_features),image_features_shape=list(self._cached_features.shape),source_image_sha256=hashlib.sha256(Path(media_path).read_bytes()).hexdigest())
            answer,info=self._discriminative(clean,pixels,options)
            if not self._bridge_checked:raise RuntimeError('Image-to-language injection was not observed')
            info['visual_embedding_injection_verified']=self._bridge_checked
            entry.update(info,prediction=answer)
            logger.info('Yi-VL family NPU sample %s: %s, option_scores=%s',self.calls,answer,info['option_scores'])
            return answer
        except Exception:
            entry['error']=traceback.format_exc();logger.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started
            self._details[(_strip_eval_instruction(prompt or '').strip(),media_path)]=entry
            with self.trace_path.open('a',encoding='utf8') as handle:handle.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._cached_features=None

    def batch_predict(self,batch):return [self.predict(prompt,path) for prompt,path in batch]

    def get_execution_lineage(self):return self._lineage

    def get_sample_execution(self,prompt,media_path):
        return self._details.get((_strip_eval_instruction(prompt or '').strip(),media_path),{})

    def unload(self):
        self._cached_features=None
        super().unload()
        for i in self.devices:
            with torch.npu.device(i):torch.npu.empty_cache()

    @staticmethod
    def _shards(model_path):
        return [str(Path(model_path)/'pytorch_model.bin')]

    @staticmethod
    def _find_vit_dir(model_path):
        return str(Path(__file__).resolve().parents[3]/'configs/LlavaLlamaForCausalLM/tinyllava_224_metadata')

    def _load_image(self,media_path):
        # Native square/center-crop path, unlike Yi's pad-to-square preprocessing.
        return self._image_processor(images=__import__('PIL.Image',fromlist=['Image']).open(media_path).convert('RGB'),return_tensors='pt')['pixel_values']

    def _chatml_text(self,question):
        prefix="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
        return prefix+self._img_token_repeat+'\n'+question+' ASSISTANT:'

    def _tokenize(self,text):
        # Match the native legacy splice boundary: tokenize the user header with
        # BOS, then replace its one image sentinel with all 256 visual positions.
        before,after=text.split(self._img_token_repeat,1)
        ids=self._tokenizer(before,add_special_tokens=True)['input_ids']+[self._img_token_id]*self._n_patches+self._tokenizer(after,add_special_tokens=False)['input_ids']
        output=torch.tensor([ids],dtype=torch.long)
        if int((output==self._img_token_id).sum())!=self._n_patches:raise RuntimeError('Image placeholder count mismatch')
        return output

    def predict(self,prompt,media_path=None):
        started=time.monotonic();self.calls+=1;self._cached_features=None;self._bridge_checked=False
        entry={'attempt':self.calls,'prompt':prompt,'media_path':media_path,'error':None,'protocol':'native greedy autoregressive generation with the existing Vicuna prompt template'}
        try:
            if not media_path or not os.path.isfile(media_path):raise FileNotFoundError(media_path)
            pixels=self._load_image(media_path)
            with torch.inference_mode():self._cached_features=super()._vision_features(pixels)
            if not bool(torch.isfinite(self._cached_features).all()):raise RuntimeError('Nonfinite image features')
            ids=self._tokenize(self._chatml_text(prompt))
            entry.update(pixel_sha256=fingerprint(pixels),pixel_shape=list(pixels.shape),source_image_sha256=hashlib.sha256(Path(media_path).read_bytes()).hexdigest(),image_features_shape=list(self._cached_features.shape),image_features_sha256=fingerprint(self._cached_features),input_ids=ids[0].tolist())
            # Profile a real native prefill; it does not choose or modify the answer.
            self._forward_logits(ids,pixels,logits_to_keep=1)
            dev=self._model.get_input_embeddings().weight.device
            with torch.inference_mode():
                result=self._model.generate(input_ids=ids.to(dev),attention_mask=torch.ones_like(ids,device=dev),pixel_values=pixels.to(dev),max_new_tokens=self._max_new_tokens,do_sample=False,eos_token_id=2,pad_token_id=0,return_dict_in_generate=True,output_scores=False)
            tokens=result.sequences[0,ids.shape[1]:].cpu().tolist()
            raw=self._tokenizer.decode(tokens,skip_special_tokens=True).strip()
            truncated=len(tokens)>=self._max_new_tokens and (not tokens or tokens[-1]!=2)
            entry.update(raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),truncated=truncated,stop_reason='length' if truncated else 'eos' if tokens and tokens[-1]==2 else 'other',visual_embedding_injection_verified=self._bridge_checked)
            if not self._bridge_checked:raise RuntimeError('Image-to-language injection was not observed')
            logger.info('TinyLLaVA NPU sample %s: generated=%s truncated=%s raw=%r',self.calls,len(tokens),truncated,raw[:180])
            return raw
        except Exception:
            entry['error']=traceback.format_exc();logger.error(entry['error']);raise
        finally:
            entry['elapsed_seconds']=time.monotonic()-started
            self._details[(_strip_eval_instruction(prompt or '').strip(),media_path)]=entry
            with self.trace_path.open('a',encoding='utf8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
            self._cached_features=None

    def get_execution_lineage(self):
        self._lineage.update(protocol='Native autoregressive greedy generation; existing Vicuna template, original 2048-token budget, full outputs preserved',vision_weight_source='All 391 CLIP vision parameters are loaded from this TinyLLaVA checkpoint, not from any external CLIP checkpoint',processor_metadata_source='Existing local CLIP-L metadata; 224 resolution is derived from this checkpoint position embeddings (257 rows) and patch size 14',serialized_rotary_buffer_policy='All 22 original BF16 per-layer inv_freq buffers are verified identical and restored without numeric change into the equivalent shared FP32 rotary buffer',image_token_count=self._n_patches,image_placeholder_token='[PAD]',vision_metadata_dir=self._find_vit_dir(self._model_path))
        return self._lineage

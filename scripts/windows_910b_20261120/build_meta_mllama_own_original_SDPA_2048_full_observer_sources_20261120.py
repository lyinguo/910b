"""Derive a private Meta native port; reuse code only, retain its own protocol."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent
packet=json.loads((V/'R_owned_MetaMllama_existing_full_observer_sources_20261120.json').read_bytes())['source_files']
old=packet['universal_eval/model/adapters/mllama11b_original_native_910b_adapter.py']['complete_text'];s=old
def replace(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b)
replace('Mllama11BOriginal910BAdapter','MetaMllamaOwnOriginal2048')
s=s.replace('Mllama11B_original_910b_','MetaMllama_own_original_SDPA2048_')
s=s.replace('import os, pathlib,','import shutil, os, pathlib,')
s=s.replace('native vision and LM on two NPUs.','native vision and LM on two NPUs, own2048/outer10-inner1/default SDPA.')
s=s.replace('original BF16/eager generation','original BF16/default-SDPA generation')
s=s.replace("return 'Native Transformers / torch_npu original BF16 eager true size-grouped greedy4096'","return 'Native Transformers / torch_npu own Meta original BF16 SDPA outer10-inner1 greedy2048'")
replace("assert p['model_id']=='alpindale/Llama-3.2-11B-Vision-Instruct'", "assert p['model_id']=='meta-llama/Llama-3.2-11B-Vision-Instruct'")
replace("assert sha(p['dataset_path'])==p['dataset_SHA'] and p['group_count']==100", "assert sha(p['dataset_path'])==p['dataset_SHA'] and p['group_count']==200 and p['original_outer10_inner_serial1']")
replace("        cfg._attn_implementation='eager';cfg.text_config._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'", "        assert cfg._attn_implementation is cfg.text_config._attn_implementation is cfg.vision_config._attn_implementation is None")
replace("        assert self.model.forward.__func__ is MllamaForConditionalGeneration.forward", "        assert self.model.forward.__func__ is MllamaForConditionalGeneration.forward\n        assert self.model.config._attn_implementation==self.model.config.text_config._attn_implementation==self.model.config.vision_config._attn_implementation=='sdpa'")
s=s.replace('max_new_tokens=4096,do_sample=False','max_new_tokens=2048,do_sample=False').replace("'original_max_new_tokens':4096","'original_max_new_tokens':2048")
s=s.replace("'protocol':'Mllama11B_original_BF16_eager_true_leftpad_size_grouped_free_generation'", "'protocol':'MetaMllama_own_original_BF16_defaultSDPA_outer10_inner1_2048'")
s=s.replace("'actual_batching':'Original stable original-width//64,original-height//64 groups within each outer10; original full group includes10, no serial fallback after group failure'", "'actual_batching':'Original TransformersVLM.batch_predict calls predict sequentially; own outer10/inner1, no size grouping or borrowed fallback schedule'")
s=s.replace("'original_eager_attention_all_text_and_vision':True", "'original_default_SDPA_all_text_and_vision':True")
s=s.replace("Native full-group eager attention remains intact.","Native original SDPA attention remains intact; physical placement transfers only.")
s=s.replace("Own560/max4tiles/double BOS/leftpad/4096/true size groups/cache/eager unchanged.","Own560/max4tiles/double BOS/leftpad/2048/outer10-inner1/cache/default SDPA unchanged. Historical A100 engine/tensors were not captured; current installed original HF default is proven by meta construction, not claimed as the historical engine.")
replace("        self._reset();self._install_observers()", "        self._reset();self._install_observers();self._install_actual_SDPA_observer()")
replace("        self._vision_calls=0;self._previous_cache=None;self._last_prefix=None;self._projected=None;self._cross_key={};self._cross_value={}", "        self._vision_calls=0;self._previous_cache=None;self._last_prefix=None;self._projected=None;self._cross_key={};self._cross_value={};self._SDPA_context=None;self._SDPA_masks={}")
replace("'original_prefix_NPU_bitwise_unchanged':prefix});now.append((k.detach(),v.detach()))", "'original_prefix_NPU_bitwise_unchanged':prefix,'complete_key':tr(k),'complete_value':tr(v)});now.append((k.detach(),v.detach()))\n                if n==1 or not cross:\n                    self._store('%04d_cache_%02d_key'%(n,i),k if n==1 else k[...,-1:,:]);self._store('%04d_cache_%02d_value'%(n,i),v if n==1 else v[...,-1:,:])")
replace("        def language_pre(module,args,kw):", "        def language_pre(module,args,kw):")
replace("                mask=kw.get('attention_mask');record=None", "                self._SDPA_context=('text',i,kind if False else ('cross' if cross else 'self'))\n                mask=kw.get('attention_mask');record=None")
replace("                mask=kw['attention_mask'];record=tr(mask)","                self._SDPA_context=('vision',name,'vision')\n                mask=kw['attention_mask'];record=tr(mask)")
replace("        self.group_calls+=1;self._reset();self._items=items;home=", "        assert len(items)==1 and shutil.disk_usage(self.evidence).free>=max(3*2**30,1*2**30+2*2**30*10)\n        self.group_calls+=1;self._reset();self._items=items;home=")
s=s.replace('len(self._steps)==generated.shape[1]<=4096','len(self._steps)==generated.shape[1]<=2048').replace("'truncated':count>=4096","'truncated':count>=2048")
tree=ast.parse(s);cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MetaMllamaOwnOriginal2048')
f=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='batch_predict');a=ast.get_source_segment(s,f)
b='''def batch_predict(self,batch):
        assert 1<=len(batch)<=10
        lookup={(x['model_prompt'],x['media_path']):x for x in self.p['input_records']};result=[]
        for prompt,media in batch:
            key=(prompt,str(pathlib.Path(media).resolve()));assert key in lookup
            result.extend(self._generate_group([dict(lookup[key],media_path=media)]))
        return result'''
s=s.replace(a,b).replace("raise RuntimeError('Use original true size-grouped batch API; no serial fallback after failed group')", "return self.batch_predict([(prompt,media_path)])[0]")
observer='''    def _install_actual_SDPA_observer(self):
        # Read-only argument capture around the installed original function.
        # Call every actual native SDPA once with exactly the received arguments.
        self._original_SDPA=torch.nn.functional.scaled_dot_product_attention
        @functools.wraps(self._original_SDPA)
        def observed(query,key,value,*args,**kwargs):
            assert self._step is not None and self._SDPA_context is not None
            section,module,kind=self._SDPA_context;n=self._step['step']
            mask=kwargs.get('attn_mask',args[0] if args else None);causal=kwargs.get('is_causal',args[2] if len(args)>2 else False)
            assert isinstance(causal,bool) and query.dtype==key.dtype==value.dtype==torch.bfloat16
            record=dict(section=section,module=module,kind=kind,query_shape=list(query.shape),key_shape=list(key.shape),value_shape=list(value.shape),dtype=str(query.dtype),device=str(query.device),is_causal=causal,dropout=kwargs.get('dropout_p',args[1] if len(args)>1 else 0.),scale=kwargs.get('scale'),enable_gqa=kwargs.get('enable_gqa',False),mask=None,mask_array=None)
            if mask is not None:
                record['mask']=tr(mask);name='%04d_original_SDPA_%s_mask'%(n,kind)
                if name not in self._arrays:self._store(name,mask)
                else:assert torch.equal(mask.detach().cpu(),self._arrays[name])
                record['mask_array']=name
            self._step.setdefault('actual_native_SDPA_calls',[]).append(record)
            return self._original_SDPA(query,key,value,*args,**kwargs)
        torch.nn.functional.scaled_dot_product_attention=observed
'''
s=s.replace('    def _generate_group(self,items):',observer+'    def _generate_group(self,items):')
s=s.replace('        self.model=None\n',"        self.model=None\n        if hasattr(self,'_original_SDPA'):torch.nn.functional.scaled_dot_product_attention=self._original_SDPA\n")
s=s.replace('self._SDPA_context=(\'text\',i,kind if False else (\'cross\' if cross else \'self\'))',"self._SDPA_context=('text',i,'cross' if cross else 'self')")
ast.parse(s);name='meta_mllama_own_original_BF16_SDPA_2048_native_v1_20261120.py';(V/name).write_text(s,encoding='utf8',newline='\n')
mc=(V/'glm41thinking_own_original_VLLM_BF16_8192_20261120_MC.py').read_text().replace('OwnGLM41Thinking','OwnMetaMllama').replace('glm41thinking_original_VLLM8192','meta_mllama_original_HF2048');ast.parse(mc);(V/'meta_mllama_own_original_HF_2048_MC_20261120.py').write_text(mc,encoding='utf8',newline='\n')
review=dict(passed=True,original_architecture_code_SHA256=packet['universal_eval/model/adapters/mllama11b_original_native_910b_adapter.py']['SHA256'],new_adapter_SHA256=hashlib.sha256(s.encode()).hexdigest(),own_model='meta-llama/Llama-3.2-11B-Vision-Instruct',budget=2048,outer=10,inner=1,attention='own original installed default SDPA, not copied old eager',code_reuse_only_no_other_checkpoint_inputs_outputs=True,all_readonly_observers_retained=True,additional_full_physical_KV_first_and_append_arrays_allSHA_and_original_SDPA_argument_capture=True,no_native_forward_attention_generate_cache_or_math_replacement=True)
(V/'MetaMllama_native_source_derivation_review_20261120.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review))

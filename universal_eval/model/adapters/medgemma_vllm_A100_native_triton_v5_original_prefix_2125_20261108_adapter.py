"""Original A100 MedGemma vLLM BF16 graph on an Ascend reference backend.

Packed learned operations, original 10-request batching, full BF16 vocabulary,
own double-BOS inputs, native incremental decoding, and one-step asynchronous
finish residency below the original output cap. Logical contiguous KV is used, not CUDA paged acceleration.
The fixed200 input qualification proves zero inter-batch prefix-cache hits;
same-batch full-block reuse is resolved before attention; an unqualified
inter-batch positive reuse is rejected rather than silently recomputed.
"""
import collections
import contextlib
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
import torch_npu
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.medgemma_vllm_native_math_v6_original_prefix_2125_20261108 import (
    MedGemmaNativeMath,SourceWeights,tensor_sha,file_sha)
from universal_eval.model.adapters.medgemma_vllm_A100_io_0251_20261105 import OwnMedGemmaIO

log=logging.getLogger('universal_eval.model.adapters.MedGemma_A100_native_0251')
ROOT=Path('/root/projects/comperror')


class MedGemmaVllmA100Native0251Adapter(IModel):
    def __init__(self,protocol_file,protocol_sha256,npu_devices=None,
                 evidence_dir='workspace/diagnostics/model_runs',
                 numerical_review=None,numerical_review_sha256=None,**kwargs):
        self.protocol_file=Path(protocol_file).resolve()
        assert file_sha(self.protocol_file)==protocol_sha256
        self.p=json.loads(self.protocol_file.read_text())
        assert self.p['model_id']=='unsloth/medgemma-1.5-4b-it'
        self.protocol_sha256=protocol_sha256
        self.devices=list(npu_devices or [0]);assert len(self.devices)==1
        self.device=torch.device('npu:'+str(self.devices[0]))
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=Path(evidence_dir).resolve()/('MedGemma_A100_native_2125_own_triton_split_patch_fa2_prefix_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl'
        self.details={};self.groups=[];self.next_index=0;self.batch_failed=False
        self.profiled=set();self.lineage={};self.group=None;self.step=None
        self.arrays={};self.activation_counts=collections.Counter()
        self.encoder_cache=collections.OrderedDict();self.prefix_hashes=set()
        self.numerical_review=numerical_review;self.numerical_review_sha256=numerical_review_sha256

    @property
    def model_name(self):return self.p['model_id']
    @property
    def model_type(self):return 'Gemma3ForConditionalGeneration'
    @property
    def inference_framework(self):return 'Ascend torch_npu original A100 vLLM BF16 packed reference, native logical KV and original batch protocol'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')

    def load(self,model_path,**kwargs):
        p=self.p;self.weight_path=Path(model_path).resolve()
        assert str(self.weight_path)==p['weight_path']
        assert p['original_request_batch_size']==10 and p['original_max_tokens']==2048
        assert p['original_max_model_len']==16384 and p['source_dtype']=='torch.bfloat16'
        assert p['fullchat_text200_qualified'] and p['genuine_A100_real3_and_B10_controls_qualified']
        assert p['duplicate_image_native_B10_qualified'] and p['async_finish_lookahead_steps']==1
        assert p['original_primary_eos']==106 and p['original_stop_token_ids']==[1]
        assert p['fixed200_no_interbatch_computed_prefix_hits_proved']
        assert p['original_prefix_caching_enabled'] and p['original_chunked_prefill_enabled']
        assert p['original_cache_block_size']==16 and p['original_encoder_budget']==8192
        assert len(p['input_records'])==200 and p['original_encoder_tokens_per_image']==256
        assert file_sha(Path(p['dataset']))==p['dataset_SHA256']
        for name,digest in p['required_code_sha256'].items():
            path=Path(name);assert not path.is_absolute() and '..' not in path.parts
            assert file_sha(ROOT/path)==digest,name
        # Execution inputs intentionally omit all reference answers and old/new
        # A100 outputs. Only genuine input/weight/protocol guards are available.
        assert all('reference' not in row and 'raw_output' not in row and 'original_A100_raw' not in row for row in p['input_records'])
        assert torch.npu.is_available();torch.npu.set_device(self.device);torch.set_num_threads(4)
        for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
        self.weights=SourceWeights(self.weight_path,p,self.device,
            lambda row:log.info('Own MedGemma original BF16 full source/cast/NPU SHA: %s',row))
        self.io=OwnMedGemmaIO(self.weight_path,p['own_resource_SHA256'],p['full_image_sequence'])
        config=json.loads((self.weight_path/'config.json').read_text())
        op = p['own_original_Triton_rounding_operator']
        assert file_sha(Path(op['path'])) == op['SHA256']
        assert json.loads(Path(op['path']).read_bytes())['passed']
        patch_control=p['actual_A100_own_B10_BF16_patch_bias_control']
        assert file_sha(Path(patch_control['path']))==patch_control['SHA256']
        assert json.loads(Path(patch_control['path']).read_bytes())['split_BF16_RMSE_improved']
        fa2_control=p['actual_A100_own_visual_FA2_rounding_control']
        assert file_sha(Path(fa2_control['vision_review']['path']))==fa2_control['vision_review']['SHA256']
        assert file_sha(Path(fa2_control['instruction_witness']['path']))==fa2_control['instruction_witness']['SHA256']
        prefix_control=p['actual_A100_original_text_prefix_control']
        assert file_sha(Path(prefix_control['operator_review']['path']))==prefix_control['operator_review']['SHA256']
        self.model=MedGemmaNativeMath(self.weights,config,self._observe)
        self.encoder_capacity=p['original_encoder_budget']//p['original_encoder_tokens_per_image']
        self.rows=p['input_records'];assert [r['index'] for r in self.rows]==list(range(200))
        assert max(len(r['ids']) for r in self.rows)+2048<16384
        stop_control=p['actual_A100_original_async_limit_control']
        assert file_sha(Path(stop_control['path']))==stop_control['SHA256']
        assert json.loads(Path(stop_control['path']).read_bytes())['passed']
        (self.evidence/'weight_loading.json').write_text(json.dumps(self.weights.records,indent=2))
        (self.evidence/'physical_packed_weights.json').write_text(json.dumps(self.weights.physical,indent=2))
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for name in p['required_code_sha256']:
            source=ROOT/name;target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        (snapshot/'execution_protocol.json').write_bytes(self.protocol_file.read_bytes())
        buffers={name:{'shape':list(rope.cache.shape),'dtype':str(rope.cache.dtype),
            'device':str(rope.cache.device),'SHA256':tensor_sha(rope.cache)}
            for name,rope in self.model.rope.items()}
        self.lineage={'hardware':'Ascend910B','devices':self.devices,'runtime_dtype':'BF16',
            'own_original_Triton_rounding_operator':op,
            'original_attention_reduction':'tile_local_exp_to_BF16_before_final_division',
            'original_attention_backend_disclosure':'Actual A100 Triton source and operator verified. Logical NPU reference uses original causal/window pruning and tile-local BF16 exponent rounding. FP32 summation association and device kernels differ.',
            'protocol':'Actual A100 MedGemma vLLM packed BF16, original10 requests, max2048, fullchat doubleBOS',
            'protocol_file':str(self.protocol_file),'protocol_SHA256':self.protocol_sha256,
            'weight_tensor_count':883,'all_expected_parameters_loaded':True,
            'weight_loading_evidence':str(self.evidence/'weight_loading.json'),
            'physical_packed_weights':str(self.evidence/'physical_packed_weights.json'),
            'physical_packed_parameter_count':673,'source_BF16_FP16_cast_not_performed':True,
            'own_tied_262208_head_embedding_same_NPU_tensor':True,
            'actual_head_embedding_data_ptr':self.weights['language_model.model.embed_tokens.weight'].data_ptr(),
            'original_padded_head_rows_retained':63,'original_full_head_dtype':'BF16',
            'original_image_tokens':256,'original_low_HD_or_pan_substitution':False,
            'original_max_new_tokens':2048,'original_outer_request_batch_size':10,
            'original_primary_eos_token_id':106,'original_additional_stop_token_ids':[1],
            'original_min_tokens':0,'original_temperature':0.0,'original_repetition_penalty':1.0,
            'original_one_async_unused_forward_after_finish_below_cap':True,
            'original_no_async_unused_forward_at_output_cap':True,
            'actual_A100_original_async_limit_control':stop_control,
            'actual_A100_own_B10_BF16_patch_bias_control':patch_control,
            'original_vision_patch_bias_rounds_convolution_to_BF16_before_addition':True,
            'actual_A100_own_visual_FA2_rounding_control':fa2_control,
            'actual_A100_original_text_prefix_control':prefix_control,
            'original_Triton_text_visibility':'(causal AND local-window where applicable) OR 256 image-token bidirectional positions',
            'original_Triton_mm_prefix_full_KV_tile_span':True,
            'original_vision_attention_reference':{
                'key_tile':64,'key_order':'descending','QKV_and_P':'BF16',
                'score_accumulator_denominator_and_PV_accumulator':'FP32','mask':'noncausal',
                'dropout':0.0,'shape':[16,4096,72],'query_probe_positions':[0,32,2048,4095],
                'physical_kernel':'native_NPU_operator_reference_for_observed_A100_FA2_boundaries',
            },
            'same_batch_real_prefix_block_sharing_before_attention':True,
            'original_stop_before_length_check':True,'native_incremental_decode_prompt_primed':True,
            'original_full_RoPE_buffers':buffers,'text_embedding_scale':{'dtype':str(self.model.scale.dtype),'SHA256':tensor_sha(self.model.scale)},
            'actual_attention':'Native NPU SDPA logical causal/sliding1024, GQA8/4; original CUDA Triton backend is not claimed on NPU',
            'cache_disclosure':'Real per-request NPU KV retained across every forward. Original prefix cache is enabled with real same-wave full-block KV reuse. Fixed200 has no inter-batch reusable prefix because its first16-token block binds the image identity, with duplicate pairs only concurrent within original batches. A real computed-prefix index is checked; positive reuse outside this qualified input scope raises before inference, never silently recomputes. No CUDA paged allocation acceleration claim.',
            'original_encoder_cache_capacity_images':self.encoder_capacity,
            'per_sample_evidence':str(self.trace),'per_group_evidence':str(self.evidence/'groups'),
            'profiler_traces':{'vision':str(self.evidence/'vision_npu_trace.json'),'language':str(self.evidence/'language_npu_trace.json')},
            'profiler_capture_scope':'Only first actual visual and first actual decoder prefill, all sample forward/token/cache/activation evidence retained separately',
            'precision_disclosure':'Own883 learned BF16 source tensors, original actual A100 native autoBF16 execution, packed QKV/gateup, original BF16 GELU intermediate then multiply, FP32(weight+1) and unrounded FP32 fused-residual RMSNorm, own full NPU FP32-formula BF16 RoPE caches, text-only BF16sqrt, complete262208 tied BF16 output head retained. No low-bit unpacking/acceleration or CPU official model answers. Genuine CANN/CPU/A100 nonzero differences are retained; no ERF, dtype, mask, budget, option constraint or output filtering used to force agreement.',
            'groups':self.groups}
        if self.numerical_review is not None:
            assert file_sha(Path(self.numerical_review))==self.numerical_review_sha256
            self.lineage.update(numerical_review=str(self.numerical_review),numerical_review_SHA256=self.numerical_review_sha256)

    def _tensor_record(self,value,capture_key=None,full_hash=False):
        assert value.device==self.device
        rec={'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
        if capture_key is not None:
            array=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
            self.arrays[capture_key]=array;rec.update(array=capture_key,SHA256=hashlib.sha256(array.tobytes()).hexdigest())
        elif full_hash:rec['SHA256']=tensor_sha(value)
        return rec

    def _observe(self,name,value,metadata):
        assert self.group is not None and value.device==self.device
        if name.startswith(('vision_activation_','language_activation_')):
            assert value.dtype==torch.bfloat16
            self.activation_counts[name]+=1
            record=self._tensor_record(value,full_hash=self.step is None or self.step['ordinal']==0)
            record.update(module=name,call=self.activation_counts[name],metadata=metadata)
            target=self.group['vision_activations'] if name.startswith('vision_') else self.step['activations']
            target.append(record);return
        if name in ['vision_actual_pixels','vision_encoder_output','visual_projector_output'] or name.startswith('vision_fa2_'):
            self.group['visual_tensors'][name]=self._tensor_record(value,'visual_'+name);return
        assert self.step is not None
        if name.startswith('KV_delta_'):
            rid=str(metadata['request_id']);key='s%04d_r%s_%s'%(self.step['ordinal'],rid,name)
            rec=self._tensor_record(value,key);rec.update(name=name,**metadata)
            self.step['KV_deltas'].append(rec);return
        if name.startswith('attention_visibility_'):
            assert value.dtype==torch.bool
            rid=str(metadata['request_id']);window=metadata['layer_type']
            key='s%04d_r%s_mask_%s'%(self.step['ordinal'],rid,window)
            rec=self._tensor_record(value,None,full_hash=True)
            if key not in self.arrays:rec=self._tensor_record(value,key)
            else:assert rec['SHA256']==hashlib.sha256(self.arrays[key].tobytes()).hexdigest();rec['array']=key
            rec.update(layer=int(name.rsplit('_',1)[1]),**metadata)
            self.step['attention'].append(rec);return
        if name=='text_input_embeddings':
            rec=self._tensor_record(value,'s%04d_inputs_embeds'%self.step['ordinal'] if self.step['ordinal']==0 else None)
            rec.update(metadata);self.step['input_embeddings']=rec;return
        if name=='full_vocabulary_logits':
            rec=self._tensor_record(value,'s%04d_full262208_BF16_logits'%self.step['ordinal']);rec.update(metadata)
            self.step['full_logits']=rec;return
        raise RuntimeError('Unreviewed native observation: '+name)

    def _profile(self,which,function,*args):
        if which in self.profiled:return function(*args)
        with contextlib.chdir(self.evidence):
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profiler:
                result=function(*args);torch.npu.synchronize(self.device)
            profiler.export_chrome_trace(str(self.evidence/(which+'_npu_trace.json')))
        self.profiled.add(which);return result

    def _prefix_block_hashes(self,tokens,media_sha,image_start):
        parent=b'MedGemma_original_prefix_equivalence_v1';out=[]
        for start in range(0,len(tokens)//16*16,16):
            payload=struct.pack('<16I',*tokens[start:start+16])
            if start<image_start+258 and start+16>image_start-2:payload+=media_sha.encode()
            parent=hashlib.sha256(parent+payload).digest();out.append(parent.hex())
        return out

    @torch.inference_mode()
    def batch_predict(self,batch):
        if self.batch_failed:raise RuntimeError('An original batch failed; silent singleton fallback is forbidden')
        assert 0<len(batch)<=10 and self.next_index+len(batch)<=200
        start=time.monotonic();ordinal=len(self.groups);folder=self.evidence/'groups'/('%03d'%ordinal);folder.mkdir()
        self.group={'ordinal':ordinal,'start_index':self.next_index,'request_count':len(batch),
            'vision_activations':[],'visual_tensors':{},'steps':[],'requests':[],
            'error':None,'native_batch_scope':'original10 or small compatibility remainder'}
        self.arrays={};self.activation_counts=collections.Counter();prepared=[];states={}
        try:
            unique=[];pixels=[];features={};original_inputs=[];scheduled_prefixes={}
            for offset,(prompt,media) in enumerate(batch):
                index=self.next_index+offset;expected=self.rows[index]
                assert prompt==expected['model_prompt'] and str(Path(media).resolve())==expected['R_media_path']
                rendered,ids,image_start=self.io.render(prompt)
                assert ids==expected['ids'] and hashlib.sha256(rendered.encode()).hexdigest()==expected['rendered_SHA256']
                value,media_sha,pixel_sha,canonical_media=self.io.pixels(media)
                assert media_sha==expected['media_SHA256'] and pixel_sha==expected['pixels_SHA256']
                blocks=self._prefix_block_hashes(ids,media_sha,image_start)
                matches=0
                for key in blocks:
                    if key not in self.prefix_hashes:break
                    matches+=1
                if matches:raise RuntimeError('Positive computed prefix reuse outside qualified fixed200 scope; no silent cache disable/recompute')
                rid=str(index)
                same_batch_count=0;same_batch_source=None
                for key in blocks[:(len(ids)-1)//16]:
                    if key not in scheduled_prefixes:break
                    same_batch_count+=16;same_batch_source=scheduled_prefixes[key]
                row={'index':index,'request_id':rid,'prompt':prompt,'media_path':str(media),
                    'canonical_media_path':canonical_media,'source_image_SHA256':media_sha,
                    'input_ids':ids,'image_soft_start_position':image_start,'rendered':rendered,
                    'CPU_pixel_shape':list(value.shape),'CPU_pixel_SHA256':pixel_sha,
                    'original_prefix_cache_enabled':True,'computed_prefix_hit_tokens':0,'prefix_source_request_id':None}
                row.update(computed_prefix_hit_tokens=same_batch_count,prefix_source_request_id=same_batch_source)
                for key in blocks:scheduled_prefixes.setdefault(key,rid)
                prepared.append(row);self.group['requests'].append(row)
                states[rid]={'tokens':[],'ended':False,'finish_reason':None,'stop_reason':None,'row':row}
                original_inputs.append(len(ids))
                if media_sha in self.encoder_cache:
                    features[media_sha]=self.encoder_cache.pop(media_sha);self.encoder_cache[media_sha]=features[media_sha]
                elif media_sha not in unique:unique.append(media_sha);pixels.append(value)
            assert sum(original_inputs)<=8192 and len(unique)*256<=8192
            self.group['unique_encoder_input_count']=len(unique)
            self.group['original_encoder_cached_keys']=list(features)
            self.group['new_encoder_keys_in_original_request_order']=list(unique)
            if unique:
                pixel_batch=torch.cat(pixels).to(self.device)
                projected=self._profile('vision',self.model.vision,pixel_batch)
                assert projected.shape==(len(unique),256,2560)
                for key,feature in zip(unique,projected):
                    features[key]=feature;self.encoder_cache[key]=feature
                while len(self.encoder_cache)>self.encoder_capacity:self.encoder_cache.popitem(last=False)
            active=[r['request_id'] for r in prepared];is_prefill=True;step_ordinal=0
            trace_path=folder/'steps.jsonl'
            with trace_path.open('w',encoding='utf8') as trace:
                while active:
                    requests=[]
                    for rid in active:
                        st=states[rid];r=st['row']
                        if is_prefill:
                            n=r['computed_prefix_hit_tokens'];ids=r['input_ids'][n:];positions=list(range(n,len(r['input_ids'])))
                            visual=features[r['source_image_SHA256']]
                        else:
                            ids=[st['tokens'][-1]];positions=[len(r['input_ids'])+len(st['tokens'])-1];visual=None
                        requests.append({'id':rid,'token_ids':torch.tensor(ids,dtype=torch.int64,device=self.device),
                            'positions':torch.tensor(positions,dtype=torch.int64,device=self.device),
                            'prefix_token_count':r['computed_prefix_hit_tokens'] if is_prefill else 0,
                            'prefix_source_request_id':r['prefix_source_request_id'] if is_prefill else None,
                            'visuals':visual,'image_start_position':r['image_soft_start_position'] if is_prefill else None,
                            'original_image_prefix_range':(r['image_soft_start_position'],r['image_soft_start_position']+255)})
                    requests.sort(key=lambda r:(r['token_ids'].numel()!=1,r['token_ids'].numel()))
                    self.step={'ordinal':step_ordinal,'request_ids':[r['id'] for r in requests],
                        'query_lengths':[r['token_ids'].numel() for r in requests],
                        'input_ids':[r['token_ids'].cpu().tolist() for r in requests],
                        'positions':[r['positions'].cpu().tolist() for r in requests],
                        'KV_deltas':[],'attention':[],'activations':[],'sample_actions':[]}
                    logits=self._profile('language',self.model.text_step,requests)
                    assert logits.dtype==torch.bfloat16 and logits.shape==(len(requests),262208)
                    assert len(self.step['activations'])==34 and len(self.step['KV_deltas'])==68*len(requests) and len(self.step['attention'])==34*len(requests)
                    argmax=logits.argmax(-1).cpu().tolist();released=[]
                    for request,token in zip(requests,argmax):
                        rid=request['id'];st=states[rid]
                        if st['ended']:
                            self.step['sample_actions'].append({'request_id':rid,'kind':'original_unused_async_lookahead',
                                'actual_argmax':token,'reported_generation_count':len(st['tokens'])})
                            released.append(rid);continue
                        st['tokens'].append(token)
                        if token==106:st.update(ended=True,finish_reason='stop',stop_reason=None)
                        elif token==1:st.update(ended=True,finish_reason='stop',stop_reason=1)
                        elif len(st['tokens'])>=2048 or len(st['row']['input_ids'])+len(st['tokens'])>=16384:
                            st.update(ended=True,finish_reason='length',stop_reason=None)
                        at_output_cap=len(st['tokens'])==2048
                        if st['ended'] and at_output_cap:
                            released.append(rid)
                        self.step['sample_actions'].append({'request_id':rid,'kind':'reported_sample',
                            'original_async_lookahead_suppressed_at_cap':at_output_cap,
                            'token_id':token,'reported_generation_count':len(st['tokens']),
                            'finished':st['ended'],'finish_reason':st['finish_reason'],'stop_reason':st['stop_reason']})
                    for rid in released:
                        records=[]
                        for layer,cache in enumerate(self.model.cache):
                            key,value=cache[rid]
                            token_count=len(states[rid]['tokens'])
                            expected_context=len(states[rid]['row']['input_ids'])+token_count-int(token_count==2048)
                            assert key.shape==value.shape and key.shape[0]==expected_context
                            records.append({'layer':layer,'key_shape':list(key.shape),'value_shape':list(value.shape),
                                'key_SHA256':tensor_sha(key),'value_SHA256':tensor_sha(value),
                                'dtype':str(key.dtype),'device':str(key.device)})
                        states[rid]['final_KV']=records
                    trace.write(json.dumps(self.step,ensure_ascii=False)+'\n');trace.flush()
                    self.group['steps'].append({'ordinal':step_ordinal,'request_ids':list(self.step['request_ids']),
                        'query_lengths':list(self.step['query_lengths']),'argmax':argmax})
                    if released:
                        self.model.release_requests(released)
                        for rid in released:active.remove(rid)
                    step_ordinal+=1;is_prefill=False;self.step=None
                    if step_ordinal%16==0:log.info('Actual native MedGemma group=%s forward=%s active=%s completed=%s/%s',ordinal,step_ordinal,len(active),len(batch)-len(active),len(batch))
            assert not any(self.model.cache)
            archive=folder/'full_vocab_KV_inputs_visual.npz';np.savez_compressed(archive,**self.arrays)
            archive_sha=file_sha(archive);trace_sha=file_sha(trace_path)
            outputs=[]
            for r in prepared:
                st=states[r['request_id']];tokens=st['tokens'];raw,warnings=self.io.decode(r['input_ids'],tokens,st['finish_reason'])
                entry=dict(r,raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),
                    original_max_new_tokens=2048,finish_reason=st['finish_reason'],stop_reason=st['stop_reason'],
                    truncated=st['finish_reason']=='length',empty=not bool(raw),error=None,
                    original_async_unused_forward_retained=len(tokens)<2048,
                    actual_request_forward_count=len(tokens)+int(len(tokens)<2048),
                    final_all34_KV=st['final_KV'],native_detokenization_warnings=warnings,
                    group=ordinal,group_archive=str(archive),group_archive_SHA256=archive_sha,
                    group_steps=str(trace_path),group_steps_SHA256=trace_sha)
                self.details[(r['prompt'].strip(),r['media_path'])]=entry
                self.details[(r['prompt'].strip(),r['canonical_media_path'])]=entry
                with self.trace.open('a',encoding='utf8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                outputs.append(raw)
                self.prefix_hashes.update(self._prefix_block_hashes(r['input_ids']+tokens,r['source_image_SHA256'],r['image_soft_start_position']))
            self.group.update(activation_call_counts=dict(self.activation_counts),actual_model_forward_count=step_ordinal,
                original_reported_request_count=len(batch),archive=str(archive),archive_SHA256=archive_sha,
                trace=str(trace_path),trace_SHA256=trace_sha,elapsed_seconds=time.monotonic()-start,
                peak_NPU_allocated_bytes=torch.npu.max_memory_allocated(self.device))
            summary=folder/'group.json';summary.write_text(json.dumps(self.group,ensure_ascii=False,indent=2))
            self.groups.append({'path':str(summary),'SHA256':file_sha(summary),'archive':str(archive),
                'archive_SHA256':archive_sha,'trace':str(trace_path),'trace_SHA256':trace_sha,
                'request_count':len(batch),'unique_encoder_input_count':len(unique),'actual_forward_count':step_ordinal})
            self.next_index+=len(batch);self.weights.assert_resources_unchanged();self.group=None;self.arrays={}
            return outputs
        except BaseException:
            self.batch_failed=True;error=traceback.format_exc();self.group['error']=error
            (folder/'failed_group.json').write_text(json.dumps(self.group,ensure_ascii=False,indent=2))
            for r in prepared:
                st=states[r['request_id']];entry=dict(r,error=error,generation_token_ids=st['tokens'],raw_output='',
                    generation_count=len(st['tokens']),truncated=False,empty=True,group=ordinal)
                self.details[(r['prompt'].strip(),r['media_path'])]=entry
                with self.trace.open('a',encoding='utf8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
            log.error(error);raise

    def predict(self,prompt,media_path=None):
        if self.batch_failed:raise RuntimeError('Original batch failure retained; no singleton fallback')
        return self.batch_predict([(prompt,media_path)])[0]
    def get_sample_execution(self,prompt,media_path):
        return self.details.get((prompt.strip(),str(media_path)),{'error':'Missing native MedGemma execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        self.model=None;self.weights=None;self.encoder_cache.clear();torch.npu.empty_cache()

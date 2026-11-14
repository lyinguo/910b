"""Original GraniteDocling A100 vLLM graph/protocol on an Ascend logical reference.

Original10 requests,8192 token/encoder budgets,16-token KV prefix equivalence,
full100352 BF16 head, native source packing and original2048 generation budget.
This is not the installed vLLM Ascend128-block backend or paged acceleration.
"""
import queue,threading,shutil
import collections,contextlib,datetime,fcntl,hashlib,json,logging,os,struct,time,traceback
from pathlib import Path
import numpy as np
import torch,torch_npu
from universal_eval.model.interface import IModel
from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2
from universal_eval.model.adapters.granite_docling_own_vllm_math_v2_config_2109_20261111 import GraniteDoclingNativeMath,SourceWeights,tensor_sha,file_sha
from universal_eval.model.adapters.granite_docling_own_vllm_io_v1_2109_20261111 import OwnGraniteDoclingIO
from universal_eval.model.adapters.granite_docling_own_vllm_scheduler_v2_async_2109_20261111 import OriginalGraniteDoclingScheduler
log=logging.getLogger('universal_eval.model.adapters.GraniteDocling_original_A100_2109')
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'

class GraniteDoclingOriginalA1002109Adapter(IModel):
    def __init__(self,protocol_file,protocol_sha256,npu_devices=None,evidence_dir='workspace/diagnostics/model_runs',numerical_review=None,numerical_review_sha256=None,**kwargs):
        self.protocol_file=Path(protocol_file).resolve();assert file_sha(self.protocol_file)==protocol_sha256
        self.p=json.loads(self.protocol_file.read_text());assert self.p['model_id']=='ibm-granite/granite-docling-258M-mlx-hf'
        self.protocol_sha256=protocol_sha256;self.devices=list(npu_devices or [1]);assert len(self.devices)==1
        self.device=torch.device('npu:'+str(self.devices[0]));self.details={};self.groups=[];self.next_index=0;self.batch_failed=False
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=Path(evidence_dir).resolve()/('GraniteDocling_A100_original_2109_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl';self.profiled=set();self.group=None;self.step=None;self.arrays={};self.activation_counts=collections.Counter();self.vision_ordinal=None
        self.features={};self.self_lock=None;self.numerical_review=numerical_review;self.numerical_review_sha256=numerical_review_sha256
        self.lineage={};self.model=None;self.weights=None
    @property
    def model_name(self):return self.p['model_id']
    @property
    def model_type(self):return 'Idefics3ForConditionalGeneration'
    @property
    def inference_framework(self):return 'Ascend torch_npu reference of original A100 vLLM graph and logical16 KV/8192 scheduler; no paged acceleration claim'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads forbidden')
    def _ensure_runner_lock(self):
        fd=(D/'runner.lock').open('a+')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);self.self_lock=fd;return
        except BlockingIOError:fd.close()
        st=(D/'runner.lock').stat();owners=[]
        for line in Path('/proc/locks').read_text().splitlines():
            parts=line.split()
            if len(parts)<6 or parts[1]!='FLOCK':continue
            dev=parts[5].split(':')
            if len(dev)==3 and int(dev[0],16)==os.major(st.st_dev) and int(dev[1],16)==os.minor(st.st_dev) and int(dev[2])==st.st_ino:owners.append(int(parts[4]))
        ancestors=[];pid=os.getppid()
        while pid>1:
            ancestors.append(pid);p=Path('/proc')/str(pid)
            if not p.exists():break
            pid=int(next(x.split(':',1)[1].strip() for x in (p/'status').read_text().splitlines() if x.startswith('PPid:')))
        assert len(owners)==1 and owners[0] in ancestors,'Unique real runner FLOCK must belong to this evaluation ancestor'
        argv=(Path('/proc')/str(owners[0])/'cmdline').read_bytes().split(b'\0');args=[x.decode() for x in argv if x]
        assert Path(args[0]).name.startswith('python') and str(R/'scripts/A100_alignment_evidence_runner_20261103.py') in args,'Old main handoff/restart is not authorized'
    def load(self,model_path,**kwargs):
        self._ensure_runner_lock();p=self.p;self.weight_path=Path(model_path).resolve();assert str(self.weight_path)==p['weight_path']
        assert p['original_max_tokens']==2048 and p['original_request_batch_size']==10 and p['original_max_model_len']==8192
        assert p['original_cache_block_size']==16 and p['original_encoder_budget']==8192
        assert p['original_primary_eos']==100257 and p['original_stop_token_ids']==[]
        assert p['all200_input_pixels_masks_fullchat_tokens_qualified'] and p['original_scheduler_matches_genuine_A100_controls']
        assert p['fixed200_no_interbatch_computed_prefix_hits_proved'] and len(p['input_records'])==200
        assert file_sha(p['dataset'])==p['dataset_SHA256']
        for name,digest in p['required_code_sha256'].items():
            path=Path(name);assert not path.is_absolute() and '..' not in path.parts;assert file_sha(R/path)==digest,name
        assert all(not any(k in row for k in ['reference','raw_output','original_A100_raw','output_token_ids']) for row in p['input_records'])
        import subprocess,re
        smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
        assert re.search(r'No running processes found in NPU\s+'+str(self.devices[0])+r'\b',smi),'Never share an occupied NPU'
        assert torch.npu.is_available();torch.npu.set_device(self.device);torch.set_num_threads(4)
        for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
        self.weights=SourceWeights(self.weight_path,p,self.device,lambda row:log.info('GraniteDocling own full source/cast/NPU SHA %s',row))
        self.io=OwnGraniteDoclingIO(self.weight_path,p['own_resource_SHA256']);self.model=GraniteDoclingNativeMath(self.weights,p['actual_AA_effective_config'],self._observe)
        self.scheduler=OriginalGraniteDoclingScheduler();self.rows=p['input_records'];assert [x['index'] for x in self.rows]==list(range(200))
        (self.evidence/'weight_loading.json').write_text(json.dumps(self.weights.records,indent=2));(self.evidence/'physical_packed_weights.json').write_text(json.dumps(self.weights.physical,indent=2));(self.evidence/'completed_groups.json').write_text('[]\n')
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for name in p['required_code_sha256']:
            dst=snapshot/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((R/name).read_bytes())
        (snapshot/'execution_protocol.json').write_bytes(self.protocol_file.read_bytes())
        rope=self.evidence/'actual_NPU_full_original_RoPE.npz';np.savez_compressed(rope,cache=self.model.rope.cache.cpu().contiguous().reshape(-1).view(torch.uint8).numpy(),vision_positions=self.model.vision_positions.cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
        self.lineage={'hardware':'Ascend910B','devices':self.devices,'runtime_dtype':'BF16','protocol':'Actual A100 GraniteDocling vLLM graph reference with original10 request/8192 token-encoder/16-block logical cache/2048 budget',
            'protocol_file':str(self.protocol_file),'protocol_SHA256':self.protocol_sha256,'weight_tensor_count':471,'all_expected_parameters_loaded':True,'weight_loading_evidence':str(self.evidence/'weight_loading.json'),'physical_packed_weights':str(self.evidence/'physical_packed_weights.json'),'physical_packed_parameter_count':333,
            'own_untied_head_embedding_preserved':True,'head_materialization_disclosure':'Own R embedding bytes equal original AA separate head. A new separate NPU head is copied only from own R embedding; no A100 or another model tensor is used.','original_full_vocabulary':100352,'original_primary_eos_token_id':100257,'original_additional_stop_token_ids':[],'original_max_new_tokens':2048,'original_outer_request_batch_size':10,'original_min_tokens':0,'original_temperature':0.0,
            'original_encoder_compute_budget':8192,'original_encoder_cache_budget':8192,'original_prefix_cache_block_size':16,'original_prefix_caching_enabled':True,'original_chunked_prefill_enabled':True,'original_async_request_submission':True,'own_actual_arrival_events_in_step_schedule':True,'no_A100_request_arrival_or_output_plan_used':True,'cache_disclosure':'Real original FCFS chunking and async EOS residency, genuine frozen A100 EncoderCacheManager,16-token logical full-block prefix sharing and retained request KV. Physical contiguous allocation is an Ascend reference, not CUDA paged-cache acceleration. No unqualified inter-batch KV hit is silently recomputed.',
            'original_PIL10_4_horizontal_then_vertical_LANCZOS':True,'own_original_max17_full512_tiles_64_context_tokens_each':True,'original_full_RoPE':{'shape':[8192,64],'dtype':'torch.bfloat16','SHA256':tensor_sha(self.model.rope.cache),'path':str(rope),'archive_SHA256':file_sha(rope),'computed_on_NPU_not_copied_from_A100':True},
            'per_sample_evidence':str(self.trace),'completed_groups_manifest':str(self.evidence/'completed_groups.json'),'completed_groups_manifest_scope':'Actual completed original scheduled groups, full SHA checked independently after inference',
            'profiler_traces':{'vision':str(self.evidence/'vision_npu_trace.json'),'language':str(self.evidence/'language_npu_trace.json')},'profiler_capture_scope':'First actual vision wave and first actual packed decoder prefill only; all sample full-logit/KV/position/mask/activation evidence retained separately',
            'precision_disclosure':'Own471 original BF16 learned tensors, packed333 physical allocations, independent untied100352 BF16 head. Original vllm_c standalone F32 norm/weight and fused BF16-rounded residual before F32 variance, BF16 SiLU intermediate then multiply, no embedding sqrt, NPU-computed original8192 RoPE table. No lower-bit acceleration or CPU answers. Genuine CANN and A100/CPU nonzero differences remain; no ERF/dtype/mask/budget/option restriction/output filtering switch is used to force agreement.'}
        if self.numerical_review is not None:
            assert file_sha(self.numerical_review)==self.numerical_review_sha256;self.lineage.update(numerical_review=self.numerical_review,numerical_review_SHA256=self.numerical_review_sha256)
    def _tensor_record(self,value,key=None,full_hash=False):
        assert value.device==self.device;record={'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
        if key is not None:
            a=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy();self.arrays[key]=a;record.update(array=key,SHA256=hashlib.sha256(a.tobytes()).hexdigest())
        elif full_hash:record['SHA256']=tensor_sha(value)
        return record
    def _observe(self,name,value,metadata):
        assert self.group is not None and value.device==self.device
        if name.startswith(('vision_activation_','language_activation_')):
            self.activation_counts[name]+=1;record=self._tensor_record(value,full_hash=self.step is None or self.step['ordinal']==0)
            record.update(module=name,call=self.activation_counts[name],metadata=metadata)
            if name.startswith('vision_'):record['encoder_wave']=self.vision_ordinal;self.group['vision_activations'].append(record)
            else:self.step['activations'].append(record)
            return
        if name in ['vision_actual_pixels','vision_encoder_output','visual_projector_output']:
            key='v%04d_%s'%(self.vision_ordinal,name);self.group['visual_tensors'].append(dict(name=name,encoder_wave=self.vision_ordinal,**self._tensor_record(value,key)));return
        assert self.step is not None
        if name.startswith('KV_delta_'):
            key='s%04d_r%s_%s'%(self.step['ordinal'],metadata['request_id'],name);self.step['KV_deltas'].append(dict(name=name,**self._tensor_record(value,key),**metadata));return
        if name.startswith('attention_visibility_'):
            key='s%04d_r%s_mask'%(self.step['ordinal'],metadata['request_id']);rec=self._tensor_record(value,None,full_hash=True)
            if key not in self.arrays:rec=self._tensor_record(value,key)
            else:assert rec['SHA256']==hashlib.sha256(self.arrays[key].tobytes()).hexdigest();rec['array']=key
            self.step['attention'].append(dict(layer=int(name.rsplit('_',1)[1]),**rec,**metadata));return
        if name=='text_input_embeddings':self.step['input_embeddings']=dict(**self._tensor_record(value,'s%04d_inputs_embeds'%self.step['ordinal']),**metadata);return
        if name=='full_vocabulary_logits':self.step['full_logits']=dict(**self._tensor_record(value,'s%04d_full100352_BF16_logits'%self.step['ordinal']),**metadata);return
        raise RuntimeError('Unreviewed native observation: '+name)
    def _profile(self,which,function,*args):
        if which in self.profiled:return function(*args)
        with contextlib.chdir(self.evidence):
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as profile:
                result=function(*args);torch.npu.synchronize(self.device)
            profile.export_chrome_trace(str(self.evidence/(which+'_npu_trace.json')))
        self.profiled.add(which);return result
    @torch.inference_mode()
    def batch_predict(self,batch):
        if self.batch_failed:raise RuntimeError('Original group failure retained; silent singleton fallback is forbidden')
        assert 0<len(batch)<=10 and self.next_index+len(batch)<=200
        ordinal=len(self.groups);folder=self.evidence/'groups'/('%03d'%ordinal);folder.mkdir();started=time.monotonic();prepared=[];pixels_by_hash={};self.arrays={};self.activation_counts=collections.Counter()
        self.group={'ordinal':ordinal,'start_index':self.next_index,'request_count':len(batch),'requests':[],'vision_activations':[],'visual_tensors':[],'encoder_waves':[],'steps':[],'error':None};self.step=None
        try:
            ready=queue.Queue();producer_errors=[];producer_done=threading.Event()
            def prepare_frontend():
                try:
                    for offset,(prompt,media) in enumerate(batch):
                        index=self.next_index+offset;expected=self.rows[index];assert prompt==expected['model_prompt'] and str(Path(media).resolve())==expected['R_media_path']
                        value,mask,grid,digest,canonical=self.io.pixels(media);rendered,expanded,ids,soft=self.io.render(prompt,grid)
                        assert ids==expected['ids'] and soft==expected['image_soft_positions'] and grid==expected['grid']
                        assert digest==expected['media_SHA256'] and fingerprint(value)==expected['pixels_SHA256'] and fingerprint(mask)==expected['pixel_mask_SHA256']
                        assert hashlib.sha256(rendered.encode()).hexdigest()==expected['rendered_SHA256'] and hashlib.sha256(expanded.encode()).hexdigest()==expected['expanded_SHA256']
                        row={'index':index,'request_id':str(index),'prompt':prompt,'media_path':str(media),'canonical_media_path':canonical,'source_image_SHA256':digest,'input_ids':ids,'image_soft_positions':soft,'rendered':rendered,'expanded_text':expanded,'CPU_pixel_shape':list(value.shape),'CPU_pixel_SHA256':fingerprint(value),'CPU_pixel_mask_shape':list(mask.shape),'CPU_pixel_mask_SHA256':fingerprint(mask),'grid':grid,'mm_hash':expected['mm_hash'],'placeholder_offset':expected['placeholder_offset'],'placeholder_length':expected['placeholder_length']}
                        prepared.append(row);self.group['requests'].append(row);pixels_by_hash.setdefault(row['mm_hash'],value);ready.put(row)
                except BaseException:producer_errors.append(traceback.format_exc())
                finally:producer_done.set()
            producer=threading.Thread(target=prepare_frontend,name="Granite_original_request_frontend");producer.start()
            assert not self.scheduler.requests and not self.scheduler.running and not self.scheduler.waiting
            self.scheduler.wave=0;self.scheduler.previous=[]
            by_id={};trace_path=folder/'steps.jsonl';step_ordinal=0
            with trace_path.open('w',encoding='utf-8') as stream:
                while True:
                    newly_available=[]
                    if not self.scheduler.running and not self.scheduler.waiting and not by_id:
                        while ready.empty() and not producer_done.is_set():producer_done.wait(.001)
                    while True:
                        try:row=ready.get_nowait()
                        except queue.Empty:break
                        by_id[row['request_id']]=row;newly_available.append(row['request_id'])
                        self.scheduler.add_available_records([self.rows[row['index']]])
                    if producer_errors:raise RuntimeError(producer_errors[0])
                    schedule=self.scheduler.schedule()
                    schedule['newly_available_request_ids']=newly_available

                    for key in schedule['freed_encoder_hashes']:self.features.pop(key,None)
                    if not schedule['chunks']:
                        if producer_done.is_set() and ready.empty():break
                        while ready.empty() and not producer_done.is_set():producer_done.wait(.001)
                        continue
                    new=schedule['new_encoder_request_ids']
                    if new:
                        self.vision_ordinal=len(self.group['encoder_waves']);ordered=[by_id[rid] for rid in new];values=[pixels_by_hash[r['mm_hash']] for r in ordered]
                        projected=self._profile('vision',self.model.vision,torch.cat(values).to(self.device));cursor=0
                        for row,value in zip(ordered,values):
                            count=value.shape[0];self.features[row['mm_hash']]=projected[cursor:cursor+count].reshape(-1,576);cursor+=count
                        assert cursor==projected.shape[0]
                        self.group['encoder_waves'].append({'ordinal':self.vision_ordinal,'model_step':step_ordinal,'request_ids':new,'tile_counts':[v.shape[0] for v in values],'total_tiles':cursor,'total_encoder_embeddings':cursor*64})
                    requests=[]
                    for chunk in schedule['chunks']:
                        rid=chunk['id'];r=self.scheduler.requests[rid];row=by_id[rid];all_ids=r.prompt_ids+r.output_ids;start,end=chunk['start'],chunk['start']+chunk['count'];assert end<=len(all_ids)
                        has_visual=any(start<=pos<end for pos in row['image_soft_positions'])
                        requests.append({'id':rid,'token_ids':torch.tensor(all_ids[start:end],device=self.device,dtype=torch.int64),'positions':torch.arange(start,end,device=self.device,dtype=torch.int64),'visuals':self.features[row['mm_hash']] if has_visual else None,'image_soft_positions':row['image_soft_positions'] if has_visual else None,'prefix_token_count':chunk['prefix_count'],'prefix_source_request_id':chunk['prefix_source']})
                    self.step={'ordinal':step_ordinal,'schedule':schedule,'request_ids':[r['id'] for r in requests],'query_lengths':[r['token_ids'].numel() for r in requests],'input_ids':[r['token_ids'].cpu().tolist() for r in requests],'positions':[r['positions'].cpu().tolist() for r in requests],'KV_deltas':[],'attention':[],'activations':[],'sample_actions':[]}
                    assert shutil.disk_usage(self.evidence).free>4*1024**3+int(sum(v.nbytes for v in self.arrays.values())*1.02),'Independent lossless evidence space reserve: 3GiB hard + 1GiB log plus current raw arrays'
                    logits=self._profile('language',self.model.text_step,requests)
                    assert len(self.step['activations'])==30 and len(self.step['KV_deltas'])==60*len(requests) and len(self.step['attention'])==30*len(requests)
                    values={r['id']:int(token) for r,token in zip(requests,logits.argmax(-1).cpu().tolist())};self.step['sample_actions']=self.scheduler.consume_argmax(values)
                    stream.write(json.dumps(self.step,ensure_ascii=False)+'\n');stream.flush();self.group['steps'].append({'ordinal':step_ordinal,'request_ids':self.step['request_ids'],'query_lengths':self.step['query_lengths'],'argmax':list(values.values())})
                    step_ordinal+=1;self.step=None
                    log.info('Actual GraniteDocling original group=%s forward=%s request_count=%s ended=%s',ordinal,step_ordinal,len(batch),sum(x.finish_reason is not None for x in self.scheduler.requests.values()))
            producer.join();assert not producer_errors and len(prepared)==len(batch)
            archive=folder/'full_vocab_KV_inputs_visual.npz';np.savez_compressed(archive,**self.arrays);archive_sha=file_sha(archive);trace_sha=file_sha(trace_path);outputs=[]
            for row in prepared:
                rid=row['request_id'];r=self.scheduler.requests[rid];assert r.finish_reason is not None;raw,warnings=self.io.decode(row['input_ids'],r.output_ids,r.finish_reason);final=[]
                for layer,cache in enumerate(self.model.cache):
                    key,value=cache[rid];final.append({'layer':layer,'key_shape':list(key.shape),'value_shape':list(value.shape),'key_SHA256':tensor_sha(key),'value_SHA256':tensor_sha(value),'dtype':str(key.dtype),'device':str(key.device)})
                count=sum(rid in step['request_ids'] for step in self.group['steps'])
                entry=dict(row,raw_output=raw,generation_token_ids=list(r.output_ids),generation_count=len(r.output_ids),original_max_new_tokens=2048,finish_reason=r.finish_reason,stop_reason=None,truncated=r.finish_reason=='length',empty=not bool(raw),error=None,computed_prefix_hit_tokens=r.prefix_count,prefix_source_request_id=r.prefix_source,actual_request_forward_count=count,final_all30_KV=final,native_detokenization_warnings=warnings,group=ordinal,group_archive=str(archive),group_archive_SHA256=archive_sha,group_steps=str(trace_path),group_steps_SHA256=trace_sha)
                self.details[(row['prompt'].strip(),row['media_path'])]=entry;self.details[(row['prompt'].strip(),row['canonical_media_path'])]=entry
                with self.trace.open('a',encoding='utf-8') as handle:handle.write(json.dumps(entry,ensure_ascii=False)+'\n')
                outputs.append(raw)
            self.model.release_requests(list(self.scheduler.requests));self.scheduler.end_batch();assert not any(self.model.cache)
            self.group.update(activation_call_counts=dict(self.activation_counts),actual_model_forward_count=step_ordinal,archive=str(archive),archive_SHA256=archive_sha,trace=str(trace_path),trace_SHA256=trace_sha,elapsed_seconds=time.monotonic()-started,peak_NPU_allocated_bytes=torch.npu.max_memory_allocated(self.device))
            summary=folder/'group.json';summary.write_text(json.dumps(self.group,ensure_ascii=False,indent=2));self.groups.append({'path':str(summary),'SHA256':file_sha(summary),'archive':str(archive),'archive_SHA256':archive_sha,'trace':str(trace_path),'trace_SHA256':trace_sha,'request_count':len(batch),'actual_forward_count':step_ordinal,'encoder_wave_count':len(self.group['encoder_waves'])})
            (self.evidence/'completed_groups.json').write_text(json.dumps(self.groups,indent=2));self.next_index+=len(batch);self.weights.assert_resources_unchanged();self.group=None;self.arrays={};return outputs
        except BaseException:
            if 'producer' in locals():producer.join()
            self.batch_failed=True;error=traceback.format_exc();self.group['error']=error;(folder/'failed_group.json').write_text(json.dumps(self.group,ensure_ascii=False,indent=2))
            if self.arrays:np.savez_compressed(folder/'failed_partial_tensors.npz',**self.arrays)
            for row in prepared:
                r=getattr(self,'scheduler',None);r=None if r is None else r.requests.get(row['request_id']);entry=dict(row,error=error,raw_output='',generation_token_ids=[] if r is None else list(r.output_ids),generation_count=0 if r is None else len(r.output_ids),empty=True,truncated=False)
                self.details[(row['prompt'].strip(),row['media_path'])]=entry
                with self.trace.open('a',encoding='utf-8') as handle:handle.write(json.dumps(entry,ensure_ascii=False)+'\n')
            log.error(error);raise
    def predict(self,prompt,media_path=None):
        if self.batch_failed:raise RuntimeError('Original group failure retained; no singleton fallback')
        return self.batch_predict([(prompt,media_path)])[0]
    def get_sample_execution(self,prompt,media_path):
        for candidate in [prompt,NativeExplicitMCV2._add_answer_instruction(prompt)]:
            key=(candidate.strip(),str(media_path))
            if key in self.details:return self.details[key]
        return {'error':'Missing original GraniteDocling execution'}
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        self.model=None;self.weights=None;self.features.clear();torch.npu.empty_cache()
        if self.self_lock is not None:fcntl.flock(self.self_lock,fcntl.LOCK_UN);self.self_lock.close();self.self_lock=None

def fingerprint(value):return tensor_sha(value)

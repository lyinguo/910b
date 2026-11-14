"""Original A100 Pali2-3B-896 caption protocol on a disclosed Ascend reference.

Own727 BF16 weights, original logical10 requests, 8192 scheduled-token and
8192 encoder-token budgets, real partial prefills, prefix/window masks, full
257216 head, original non-MC caption extraction and asynchronous EOS boundary.
This is not CUDA Triton/paged acceleration or an approved result by itself.
"""
import collections,contextlib,datetime,fcntl,hashlib,json,logging,os,time,traceback
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np,torch,torch_npu
from universal_eval.model.interface import IModel
from universal_eval.model.adapters.paligemma2_3b_pt896_vllm_A100_math_v1_0403_20261106 import SourceWeights,Pali3Pt896NativeMath,tensor_sha,file_sha
from universal_eval.model.adapters.paligemma2_3b_pt896_vllm_A100_io_v1_0403_20261106 import OwnPali3Pt896IO
from universal_eval.model.adapters.paligemma2_3b_pt896_vllm_A100_scheduler_v3_original_limit_0403_20261106 import OriginalPaliScheduler
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';log=logging.getLogger(__name__)


def save_evidence_arrays(path,arrays):
    """Lossless NPZ level1; all original array bytes/shapes/keys are retained."""
    import zipfile
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for key,value in arrays.items():
            assert isinstance(key,str) and '/' not in key and isinstance(value,np.ndarray)
            with archive.open(key+'.npy','w',force_zip64=True) as member:
                np.lib.format.write_array(member,value,allow_pickle=False)

class Pali3Pt896A100VllmNative0403Adapter(IModel):
    def __init__(self,protocol_file,protocol_sha256,npu_devices=None,
                 evidence_dir='workspace/diagnostics/model_runs',
                 numerical_review=None,numerical_review_sha256=None,**kwargs):
        self.protocol_file=Path(protocol_file);assert file_sha(self.protocol_file)==protocol_sha256
        self.protocol_sha256=protocol_sha256;self.p=json.loads(self.protocol_file.read_text())
        assert self.p['model_id']=='google/paligemma2-3b-pt-896'
        self.devices=list(npu_devices or [1]);assert self.devices==[1];self.device=torch.device('npu:1')
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=Path(evidence_dir).resolve()/('Pali3Pt896_A100_vllm_0403_'+stamp+'_'+str(os.getpid()))
        self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl';self.details={};self.groups=[];self.next_index=0
        self.batch_failed=False;self.self_lock=None;self.profiled=set();self.group=None;self.step=None
        self.arrays={};self.activation_counts=collections.Counter();self.vision_wave=0;self.prefix_images=set()
        self.numerical_review=numerical_review;self.numerical_review_sha256=numerical_review_sha256

    @property
    def model_name(self):return self.p['model_id']
    @property
    def model_type(self):return 'PaliGemmaForConditionalGeneration'
    @property
    def inference_framework(self):return 'Ascend torch_npu original A100 vLLM BF16 logical reference'
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
            ancestors.append(pid);path=Path('/proc')/str(pid)
            if not path.exists():break
            pid=int(next(x.split(':',1)[1].strip() for x in (path/'status').read_text().splitlines() if x.startswith('PPid:')))
        assert len(owners)==1 and owners[0] in ancestors
        args=[x.decode() for x in (Path('/proc')/str(owners[0])/'cmdline').read_bytes().split(b'\0') if x]
        assert Path(args[0]).name.startswith('python') and str(R/'scripts/A100_alignment_evidence_runner_20261103.py') in args

    def load(self,model_path,**kwargs):
        self._ensure_runner_lock();p=self.p;self.weight_path=Path(model_path).resolve()
        assert str(self.weight_path)==p['weight_path']
        assert (p['original_max_tokens'],p['original_request_batch_size'],p['original_max_model_len'])==(1024,10,8192)
        assert p['original_primary_eos']==1 and p['original_stop_token_ids']==[]
        assert p['all200_unique_native_MM_hashes_and_pixels'] and p['full_original_algorithm_input_CPU_delta_qualified']
        assert p['actual_AA3_and_B10_scheduler_and_mask_qualified']
        assert file_sha(Path(p['dataset']))==p['dataset_SHA256']
        for name,digest in p['required_code_sha256'].items():
            path=Path(name);assert not path.is_absolute() and '..' not in path.parts;assert file_sha(R/path)==digest,name
        assert len(p['input_records'])==200 and [r['index'] for r in p['input_records']]==list(range(200))
        assert all('reference' not in row and 'original_A100_raw' not in row and 'raw_output' not in row for row in p['input_records'])
        torch.set_num_threads(4);torch.npu.set_device(self.device)
        for kind in ['conv','matmul']:getattr(torch.npu,kind).allow_hf32=False
        self.weights=SourceWeights(self.weight_path,p,self.device,lambda e:log.info('Own Pali3Pt896 source/cast/NPU SHA %s',e))
        self.io=OwnPali3Pt896IO(self.weight_path,p['own_resource_SHA256']);self.rows=p['input_records']
        self.model=Pali3Pt896NativeMath(self.weights,p['own_actual_AA_effective_config'],self._observe)
        for name,data in [('weight_loading.json',self.weights.records),('physical_packed_weights.json',self.weights.physical)]:
            (self.evidence/name).write_text(json.dumps(data,indent=2)+'\n')
        snapshot=self.evidence/'source_snapshot';snapshot.mkdir()
        for name in p['required_code_sha256']:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/name).read_bytes())
        (snapshot/'execution_protocol.json').write_bytes(self.protocol_file.read_bytes())
        rope=self.model.rope['full_attention'].cache
        buffer=self.evidence/'actual_NPU_full8192_BF16_RoPE_and_text_scale.npz'
        np.savez_compressed(buffer,rope=rope.cpu().reshape(-1).view(torch.uint8).numpy(),scale=self.model.scale.cpu().reshape(-1).view(torch.uint8).numpy())
        self.lineage={'hardware':'Ascend910B','devices':self.devices,'runtime_dtype':'BF16',
            'model_id':self.p['model_id'],'protocol_file':str(self.protocol_file),'protocol_SHA256':self.protocol_sha256,
            'weight_tensor_count':727,'physical_packed_parameter_count':541,'all_expected_parameters_loaded':True,
            'weight_loading_evidence':str(self.evidence/'weight_loading.json'),'physical_packed_weights':str(self.evidence/'physical_packed_weights.json'),
            'original_max_new_tokens':1024,'original_outer_request_batch_size':10,'original_max_model_len':8192,
            'original_primary_eos_token_id':1,'original_additional_stop_token_ids':[],'original_min_tokens':0,
            'original_full_head_dtype':'BF16','original_full_vocabulary':257216,'head_uses_source_embedding_directly_without_separate_parameter':True,
            'original_image_tokens':4096,'single_own_BOS_after_image_prefix':True,'own_template_is_None_original_builtin_Pali_fallback':True,
            'original_algorithm_caption_instruction_retained':True,'original_non_MC_extract_answer_true':True,
            'original_request_scheduled_token_budget':8192,'original_encoder_token_budget':8192,'original_encoder_tokens_per_image':4096,
            'original_prefix_caching_enabled':True,'original_block_size':16,'original_MM_prefix_inclusive_range':[0,4095],
            'original_softcap_attention_before_mask':50.0,'original_head_separate_BF16_div_tanh_multiply':30.0,
            'original_full_RoPE_buffer':{'shape':[8192,256],'dtype':str(rope.dtype),'device':str(rope.device),'SHA256':tensor_sha(rope)},
            'actual_NPU_RoPE_and_scale':{'path':str(buffer),'SHA256':file_sha(buffer)},
            'completed_groups_manifest':str(self.evidence/'completed_groups.json'),'per_sample_evidence':str(self.trace),
            'profiler_traces':{s:str(self.evidence/(s+'_npu_trace.json')) for s in ['vision','language']},
            'profiler_capture_scope':'Only first actual vision wave and first decoder prefill; all forward/sample/cache evidence retained separately.',
            'cache_disclosure':'Original unique-image fixed200 scope has no computed prefix or MM cache hits. First block binds own image hash. Two encoder entries stay reserved across original two-stage asynchronous planner; partial4096 image prefixes and discarded partial-prefill sampling retained. Logical contiguous BF16 KV implements original visibility; no CUDA paged allocation acceleration claim.',
            'precision_disclosure':'Own727 BF16 source and actual NPU weights,541 physical packed parameters. Original unrounded F32 residual and F32(weight+1) RMSNorm, BF16 GELU intermediate, original BF16 head with separate30 softcap ops, text-only48 and visual inverse48, own-device F32-formula BF16 RoPE. Attention uses original BF16 operand values with F32 score accumulation/softcap/softmax and BF16 value products. Backend differences remain real and disclosed; no official FP32/CPU model answer, precision/mask/budget change or low-bit acceleration claim.'}
        if self.numerical_review is not None:
            assert file_sha(Path(self.numerical_review))==self.numerical_review_sha256
            self.lineage.update(numerical_review=self.numerical_review,numerical_review_SHA256=self.numerical_review_sha256)

    def _tensor_record(self,value,key=None,full_hash=False):
        assert value.device==self.device
        rec={'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
        if key is not None:
            a=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy();assert key not in self.arrays
            self.arrays[key]=a;rec.update(array=key,SHA256=hashlib.sha256(a.tobytes()).hexdigest())
        elif full_hash:rec['SHA256']=tensor_sha(value)
        return rec

    def _observe(self,name,value,metadata):
        assert self.group is not None
        if name.startswith(('vision_activation_','language_activation_')):
            self.activation_counts[name]+=1
            rec=self._tensor_record(value,full_hash=name.startswith('vision_') or self.step['ordinal']==0)
            rec.update(module=name,call=self.activation_counts[name],metadata=metadata)
            (self.group['vision_waves'][-1]['activations'] if name.startswith('vision_') else self.step['activations']).append(rec);return
        if name in ['vision_actual_pixels','vision_encoder_output','visual_projector_output','original_Pali_normalized_visual_embeddings']:
            self.group['vision_waves'][-1]['tensors'][name]=self._tensor_record(value,'v%02d_%s'%(self.vision_wave,name));return
        assert self.step is not None
        n=self.step['ordinal']
        if name.startswith('KV_delta_'):
            rec=self._tensor_record(value,'s%04d_r%s_%s'%(n,metadata['request_id'],name));rec.update(name=name,**metadata);self.step['KV_deltas'].append(rec);return
        if name.startswith('attention_visibility_'):
            key='s%04d_r%s_mask_%s'%(n,metadata['request_id'],metadata['layer_type'])
            if key in self.arrays:
                rec=self._tensor_record(value,full_hash=True);assert rec['SHA256']==hashlib.sha256(self.arrays[key].tobytes()).hexdigest();rec['array']=key
            else:rec=self._tensor_record(value,key)
            rec.update(layer=int(name.rsplit('_',1)[1]),**metadata);self.step['attention'].append(rec);return
        if name=='text_input_embeddings':
            rec=self._tensor_record(value,'s%04d_inputs_embeds'%n if n==0 else None);rec.update(metadata);self.step['input_embeddings']=rec;return
        if name=='full_vocabulary_logits':
            rec=self._tensor_record(value,'s%04d_full257216_BF16_logits'%n);rec.update(metadata);self.step['full_logits']=rec;return
        raise RuntimeError('Unqualified observation '+name)

    def _profile(self,which,function,*args):
        if which in self.profiled:return function(*args)
        with contextlib.chdir(self.evidence):
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True) as prof:
                result=function(*args);torch.npu.synchronize(self.device)
            prof.export_chrome_trace(str(self.evidence/(which+'_npu_trace.json')))
        self.profiled.add(which);return result

    @torch.inference_mode()
    def batch_predict(self,batch):
        if self.batch_failed:raise RuntimeError('Previous original batch failed; no singleton retry')
        assert 0<len(batch)<=10 and self.next_index+len(batch)<=200
        begin=time.monotonic();ordinal=len(self.groups);folder=self.evidence/'groups'/('%03d'%ordinal);folder.mkdir()
        self.arrays={};self.activation_counts=collections.Counter();self.vision_wave=0
        self.group={'ordinal':ordinal,'start_index':self.next_index,'request_count':len(batch),'requests':[],'vision_waves':[],'steps':[],'error':None}
        states={};pixels={};features={}
        try:
            for offset,(prompt,media) in enumerate(batch):
                index=self.next_index+offset;expected=self.rows[index];rid=str(offset)
                assert prompt==expected['model_prompt'] and str(Path(media).resolve())==expected['R_media_path']
                rendered,ids,start=self.io.render(prompt);assert ids==expected['ids'] and hashlib.sha256(rendered.encode()).hexdigest()==expected['rendered_SHA256']
                pix,ms,ps,canonical=self.io.pixels(media);assert ms==expected['media_SHA256'] and ps==expected['pixels_SHA256']
                assert ms not in self.prefix_images,'Computed prefix reuse beyond unique-image qualified scope';self.prefix_images.add(ms)
                row={'index':index,'request_id':rid,'prompt':prompt,'original_sample_prompt':expected['original_prompt'],'media_path':str(media),'canonical_media_path':canonical,'source_image_SHA256':ms,'input_ids':ids,'image_soft_start_position':start,'rendered':rendered,'CPU_pixel_SHA256':ps,'CPU_pixel_shape':list(pix.shape),'computed_prefix_hit_tokens':0,'original_prefix_cache_enabled':True}
                self.group['requests'].append(row);states[rid]={'row':row,'tokens':[],'forward_count':0};pixels[rid]=pix
            sched=OriginalPaliScheduler([len(s['row']['input_ids']) for s in states.values()]);trace_path=folder/'steps.jsonl'
            with trace_path.open('w',encoding='utf8') as trace:
                while not sched.complete:
                    plan=sched.plan();new=plan['encoder_requests']
                    if new:
                        wave={'ordinal':self.vision_wave,'request_ids':list(new),'tensors':{},'activations':[]};self.group['vision_waves'].append(wave)
                        projected=self._profile('vision',self.model.vision,torch.cat([pixels[r] for r in new]).to(self.device))
                        assert projected.shape==(len(new),4096,2304)
                        for rid,value in zip(new,projected):features[rid]=value
                        self.vision_wave+=1
                    requests=[]
                    for req in plan['requests']:
                        rid=req['id'];st=states[rid];past=req['past'];q=req['query_length']
                        ids=st['row']['input_ids'][past:past+q] if past<4111 else [st['tokens'][-1]]
                        assert len(ids)==q
                        requests.append({'id':rid,'token_ids':torch.tensor(ids,dtype=torch.int64,device=self.device),'positions':torch.arange(past,past+q,dtype=torch.int64,device=self.device),'visuals':features[rid] if past<4096 else None,'image_start_position':0 if past<4096 else None,'mm_prefix_range':(0,4095)})
                    n=plan['ordinal'];self.step={'ordinal':n,'request_ids':[r['id'] for r in requests],'query_lengths':[r['token_ids'].numel() for r in requests],'input_ids':[r['token_ids'].cpu().tolist() for r in requests],'positions':[r['positions'].cpu().tolist() for r in requests],'encoder_requests':new,'KV_deltas':[],'attention':[],'activations':[]}
                    logits=self._profile('language',self.model.text_step,requests);assert logits.shape==(len(requests),257216) and logits.dtype==torch.bfloat16
                    assert len(self.step['activations'])==26 and len(self.step['KV_deltas'])==52*len(requests) and len(self.step['attention'])==26*len(requests)
                    best=logits.argmax(-1).cpu().tolist();actions,released,freed=sched.advance(best);self.step['sample_actions']=actions;self.step['encoder_freed_after_async_previous_output']=freed
                    for action in actions:
                        st=states[action['request_id']];st['forward_count']+=1
                        if action['kind']=='reported_sample':st['tokens'].append(action['token_id'])
                    for rid in released:
                        st=states[rid];records=[]
                        for layer,cache in enumerate(self.model.cache):
                            key,val=cache[rid];records.append({'layer':layer,'key_shape':list(key.shape),'value_shape':list(val.shape),'key_SHA256':tensor_sha(key),'value_SHA256':tensor_sha(val),'device':str(key.device),'dtype':str(key.dtype)})
                        st['final_KV']=records
                    trace.write(json.dumps(self.step,ensure_ascii=False)+'\n');trace.flush()
                    self.group['steps'].append({'ordinal':n,'request_ids':self.step['request_ids'],'query_lengths':self.step['query_lengths'],'argmax':best})
                    if released:self.model.release_requests(released)
                    for rid in freed:features.pop(rid,None)
                    self.step=None
                    if n%16==0:log.info('Pali3Pt896 original group=%s forward=%s active=%s finished=%s/%s',ordinal,n+1,len(sched.running),sum(r['finished'] for r in sched.requests),len(batch))
            assert not any(self.model.cache) and not features
            archive=folder/'full_vocab_KV_inputs_visual.npz';save_evidence_arrays(archive,self.arrays);archive_sha=file_sha(archive);trace_sha=file_sha(trace_path);outputs=[]
            for rid,st in states.items():
                sr=sched.requests[int(rid)];row=st['row'];tokens=st['tokens'];assert sr['finished'] and len(tokens)==sr['generated']
                raw,warnings=self.io.decode(row['input_ids'],tokens,sr['finish'])
                entry=dict(row,raw_output=raw,generation_token_ids=tokens,generation_count=len(tokens),original_max_new_tokens=1024,finish_reason=sr['finish'],stop_reason=None,truncated=sr['finish']=='length',empty=not raw,error=None,actual_request_forward_count=st['forward_count'],final_all26_KV=st['final_KV'],original_partial_prefill_and_EOS_lookahead_preserved=True,native_detokenization_warnings=warnings,group=ordinal,group_archive=str(archive),group_archive_SHA256=archive_sha,group_steps=str(trace_path),group_steps_SHA256=trace_sha)
                for prompt_key in [row['prompt'],row['original_sample_prompt']]:
                    for media_key in [row['media_path'],row['canonical_media_path']]:self.details[(prompt_key.strip(),media_key)]=entry
                with self.trace.open('a',encoding='utf8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                outputs.append(raw)
            self.group.update(activation_call_counts=dict(self.activation_counts),actual_model_forward_count=sched.ordinal,archive=str(archive),archive_SHA256=archive_sha,trace=str(trace_path),trace_SHA256=trace_sha,elapsed_seconds=time.monotonic()-begin,peak_NPU_allocated_bytes=torch.npu.max_memory_allocated(self.device))
            summary=folder/'group.json';summary.write_text(json.dumps(self.group,ensure_ascii=False,indent=2)+'\n')
            self.groups.append({'path':str(summary),'SHA256':file_sha(summary),'archive':str(archive),'archive_SHA256':archive_sha,'trace':str(trace_path),'trace_SHA256':trace_sha,'request_count':len(batch),'actual_forward_count':sched.ordinal,'encoder_wave_sizes':[len(w['request_ids']) for w in self.group['vision_waves']]})
            (self.evidence/'completed_groups.json').write_text(json.dumps(self.groups,indent=2)+'\n');self.next_index+=len(batch);self.weights.assert_resources_unchanged();self.group=None;self.arrays={};return outputs
        except BaseException:
            self.batch_failed=True;error=traceback.format_exc();self.group['error']=error;(folder/'failed_group.json').write_text(json.dumps(self.group,ensure_ascii=False,indent=2)+'\n');log.error(error);raise

    def predict(self,prompt,media_path=None):
        if self.batch_failed:raise RuntimeError('Batch failure retained; no singleton fallback')
        return self.batch_predict([(prompt,media_path)])[0]
    def get_sample_execution(self,prompt,media_path):return self.details.get((prompt.strip(),str(media_path)),{'error':'Missing exact original Pali3Pt896 execution'})
    def get_execution_lineage(self):return self.lineage
    def unload(self):
        self.model=None;self.weights=None;torch.npu.empty_cache()
        if self.self_lock is not None:fcntl.flock(self.self_lock,fcntl.LOCK_UN);self.self_lock.close();self.self_lock=None

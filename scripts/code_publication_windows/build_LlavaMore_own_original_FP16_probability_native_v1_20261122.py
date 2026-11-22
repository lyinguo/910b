"""Build the own original letter-scoring native adapter; never alter active Xtuner."""
from pathlib import Path
import ast,hashlib,json
V=Path(__file__).resolve().parent
old=(V/'xtuner_llava3_own_original_Transformers_FP16_2048_native_v2_IModel_20261122.py').read_text()
tree=ast.parse(old);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OwnXtunerLlava3OriginalFP16')
def method(name):
 node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
 return '\n'.join(old.splitlines()[node.lineno-1:node.end_lineno])+'\n'
graph=(V/'xtuner_llava3_own_saved_Transformers_graph_20261122.py').read_text().replace('Only the saved own legacy configuration/name mapping; no learned forward replacement.','Own More saved inherited legacy graph: flat raw Llama3.1 scaling omitted by original conversion; no learned forward replacement.')
graph_name='llava_more_own_original_saved_Transformers_FP16_graph_20261122.py'
(V/graph_name).write_text(graph,encoding='utf8',newline='\n')
header='''"""Own saved LLaVA_MORE pad/system/single-forward letter-logit protocol on NPU."""
import datetime,hashlib,inspect,json,logging,os,pathlib,shutil,traceback
import numpy as np
import torch,torch_npu
from torch import nn
from PIL import Image
from transformers import AutoTokenizer,CLIPImageProcessor,LlavaProcessor,LlavaForConditionalGeneration,GenerationConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from universal_eval.model.interface import IModel
from universal_eval.model.llava_more_own_original_saved_Transformers_FP16_graph_20261122 import build_own_meta
from universal_eval.model.adapters.llava_more_vlm import LlavaMoreVLM
from universal_eval.utils.complete_standard_LZMA_NPZ_parallel_values_20261122 import write_full_standard_npz
from scripts.egm8b_original_independent_cpu_910b import sha,fp,tr
log=logging.getLogger(__name__)
MID='aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning'

class OwnObservedOriginalLlavaMore(LlavaMoreVLM):
    def _letter_scores(self,last_logits,letters):
        scores=super()._letter_scores(last_logits,letters)
        self.own_complete_letter_observer(last_logits,letters,scores)
        return scores

class OwnLlavaMoreOriginalFP16Letter(IModel):
    def __init__(self,protocol_file,protocol_sha256,npu_device=4,evidence_dir='workspace/diagnostics/model_runs',**kwargs):
        assert sha(protocol_file)==protocol_sha256
        self.p=json.loads(pathlib.Path(protocol_file).read_bytes());self.protocol_file=str(pathlib.Path(protocol_file).resolve());self.protocol_sha=protocol_sha256
        self.device=torch.device('npu:'+str(npu_device));assert self.device.index==4
        stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        self.evidence=pathlib.Path(evidence_dir).resolve()/('LlavaMore_own_original_FP16_letter_'+stamp+'_'+str(os.getpid()));self.evidence.mkdir(parents=True,exist_ok=False);(self.evidence/'groups').mkdir()
        self.trace=self.evidence/'sample_execution.jsonl';self.details={};self.group_count=0;self.report_index=0;self.lineage={};self._model=None
    @property
    def model_type(self):return 'LlavaLlamaForCausalLM'
    @property
    def model_name(self):return MID
    @property
    def inference_framework(self):return 'Own saved native Transformers FP16 / original discriminative_letter / Ascend910B'
    def download(self,*args,**kwargs):raise RuntimeError('Downloads prohibited')
'''
load=method('load')
load=load.replace('model=build_own_meta(raw);model.__class__=OwnPrivateCacheLlama;','model=build_own_meta(raw);')
load=load.replace("source.dtype in [torch.float16,torch.float32]","source.dtype==torch.float16")
load=load.replace("log.info('Own Xtuner source/cast/actual NPU verified %d/686'","log.info('Own More source/cast/actual NPU verified %d/686'")
load=load.replace('assert model.generation_config.eos_token_id==128001','assert model.generation_config.eos_token_id==[128001,128008,128009]')
a=load.index('        gen,_=model._prepare_generation_config');b=load.index('        def fused_pre',a)
load=load[:a]+'''        self.generation_config=model.generation_config.to_dict();self._dtype=torch.float16;self._max_new_tokens=2048;self.hooks=[];self.module_kinds={};self._install_observers()
        self.actor=OwnObservedOriginalLlavaMore(device='auto',max_new_tokens=2048,torch_dtype='auto',image_mode='pad',mode='discriminative_letter')
        self.actor._model=model;self.actor._processor=self.processor;self.actor._tokenizer=self._tokenizer;self.actor._model_name=MID;self.actor._model_path=str(wp)
        self.actor.own_complete_letter_observer=self._capture_original_letter_scores
        assert self.actor._predict_letter.__func__ is LlavaMoreVLM._predict_letter and self.actor.predict.__func__ is LlavaMoreVLM.predict
''' +load[b:]
a=load.index('        self.lineage=dict(')
load=load[:a]+'''        self.lineage=dict(hardware='Ascend910B',device=str(self.device),protocol_file=self.protocol_file,protocol_sha256=self.protocol_sha,weight_loading_evidence=str(weight_file),source_tensor_count=686,actual_FP16_parameter_count=686,source_dtype_counts={'F16':686},original_max_new_tokens=2048,original_batch_size=10,actual_inner_batch_size=1,original_probability_letter_one_forward_cacheFalse=True,original_use_cache=False,physical_KV_cache_not_created_by_original_protocol=True,original_plain_F32_RoPE_buffers_preserved=True,original_raw_llama31_scaling_omitted_by_saved_legacy_conversion=True,original_CLIP24_projector2_Llama32_full128257_head_preserved=True,own_original_constructor_review=p['own_original_constructor_review'],own_original_constructor_review_SHA256=p['own_original_constructor_review_SHA256'],original_constructor_derived_extra_row_values=extra_path,original_constructor_derived_extra_row_values_SHA256=sha(extra_path),original_raw_generation_config_unused_in_scoring=self.generation_config,original_analytic_buffers={n:tr(t) for n,t in model.named_buffers()},per_sample_evidence=str(self.trace),per_group_evidence=str(self.evidence/'groups'),complete_numeric_module_kinds=self.module_kinds,profiler=str(self.evidence/'first_vision_prefill_npu_trace.json'),profiler_scope='First own complete vision and full text probability prefill only',historical_A100_engine_or_tensor_capture_claimed=False,all_learned_forwards_native_NPU=True,original_pad_RGB=[122,116,104],original_system_prompt=self.actor._system_prompt,original_letter_score_function_unchanged=True,precision_disclosure='Own686F16 physical source and original requested FP16; own saved resize-before-remap derives one complete head/image row with original CPU constructor/RNG proof, no foreign checkpoint. Original FP32 RoPE buffers and original probability score .float cast preserved.',complete_standard_NPY_LZMA_storage='Bounded 8 CPU workers produce the original complete members in original order; all value bits and standard archive bytes independently tested equal; not parallel model NN',HF32_enabled={'conv':False,'matmul':False})
'''
observer=method('_install_observers')
a=observer.index('                cache=out.past_key_values');b=observer.index('                if self.profile is not None:',a)
observer=observer[:a]+'''                assert kw.get('use_cache') is False and out.past_key_values is None
                row=self.steps[-1];row['original_use_cache']=False;row['past_key_values_is_None']=True
                row['full_logits']=self._tensor(out.logits)
                row['native_full_vocabulary_argmax_information_only']=out.logits[:,-1,:].argmax(-1).tolist()
                assert out.logits.shape[0]==1 and out.logits.shape[-1]==128257 and out.logits.dtype==torch.float16
''' +observer[b:]
score='''    def _capture_original_letter_scores(self,last_logits,letters,scores):
        assert self.active and last_logits.device==self.device and last_logits.dtype==torch.float32 and last_logits.shape==(128257,)
        self.score_calls+=1;assert self.score_calls==1
        row=self.steps[-1];row['original_F32_last_logits']=self._tensor(last_logits);row['original_candidate_letters']=letters;row['original_variant_ID_sets']=self.front[self.current_index]['original_letter_candidate_ID_sets'];row['original_variant_max_scores']=scores
        with (self.evidence/'probability_progress.jsonl').open('a',encoding='utf8') as file:file.write(json.dumps(dict(index=self.current_index,native_forward_count=len(self.steps),original_probability_branch=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\\n')
'''
predict='''    def predict(self,prompt,media_path=None):
        assert self._model is not None and media_path and pathlib.Path(media_path).is_file()
        i=len(self.details);row=self.front[i];assert row['actual_algorithm_prompt']==prompt and row['media_file_SHA256']==sha(media_path)
        reserve=self.p['hard_reserve_bytes']+self.p['log_reserve_bytes']+self.p['precheck_reserve_bytes']+self.p['per_sample_lossless_space_estimate_bytes'];assert shutil.disk_usage(self.evidence).free>=reserve
        self.group_count+=1;group=self.evidence/'groups'/('%04d'%self.group_count);group.mkdir();self.arrays={};self.calls=[];self.steps=[];self.stacks={};self.text_step=0;self.current_index=i;self.occurrences=0;self.score_calls=0;self.active=True;error=None;self.inputs=None;self.fused=None
        self.profile=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,profile_memory=False,with_stack=False) if i==0 else None;self.profiler=self.profile
        if self.profile is not None:self.profile.start()
        def original_native_input_pre(mod,args,kw):
            assert kw['input_ids'][0].tolist()==row['input_ids'] and fp(kw['pixel_values'])==row['original_actual_FP16_inputs_SHA256'] and kw['pixel_values'].dtype==torch.float16 and kw.get('use_cache') is False
            self.inputs=self._tree(dict(pixels_FP16=kw['pixel_values'],input_ids=kw['input_ids'],attention_mask=kw['attention_mask']))
        check=self._model.register_forward_pre_hook(original_native_input_pre,with_kwargs=True)
        try:
            from universal_eval.model.adapters.llava_more_vlm import _strip_eval_instruction,_parse_options
            clean=_strip_eval_instruction(prompt).replace('<image>','',1).lstrip('\\n').strip();assert self.actor._build_text(clean)==row['text'] and [list(x) for x in _parse_options(clean)]==row['original_options']
            with torch.inference_mode():raw=self.actor.predict(prompt,media_path)
            assert isinstance(raw,str) and self.score_calls==1 and len(self.steps)==1 and raw==max(self.steps[0]['original_variant_max_scores'],key=self.steps[0]['original_variant_max_scores'].get)
        except BaseException:error=traceback.format_exc();raise
        finally:
            check.remove()
            if self.profile is not None:self.profile.stop();self.profile=None;self.profiler.export_chrome_trace(str(self.evidence/'first_vision_prefill_npu_trace.json'))
            self.active=False;archive=group/'complete_actual_values.npz';write_full_standard_npz(archive,self.arrays)
            record=dict(index=i,group=self.group_count,error=error,inputs=self.inputs,fused=self.fused,calls=self.calls,steps=self.steps,archive=str(archive),archive_SHA256=sha(archive),complete_copied_numeric_occurrences=self.occurrences,unique_full_NPY_values=len(self.arrays),original_full_values_shared_only_after_shape_dtype_and_every_raw_bit_equal=True,original_probability_branch_not_generation=True,all_complete_values_written_by_standard_LZMA_parallel_members=True);(group/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
        execution=dict(index=i,prompt=row['prompt'],actual_model_prompt=prompt,original_probability_text=row['text'],media_path=media_path,media_SHA256=sha(media_path),raw_output=raw,generation_token_ids=[],decoded_token_ids=[],first_EOS_index=None,generation_count=0,native_forward_count=1,original_probability_not_generation=True,original_use_cache=False,truncated=False,empty=not raw,invalid=raw not in self.steps[0]['original_candidate_letters'],error=None,group_execution=str(group/'execution.json'),group_execution_SHA256=sha(group/'execution.json'),actual_device=str(self.device))
        self.details[i]=execution
        with self.trace.open('a',encoding='utf8') as file:file.write(json.dumps(execution,ensure_ascii=False)+'\\n')
        self.arrays={};log.info('Own More complete original probability sample %d, native forwards 1',i+1);return raw
'''
other='''    def batch_predict(self,batch):return [self.predict(prompt,media) for prompt,media in batch]
'''+method('get_sample_execution')+method('get_execution_lineage')+method('unload')
model=header+load+method('_tensor')+method('_tree')+observer+score+predict+other
ast.parse(model);name='llava_more_own_original_FP16_discriminative_letter_native_v1_20261122.py';(V/name).write_text(model,encoding='utf8',newline='\n')
algo='''"""Preserve saved Greedy tail and Accuracy; the own adapter retains original probability scoring."""
from universal_eval.algorithm.gemma3n_E4B_actual_original_HighTen_native_explicit_MC_v3_after_reboot_20261113 import E4BActualOriginalHighTenMC
from universal_eval.algorithm.tinyllava_strict_mc import StrictMCAccuracy
class OwnLlavaMoreOriginalGreedyTail(E4BActualOriginalHighTenMC):pass
class OwnLlavaMoreOriginalAccuracy(StrictMCAccuracy):pass
'''
algo_name='llava_more_own_original_FP16_discriminative_letter_MC_20261122.py';(V/algo_name).write_text(algo,encoding='utf8',newline='\n')
primary={'universal_eval/model/'+graph_name:hashlib.sha256(graph.encode()).hexdigest(),'universal_eval/model/adapters/'+name:hashlib.sha256(model.encode()).hexdigest(),'universal_eval/algorithm/'+algo_name:hashlib.sha256(algo.encode()).hexdigest(),'scripts/code_publication_windows/'+Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for n in primary:
 p=V/Path(n).name;ast.parse(p.read_bytes())
(V/'LlavaMore_own_native_probability_primary_paths_v1_20261122.json').write_text(json.dumps(primary,indent=2)+'\n',encoding='utf8');print(json.dumps(primary))

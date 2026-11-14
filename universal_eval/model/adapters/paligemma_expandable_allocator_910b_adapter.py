"""Per-process allocator-only remedy; all native model arithmetic remains inherited unchanged."""
import os,json,pathlib,hashlib,inspect,torch,torch_npu
from universal_eval.model.adapters.paligemma_runtime_log_caption_910b_adapter import PaliGemmaRuntimeLogCaption910BAdapter

class PaliGemmaExpandableAllocator910BAdapter(PaliGemmaRuntimeLogCaption910BAdapter):
    def load(self,model_path,**kwargs):
        if self.source_dtype!='BF16' or self.expected_image_size!=896 or len(self.devices)!=1:
            raise RuntimeError('This allocator remedy was reviewed only for the exact BF16 896-pixel PaliGemma2 checkpoint')
        torch.npu.set_device(self.device)
        torch_npu.npu.memory._set_allocator_settings('expandable_segments:True')
        self.allocator_log=self.evidence/'allocator_execution.jsonl'
        super().load(model_path,**kwargs)
        if self.model_name!='google/paligemma2-28b-pt-896':raise RuntimeError('Not the separately reviewed exact model')
        memory_path=pathlib.Path(torch_npu.npu.memory.__file__).resolve()
        self.lineage['allocator_remedy']={'setting':'expandable_segments:True','applied_API':'torch_npu.npu.memory._set_allocator_settings','API_source':str(memory_path),'API_source_sha256':hashlib.sha256(memory_path.read_bytes()).hexdigest(),'allocator_execution':str(self.allocator_log),'allocation_after_load':self._memory(),'arithmetic_changed':False,'inherited_native_load_and_predict_source':'universal_eval/model/adapters/paligemma_runtime_log_caption_910b_adapter.py','scope':'Only this model process allocator segment growth. Native attention, softmax, matmul, generation/cache, all source BF16 weights, 896-pixel images and original token budget unchanged.'}
    def _memory(self):
        stats=torch.npu.memory_stats(self.device)
        return {k:int(v) for k,v in stats.items() if k in ('allocated_bytes.all.current','allocated_bytes.all.peak','reserved_bytes.all.current','reserved_bytes.all.peak','active_bytes.all.current','inactive_split_bytes.all.current','num_alloc_retries','num_ooms')}
    def predict(self,prompt,media_path=None):
        before=self._memory()
        try:return super().predict(prompt,media_path)
        finally:
            with self.allocator_log.open('a') as handle:handle.write(json.dumps({'attempt':self.calls,'setting':'expandable_segments:True','before':before,'after':self._memory()})+'\n')

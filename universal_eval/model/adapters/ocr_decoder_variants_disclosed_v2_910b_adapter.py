"""Post-sample native execution evidence while retaining the original native model graph."""
import json
from universal_eval.model.adapters.ocr_decoder_variants_disclosed_910b_adapter import OCRDecoderVariantsDisclosed910BAdapter

class OCRDecoderVariantsDisclosedV2Adapter(OCRDecoderVariantsDisclosed910BAdapter):
    def load(self,model_path,**kwargs):
        super().load(model_path,**kwargs)
        self._gelu_trace=self.evidence/'native_GELU_execution.jsonl'
        self.lineage['native_GELU_disclosure']['execution_evidence']=str(self._gelu_trace)
        self.lineage['native_GELU_disclosure']['lineage_snapshot_phase']='Loaded model; complete actual execution is recorded after each sample in the referenced evidence'
    def predict(self,prompt,media_path=None):
        before=dict(self._gelu_counts)
        try:return super().predict(prompt,media_path)
        finally:
            entry={'attempt':self.calls,'module_forward_counts_delta':{n:v-before[n] for n,v in self._gelu_counts.items()},'cumulative_counts':dict(self._gelu_counts),'actual_devices':{n:sorted(v) for n,v in self._gelu_devices.items()},'actual_dtypes':{n:sorted(v) for n,v in self._gelu_dtypes.items()}}
            with self._gelu_trace.open('a') as f:f.write(json.dumps(entry)+'\n')
    def get_execution_lineage(self):return self.lineage

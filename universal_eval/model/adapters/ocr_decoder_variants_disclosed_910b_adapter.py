"""Preserve the native NPU graph; explicitly disclose measured vendor GELU approximation semantics."""
import pathlib,json,hashlib,torch,numpy as np
import torch.nn.functional as F
from transformers.activations import GELUActivation
from universal_eval.model.adapters.ocr_decoder_variants_v2_910b_adapter import OCRDecoderVariants910BAdapterV2

class OCRDecoderVariantsDisclosed910BAdapter(OCRDecoderVariants910BAdapterV2):
    def load(self,model_path,**kwargs):
        super().load(model_path,**kwargs)
        x=torch.linspace(-8,8,8193,dtype=torch.float32)
        with torch.inference_mode():native=F.gelu(x.to(self.device),approximate='none').cpu()
        erf=F.gelu(x,approximate='none');tanh=F.gelu(x,approximate='tanh')
        exact_difference=float((native-erf).abs().max());tanh_difference=float((native-tanh).abs().max())
        assert exact_difference>1e-4 and tanh_difference<2e-6
        path=self.evidence/'native_GELU_semantics.npz';np.savez_compressed(path,input_F32=x.numpy(),actual_native_NPU_GELU=native.numpy())
        self._gelu_counts={};self._gelu_devices={};self._gelu_dtypes={}
        for name,module in self.model.named_modules():
            if type(module) is GELUActivation:
                assert module.act is F.gelu
                self._gelu_counts[name]=0;self._gelu_devices[name]=set();self._gelu_dtypes[name]=set()
                def observed(module,args,value,key=name):
                    assert value.device==self.device
                    self._gelu_counts[key]+=1;self._gelu_devices[key].add(str(value.device));self._gelu_dtypes[key].add(str(value.dtype))
                module.register_forward_hook(observed)
        assert self._gelu_counts
        self.lineage['native_GELU_disclosure']={'source_API':'torch.nn.functional.gelu approximate=none via original GELUActivation','observed_native_formula':'CANN fused native GELU follows tanh approximation on this stack','native_graph_unchanged':True,'no_NPU_ERF_replacement':True,'probe':str(path),'probe_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_NPU_vs_CPU_erf_max_abs':exact_difference,'native_NPU_vs_CPU_tanh_max_abs':tanh_difference,'probe_input_dtype':'torch.float32','original_source_class':'transformers.activations.GELUActivation','module_names':sorted(self._gelu_counts),'interpretation':'Real native hardware/operator numerical behavior. Original weights, graph API, precision and generation settings remain unchanged. CPU erf and CPU vendor-matched tanh references are diagnosis only and never substitute official NPU metrics.'}
        self.lineage['precision']+=' Native CANN fused GELU requested with approximate=none uses a measured tanh approximation. This observed vendor operator behavior is explicitly disclosed; no NPU ERF substitute, activation change or CPU learned-layer offload is used.'
    def get_execution_lineage(self):
        result=super().get_execution_lineage()
        result['native_GELU_disclosure']['actual_forward_counts']=dict(self._gelu_counts)
        result['native_GELU_disclosure']['actual_devices']={n:sorted(v) for n,v in self._gelu_devices.items()}
        result['native_GELU_disclosure']['actual_dtypes']={n:sorted(v) for n,v in self._gelu_dtypes.items()}
        return result

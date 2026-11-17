"""Own constructor model-ID binding; inherited original math is unchanged."""
from pathlib import Path
from universal_eval.model.adapters.deadbydawn_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261117 import DeadbydawnGemma4E2BOwnBF16,MID
class DeadbydawnGemma4E2BOwnBF16V2(DeadbydawnGemma4E2BOwnBF16):
    def __init__(self,protocol_file,protocol_sha256,protocol,npu_devices,**kwargs):
        assert protocol['model_id']==MID and npu_devices==[4]
        self.protocol_file=Path(protocol_file);self.protocol_sha256=protocol_sha256;self.devices=npu_devices

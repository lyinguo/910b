"""Retain each Gemma3 model's own processing metadata when available."""
from pathlib import Path
from universal_eval.model.adapters.gemma3_native_910b_adapter import Gemma3Native910BAdapter

class Gemma3ExistingResources910BAdapter(Gemma3Native910BAdapter):
    def __init__(self,processor_is_own=True,**kwargs):
        self.processor_is_own=bool(processor_is_own)
        super().__init__(**kwargs)

    def load(self,model_path,**kwargs):
        own=Path(model_path).resolve()/'preprocessor_config.json'
        if self.processor_is_own:
            if Path(self.processor_metadata).resolve()!=own:raise ValueError('Use the current model own processor metadata')
        elif own.is_file():raise ValueError('Existing own processor metadata must be reviewed first')
        super().load(model_path,**kwargs)
        if self.processor_is_own:
            self.lineage['processor_metadata_disclosure']='Exact current checkpoint own preprocessor_config.json, tokenizer and chat template; no other model weights, tokenizer or image metadata are used'
        else:
            self.lineage['processor_metadata_disclosure']='Current checkpoint omits preprocessor_config.json. Reuses existing local stock Gemma3 896px metadata matching this vision architecture; all model/vision/projector weights, tokenizer and chat template are from the current checkpoint. No downloads'

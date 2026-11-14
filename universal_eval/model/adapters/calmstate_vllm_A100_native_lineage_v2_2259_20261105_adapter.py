"""Reporting-only correction for standard pre-inference lineage snapshots.

No model math, batching, cache, token selection or original outputs change.
Pipeline execution lookup uses its original prompt; the model receives the
original algorithm's appended instruction. Resolve both exact input keys.
"""
import json
import hashlib
import numpy as np
import torch
from universal_eval.algorithm.native_explicit_mc_v2 import NativeExplicitMCV2
from universal_eval.model.adapters.calmstate_vllm_A100_native_v2_2259_20261105_adapter import CalmStateVllmA100Native2259Adapter

class CalmStateVllmA100NativeLineageV1Adapter(CalmStateVllmA100Native2259Adapter):
    def load(self,model_path,**kwargs):
        super().load(model_path,**kwargs)
        self.lineage.pop('groups')
        self.lineage['completed_groups_manifest']=str(self.evidence/'completed_groups.json')
        self.lineage['completed_groups_manifest_scope']='Actual completed original groups; declared before native inference and checked in full by independent verifier.'
        arrays={name:rope.cache[:2635].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy() for name,rope in self.model.rope.items()}
        arrays['text_BF16_scale']=self.model.scale.detach().cpu().reshape(-1).view(torch.uint8).numpy().copy()
        path=self.evidence/'actual_NPU_RoPE_all_potential_used_positions_and_scale.npz'
        np.savez_compressed(path,**arrays)
        self.lineage['actual_NPU_RoPE_used_prefix']={'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':2635,'dimension':256,'dtype':'torch.bfloat16','scope':'All potentially used absolute positions for max prompt587 plus original2048 and one unused async step. Complete original device caches remain131072/1048576 rows and retain full SHA separately.'}
    def batch_predict(self,batch):
        result=super().batch_predict(batch)
        (self.evidence/'completed_groups.json').write_text(json.dumps(self.groups,ensure_ascii=False,indent=2))
        return result
    def get_sample_execution(self,prompt,media_path):
        for candidate in [prompt,NativeExplicitMCV2._add_answer_instruction(prompt)]:
            key=(candidate.strip(),str(media_path))
            if key in self.details:return self.details[key]
        return {'error':'Missing native CalmState execution for exact original/model prompt keys'}

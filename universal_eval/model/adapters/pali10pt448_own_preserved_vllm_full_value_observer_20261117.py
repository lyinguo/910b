"""Add value capture to the unchanged original own Pali10pt vLLM adapter.

This untested source is not a dispatch authorization. A separate complete
independent verifier and source-bound private YAML/job must pass first.
"""
from pathlib import Path
import json,hashlib
from universal_eval.model.adapters.paligemma2_10b_pt448_A100_native_vllm_20261110_adapter import PaliGemma10BPt448Native910B,MODEL_ID,digest
from universal_eval.model.pali10pt448_full_native_value_20261117.worker import install_capture,begin_capture,finish_capture

class Pali10Pt448OwnPreservedVllmFullValues(PaliGemma10BPt448Native910B):
 def load(self,model_path,**kwargs):
  # Original load keeps the engine arguments, dtype, checkpoint verification,
  # sampling, model placement, native batching and original prompt handling.
  super().load(model_path,**kwargs)
  p=self._protocol;assert p['model_id']==MODEL_ID and p['full_numeric_observer_not_model_math_change']
  for n,h in p['full_numeric_implementation_sources_SHA256'].items():assert digest(Path.cwd()/n)==h,n
  header=Path(p['own_source_headers']);assert digest(header)==p['own_source_headers_SHA256'];headers=json.loads(header.read_bytes())
  self._numeric_home=Path(self._lineage['evidence_directory']);result=self._engine.collective_rpc(install_capture,args=(str(Path(model_path).resolve()),headers,str(self._numeric_home)));assert len(result)==1 and result[0]['installed']
  self._group_manifests=[];self._numeric_report_index=0
  self._lineage.update(full_numeric_group_index=str(self._numeric_home/'full_numeric_group_index.json'),own_actual_worker_source_inspection=result[0]['own_source_inspection'],full_numeric_runtime_sources=p['full_numeric_implementation_sources_SHA256'],all_original_engine_arguments_dtype_sampling_prompt_and_batching_inherited=True,new_full_numeric_phase_old_compatibility3_and_B10_preserved=True)
  self._save_numeric_index()
 def _save_numeric_index(self):
  f=self._numeric_home/'full_numeric_group_index.json';tmp=f.with_name(f.name+'.tmp');tmp.write_bytes(json.dumps(dict(model_id=MODEL_ID,groups=self._group_manifests,own_actual_worker_source_inspection=self._lineage['own_actual_worker_source_inspection']),ensure_ascii=False,indent=2).encode());tmp.replace(f)
 def batch_predict(self,batch):
  group=self._groups;began=self._engine.collective_rpc(begin_capture,args=(group,));assert len(began)==1 and began[0]['started']
  # Exactly one invocation of the original native chat execution. No alternate
  # generation, replay, output replacement or preprocessing is introduced.
  result=super().batch_predict(batch)
  record=self._engine.collective_rpc(finish_capture);assert len(record)==1;self._group_manifests.append(record[0]);self._save_numeric_index();return result
 def get_sample_execution(self,prompt,media_path):
  rows=[json.loads(s) for s in self._trace.read_text().splitlines()];row=rows[self._numeric_report_index];assert row['index']==self._numeric_report_index;self._numeric_report_index+=1
  return dict(original_native_request=row,full_numeric_group_manifest=self._group_manifests[row['group_index']]['path'],full_numeric_group_manifest_SHA256=self._group_manifests[row['group_index']]['SHA256'])

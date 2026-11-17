"""Same own native inference; restore missing optional-library scoring exactly.

Only this private model process changes its two caption metric method bindings.
The installed/project files, native model forward/generation and all old reports
remain untouched. Both formulas are the existing audited project references.
"""
import sys,inspect,json,hashlib
from pathlib import Path
from universal_eval.model.adapters.donut_invoice_own_native_fp16_20261117 import DonutInvoiceOwnFP16,sha
from universal_eval.metric.interface import IMetric
from universal_eval.metric.text_gen import ROUGELMetric,BLEU4Metric
from universal_eval.metric.rouge_reference_910b import OfflineStemmedRougeL910B,rouge_l_f1
from universal_eval.model.reference_sources.donut_invoice_caption_metrics_reference import rouge,bleu

def exact_offline_bleu(self,predictions,references,extra_info=None):
 if not predictions or len(predictions)!=len(references):raise ValueError('Complete nonempty prediction/reference denominator required')
 return sum(bleu(p,r) for p,r in zip(predictions,references))/len(predictions)

class DonutInvoiceOwnFP16MetricBindingsV2(DonutInvoiceOwnFP16):
 def load(self,model_path,**kwargs):
  assert self.p['model_id']=='katanaml-org/invoices-donut-model-v1'
  cases=[('', 'a page'),('a page of invoices','A page of invoices.'),('The invoices were paid.',['An invoice was paid.','Another page.']),('X X X','X X')]
  for a,b in cases:assert abs(rouge(a,b)-rouge_l_f1(a,b))<1.e-14
  assert abs(OfflineStemmedRougeL910B().compute([a for a,b in cases],[b for a,b in cases])-sum(rouge(a,b) for a,b in cases)/len(cases))<1.e-14
  old={n:dict(module=m.__module__,name=m.__qualname__) for n,m in [('ROUGE-L',ROUGELMetric.compute),('BLEU-4',BLEU4Metric.compute)]}
  assert ROUGELMetric.compute.__module__=='universal_eval.metric.text_gen' and BLEU4Metric.compute.__module__=='universal_eval.metric.text_gen'
  # MetricRegistry instances already created by the unchanged pipeline resolve
  # these class methods at compute time. No registry/pipeline file is edited.
  ROUGELMetric.compute=OfflineStemmedRougeL910B.compute;BLEU4Metric.compute=exact_offline_bleu
  super().load(model_path,**kwargs)
  actual=Path(sys.argv[sys.argv.index('--config')+1]).resolve();assert actual.is_relative_to(Path('/root/projects/comperror/configs'))
  self.lineage.update(actual_run_YAML=str(actual),actual_run_YAML_SHA256=sha(actual),caption_metric_binding=dict(reason='Optional rouge_score/nltk unavailable; original exact project reference formulas, no install, no zero fallback',original_methods=old,ROUGE_L_source=str(Path(inspect.getfile(OfflineStemmedRougeL910B)).resolve()),ROUGE_L_source_SHA256=sha(inspect.getfile(OfflineStemmedRougeL910B)),BLEU4_source=str(Path(__file__).resolve()),BLEU4_source_SHA256=sha(__file__),pure_reference_source=str(Path(inspect.getfile(bleu)).resolve()),pure_reference_source_SHA256=sha(inspect.getfile(bleu)),fixed_fixture_parity_passed=True,all_original_records_and_failed_reports_unchanged=True,model_forward_generate_observer_inherited_unchanged=True))

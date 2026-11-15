"""Exercise the real pipeline's early-lineage/late-execution lifecycle; no NN."""
import ast, hashlib, json, tempfile, traceback, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'universal_eval/model/adapters/unsloth_gemma3n_e2bit_native_v3_completed_batch_trace.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f)
tree=ast.parse(SOURCE.read_text())
owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UnslothGemma3nE2Bit')
methods=[n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in {'batch_predict','get_execution_lineage','get_sample_execution'}]
scope=dict(sha=sha,write=write,traceback=traceback)
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(SOURCE),'exec'),scope)
class BookkeepingHarness:
    batch_predict=scope['batch_predict'];get_execution_lineage=scope['get_execution_lineage'];get_sample_execution=scope['get_sample_execution']
    def __init__(self,home):
        self.home=Path(home);self.trace=self.home/'trace.jsonl';self.trace.touch()
        self.lineage=dict(final_trace_manifest=str(self.home/'completed_trace.json'))
        self.index=self.report_index=self.audio_tower_forward_calls=0
        self.outer_batches=[];self.outer_dispatches=[];self.details=[];self.failed=False
    def predict(self,prompt,media):
        self.details.append(dict(index=self.index,prompt=prompt));self.index+=1
        return prompt
class LifecycleTests(unittest.TestCase):
    def test_early_snapshot_points_to_final_real_batch_groups(self):
        with tempfile.TemporaryDirectory() as home:
            model=BookkeepingHarness(home)
            early=json.loads(json.dumps(model.get_execution_lineage()))
            model.batch_predict([(str(i),None) for i in range(10)])
            model.batch_predict([('10',None),('11',None)])
            model.get_sample_execution('0',None)
            self.assertEqual(early,model.get_execution_lineage())
            final=json.loads(Path(early['actual_batch_manifest']).read_bytes())
            self.assertEqual(final['actual_outer_batch_sizes'],[10,2])
            self.assertEqual(final['actual_outer_batch_indices'],[list(range(10)),[10,11]])
            self.assertEqual(final['samples'],12)
    def test_failed_dispatch_never_becomes_a_completed_group(self):
        with tempfile.TemporaryDirectory() as home:
            model=BookkeepingHarness(home)
            def fail(prompt,media):raise RuntimeError('intentional bookkeeping test')
            model.predict=fail
            with self.assertRaises(RuntimeError):model.batch_predict([('0',None)])
            self.assertEqual(model.outer_dispatches[0]['status'],'failed')
            self.assertEqual(model.outer_dispatches[0]['committed_count'],0)
if __name__=='__main__':unittest.main()

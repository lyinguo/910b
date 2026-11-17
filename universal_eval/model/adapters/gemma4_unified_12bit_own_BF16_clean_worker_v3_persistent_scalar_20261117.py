"""CLI adapter transporting unchanged requests to a genuine fresh worker.

The original CLI has already imported legacy compatibility initializers.
This adapter performs no learned operation in that parent interpreter.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,hashlib,subprocess,traceback
from universal_eval.model.interface import IModel
R=Path('/root/projects/comperror');MID='google/gemma-4-12B-it'

class OwnUnified12BitCleanWorker(IModel):
    def __init__(self,**kwargs):
        self.kwargs=kwargs;stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f');self.rpc_home=R/'workspace/diagnostics/model_runs'/('Gemma4Unified12Bit_clean_worker_RPC_'+stamp+'_'+str(os.getpid()));self.rpc_home.mkdir(exist_ok=False);self.child=None;self.sequence=0
    @property
    def model_name(self):return MID
    @property
    def model_type(self):return 'Gemma4UnifiedForConditionalGeneration'
    @property
    def inference_framework(self):return 'Genuine installed Transformers Unified12B / isolated fresh learned NPU worker'
    def download(self,*a,**kw):raise RuntimeError('Downloads forbidden')
    def _call(self,command,**kwargs):
        request=dict(command=command,**kwargs);self.sequence+=1
        (self.rpc_home/f'{self.sequence:04}_{command}_request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
        assert self.child is not None and self.child.poll() is None,'Learned worker exited; never rerun completed requests'
        self.child.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.child.stdin.flush();line=self.child.stdout.readline()
        if not line:
            rc=self.child.wait();raise RuntimeError(('Genuine learned worker exited without response',rc))
        reply=json.loads(line);(self.rpc_home/f'{self.sequence:04}_{command}_response.json').write_text(json.dumps(reply,ensure_ascii=False),encoding='utf-8')
        if not reply['ok']:
            self.child.wait();raise RuntimeError(reply['error'])
        return reply['result']
    def load(self,model_path,**kwargs):
        assert self.child is None;worker=R/'scripts/gemma4_unified12bit_genuine_clean_worker_v3_persistent_scalar_20261117.py'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        self.stderr=(self.rpc_home/'worker_stderr.log').open('x',encoding='utf-8')
        self.child=subprocess.Popen([sys.executable,'-B',str(worker)],cwd=R,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',bufsize=1)
        self._call('load',model_path=model_path,kwargs=self.kwargs)
    def predict(self,prompt,media_path=None):return self.batch_predict([(prompt,media_path)])[0]
    def batch_predict(self,batch):return self._call('batch',batch=batch)
    def get_execution_lineage(self):return self._call('lineage')
    def get_sample_execution(self,prompt,media_path):return self._call('sample',prompt=prompt,media_path=media_path)
    def unload(self):
        if self.child is not None and self.child.poll() is None:self._call('unload');assert self.child.wait()==0
        if hasattr(self,'stderr'):self.stderr.close()

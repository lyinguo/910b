"""Own Merve named RPC observers with unchanged original resource guards."""
from pathlib import Path
import shutil
class OwnMerveFullValueWorkerExtension:
 def own_pali_install_full_value_capture(self,root,headers,home):
  from universal_eval.model.merve_pali_full_native_value_20261117.worker import install_capture
  assert shutil.disk_usage(Path(home)).free>3*1024**3+1024**3+512*1024**2
  return install_capture(self,root,headers,home)
 def own_pali_begin_full_value_capture(self,group,count):
  from universal_eval.model.merve_pali_full_native_value_20261117.worker import begin_capture
  observer=self._own_pali10pt_full_capture;assert 1<=count<=10
  assert shutil.disk_usage(observer.home).free>3*1024**3+1024**3+512*1024**2+count*2*1024**3
  return begin_capture(self,group)
 def own_pali_finish_full_value_capture(self):
  from universal_eval.model.merve_pali_full_native_value_20261117.worker import finish_capture
  observer=self._own_pali10pt_full_capture;exact=sum(a.nbytes for a in observer.arrays.values());assert shutil.disk_usage(observer.home).free>3*1024**3+1024**3+512*1024**2+exact+1024*1024
  return finish_capture(self)

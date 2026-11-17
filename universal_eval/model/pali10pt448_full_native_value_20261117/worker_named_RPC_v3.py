"""Only named observation methods are added to the existing native worker."""
from pathlib import Path
import shutil
class OwnPaliFullValueWorkerExtension:
    def own_pali_install_full_value_capture(self,root,headers,home):
        from universal_eval.model.pali10pt448_full_native_value_20261117.worker_v2_ABI import install_capture
        assert shutil.disk_usage(Path(home)).free > 3*1024**3+1024**3+512*1024**2
        return install_capture(self,root,headers,home)
    def own_pali_begin_full_value_capture(self,group):
        from universal_eval.model.pali10pt448_full_native_value_20261117.worker_v2_ABI import begin_capture
        observer=self._own_pali10pt_full_capture
        assert shutil.disk_usage(observer.home).free > 3*1024**3+1024**3+512*1024**2+3*2*1024**3
        return begin_capture(self,group)
    def own_pali_finish_full_value_capture(self):
        from universal_eval.model.pali10pt448_full_native_value_20261117.worker_v2_ABI import finish_capture
        observer=self._own_pali10pt_full_capture
        exact_uncompressed_bytes=sum(a.nbytes for a in observer.arrays.values())
        assert shutil.disk_usage(observer.home).free > 3*1024**3+1024**3+512*1024**2+exact_uncompressed_bytes+1024*1024
        return finish_capture(self)

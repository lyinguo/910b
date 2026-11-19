"""Complete codec round trips, occurrence binding and corruption refusals."""
from pathlib import Path
import importlib.util,json,zipfile,os
import numpy as np
here=Path(__file__).resolve().parent
utility=here/'complete_bitplane_shared_npz_20261119.py'
if not utility.exists():utility=here.parent.parent/'universal_eval/utils/complete_bitplane_shared_npz_20261119.py'
spec=importlib.util.spec_from_file_location('reviewed_complete_bitplanes',utility);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 base=Path(os.environ.get('CODEC_SCOPE_TEST_HOME',str(here if here.name!='tests' else here.parent.parent/'workspace/diagnostics')))
 h=base/('bitplane_complete_codec_checks_pid_'+str(os.getpid()));h.mkdir(parents=True,exist_ok=False);arrays=dict(scalar=np.array(7,dtype='<i8'),zero=np.empty((3,0,2),dtype='<f4'),bf16bits=np.array([0,1,65535,16256,32768],dtype='<u2'),float32=np.array([0.,-0.,1.25,np.inf,-np.inf,np.nan],dtype='<f4'),noncontiguous=np.arange(240,dtype='<i2').reshape(10,24).T[::2],boolean=np.array([True,False,True]),bigendian=np.array([1,256,513],dtype='>u2'))
 arrays['duplicate']=arrays['noncontiguous'].copy();arrays['same_bytes_different_shape']=arrays['bf16bits'].reshape(1,5)
 p=h/'full.npz';r=m.write_complete_bitplane_npz(p,arrays);assert r['logical_arrays']==9 and r['physical_complete_arrays']==8
 with m.CompleteBitplaneNPZ(p) as z:
  assert set(z.files)==set(arrays)
  for n,a in arrays.items():
   b=z[n];assert a.dtype==b.dtype and a.shape==b.shape and m.raw_metadata(a)==m.raw_metadata(b),n
 # Object payloads and preexisting targets are refused.
 refused=[]
 for label,call in [('object',lambda:m.write_complete_bitplane_npz(h/'bad.npz',dict(x=np.array([object()],dtype=object)))),('overwrite',lambda:m.write_complete_bitplane_npz(p,arrays))]:
  try:call()
  except (AssertionError,ValueError):refused.append(label)
  else:raise AssertionError(label)
 with zipfile.ZipFile(p) as src:
  members={n:src.read(n) for n in src.namelist()}
 manifest=json.loads(members[m.META]);manifest['original_occurrences']['scalar']['raw_C_SHA256']='0'*64
 members[m.META]=json.dumps(manifest).encode();tampered=h/'tampered.npz'
 with zipfile.ZipFile(tampered,'x') as dst:
  for n,b in members.items():dst.writestr(n,b)
 try:
  with m.CompleteBitplaneNPZ(tampered) as z:z['scalar']
 except AssertionError:refused.append('raw_hash_tamper')
 else:raise AssertionError('tamper accepted')
 assert len(refused)==3
 print(json.dumps(dict(passed=True,complete_logical_arrays_roundtripped=9,physical_arrays=8,scalar_empty_NaN_infinity_signedzero_endian_noncontiguous_preserved=True,distinct_shape_not_deduplicated=True,exact_duplicate_checked=True,refusals=refused)))
if __name__=='__main__':main()

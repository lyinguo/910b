import tempfile
from pathlib import Path
import unittest
import numpy as np
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ


class ReaderTests(unittest.TestCase):
    def test_exact_bytes_dtype_shape_and_order(self):
        arrays = {'bf16_bits':np.arange(32, dtype=np.uint16).reshape(4,8),
                  'f32':np.array([float('nan'),float('inf'),-0.0],dtype=np.float32),
                  'bool':np.array([True,False]), 'scalar':np.array(42),
                  'fortran':np.asfortranarray(np.arange(12).reshape(3,4)),
                  'big_endian':np.arange(3,dtype='>i8')}
        with tempfile.TemporaryDirectory() as d:
            for writer in (np.savez, np.savez_compressed):
                p=Path(d)/'arrays.npz';writer(p, **arrays)
                with np.load(p,allow_pickle=False) as old, ReadOnlyNPZ(p) as new:
                    self.assertEqual(old.files,new.files)
                    for name in old.files:
                        a,b=old[name],new[name]
                        self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape)
                        self.assertEqual(a.tobytes(order='A'),b.tobytes(order='A'))
                        self.assertFalse(b.flags.writeable)
                        self.assertIs(b,new[name+'.npy'])

    def test_bounded_cache_eviction_and_closed_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';np.savez(p,a=np.arange(8,dtype=np.uint8),b=np.arange(8,dtype=np.uint8),large=np.arange(100))
            with ReadOnlyNPZ(p,max_bytes=8,max_entries=1) as r:
                a=r['a'];self.assertIs(a,r['a']);r['b'];r['large']
                self.assertLessEqual(r._stats['max_cached_bytes'],8)
                self.assertGreater(r._stats['evictions'],0)
                with self.assertRaises(KeyError):r['missing']
                with self.assertRaises(ValueError):a[0]=99
            with self.assertRaises(ValueError):r['a']

    def test_object_arrays_and_pickle_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';np.savez(p,a=np.array([object()],dtype=object))
            with self.assertRaises(ValueError):ReadOnlyNPZ(p,allow_pickle=True)
            with ReadOnlyNPZ(p) as r:
                with self.assertRaises(ValueError):r['a']


if __name__=='__main__':unittest.main()

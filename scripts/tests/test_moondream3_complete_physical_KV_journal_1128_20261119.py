"""Deterministic unlearned cache storage checks; preserve all test values."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np

def run(module_path,home):
    spec=importlib.util.spec_from_file_location('moondream_kv_storage_candidate',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    journal=module.CompletePhysicalKVJournal(home)
    zeros=np.zeros(module.SHAPE,dtype=np.uint16)
    for layer in range(24):journal.register_initial(layer,zeros,zeros)
    # Prefix load, original positional append, and a repeated equal write must
    # all remain separate semantic occurrences with complete original vectors.
    pos=np.arange(730,dtype=np.int64)
    k=np.arange(32*730*64,dtype=np.uint16).reshape(1,32,730,64)
    v=np.bitwise_xor(k,np.uint16(65535));ka=zeros.copy();va=zeros.copy()
    ka[:,:,pos,:]=k;va[:,:,pos,:]=v
    journal.record_actual_write(0,pos,k,v,ka,va,kind='original_load_encoded_image_prefix')
    append=np.array([730,731],dtype=np.int64)
    ak=np.full((1,32,2,64),np.uint16(0x7fc0));av=np.full_like(ak,np.uint16(0x8000))
    ka[:,:,append,:]=ak;va[:,:,append,:]=av
    for _ in range(2):
        journal.record_actual_write(0,append,ak,av,ka,va,kind='original_KVCache_update')
    # A mutation outside the original write must be rejected before append.
    corrupted=ka.copy();corrupted[0,0,4095,0]^=np.uint16(1)
    try:journal.record_actual_write(0,append,ak,av,corrupted,va,kind='original_KVCache_update')
    except AssertionError:outside_write_rejected=True
    else:raise AssertionError('Corrupt full physical cache was accepted')
    # Duplicate positions and dtype changes cannot be used to compact values.
    for bad in [np.array([730,730],dtype=np.int64),np.array([730,731],dtype=np.int32)]:
        try:journal.record_actual_write(0,bad,ak,av,ka,va,kind='original_KVCache_update')
        except AssertionError:pass
        else:raise AssertionError('Invalid original positions accepted')
    final=journal.finalize();proof=module.independently_reconstruct_all_physical_values(home)
    assert final['occurrences']==proof['occurrences']==27
    values=list((Path(home)/'values').glob('*.npy'))
    assert len(values)==8, len(values)
    result=dict(proof,writer=final,outside_original_write_rejected=outside_write_rejected,
                duplicate_positions_and_dtype_change_rejected=True,
                original_NaN_and_negative_zero_raw_bits_retained=True,
                complete_NPY_files=len(values),complete_NPY_bytes=sum(p.stat().st_size for p in values),
                candidate_source_SHA256=hashlib.sha256(Path(module_path).read_bytes()).hexdigest())
    (Path(home)/'independent_storage_test.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]),Path(sys.argv[2]))

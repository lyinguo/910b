"""Lossless observation of the original 24 physical BF16 KV caches.

This utility does not run a model or change its cache. Callers supply complete
copies taken from the actual device, plus the actual original write operands.
Each occurrence remains in the journal, even when an equal complete NPY is
shared. Independent replay reconstructs every physical value after each write.
It is a candidate observation component, not a model numerical verifier.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np

SHAPE=(1,32,4096,64)

def value_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()

def file_sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def _physical(a):
    a=np.asarray(a)
    assert a.dtype==np.uint16 and a.shape==SHAPE
    return a.copy(order='C')

def _positions(a):
    a=np.asarray(a)
    assert a.dtype==np.int64 and a.ndim==1 and len(a)>0
    assert np.all((a>=0)&(a<SHAPE[2]))
    assert len(np.unique(a))==len(a)
    return a.copy()

class CompletePhysicalKVJournal:
    """Keep all original values; prove each compact representation lossless."""
    def __init__(self,home):
        self.home=Path(home);self.home.mkdir(exist_ok=False,parents=True)
        self.values=self.home/'values';self.values.mkdir()
        self.path=self.home/'all_original_KV_occurrences.jsonl'
        self.previous={};self.events=0

    def _save(self,a):
        a=np.ascontiguousarray(a)
        digest=value_sha(a)
        identity=hashlib.sha256((a.dtype.str+repr(a.shape)+digest).encode()).hexdigest()
        p=self.values/(identity+'.npy')
        if p.exists():
            old=np.load(p,allow_pickle=False)
            assert old.dtype==a.dtype and old.shape==a.shape
            assert np.array_equal(old.view(np.uint8),a.view(np.uint8))
        else:
            tmp=p.with_suffix('.npy.tmp')
            with tmp.open('xb') as f:np.save(f,a,allow_pickle=False)
            os.replace(tmp,p)
        return dict(array=p.relative_to(self.home).as_posix(),shape=list(a.shape),
                    dtype=a.dtype.str,raw_SHA256=digest,file_SHA256=file_sha(p))

    def _append(self,event):
        event=dict(event,occurrence=self.events)
        with self.path.open('a',encoding='utf8',newline='\n') as f:
            f.write(json.dumps(event,sort_keys=True,separators=(',',':'))+'\n')
            f.flush()
        self.events+=1
        return event

    def register_initial(self,layer,k,v):
        assert isinstance(layer,int) and 0<=layer<24 and layer not in self.previous
        k,v=_physical(k),_physical(v)
        event=self._append(dict(kind='initial_complete_physical_values',layer=layer,
                                k=self._save(k),v=self._save(v),
                                complete_K_SHA256=value_sha(k),complete_V_SHA256=value_sha(v)))
        self.previous[layer]=(k,v)
        return event

    def record_actual_write(self,layer,positions,k_operand,v_operand,k_actual,v_actual,*,kind):
        assert kind in ['original_KVCache_update','original_load_encoded_image_prefix']
        assert layer in self.previous
        positions=_positions(positions)
        if kind=='original_load_encoded_image_prefix':
            assert np.array_equal(positions,np.arange(len(positions),dtype=np.int64))
        k_actual,v_actual=_physical(k_actual),_physical(v_actual)
        operands=[]
        for a in [k_operand,v_operand]:
            a=np.asarray(a)
            assert a.dtype==np.uint16 and a.shape==(1,32,len(positions),64)
            operands.append(a.copy(order='C'))
        previous=self.previous[layer]
        expected=[]
        for before,operand,actual in zip(previous,operands,[k_actual,v_actual]):
            rebuilt=before.copy();rebuilt[:,:,positions,:]=operand
            assert np.array_equal(rebuilt.view(np.uint8),actual.view(np.uint8)), \
                'Actual full cache differs inside or outside the original write'
            expected.append(rebuilt)
        event=self._append(dict(kind=kind,layer=layer,positions=self._save(positions),
                                k_operand=self._save(operands[0]),v_operand=self._save(operands[1]),
                                prior_K_SHA256=value_sha(previous[0]),prior_V_SHA256=value_sha(previous[1]),
                                complete_K_SHA256=value_sha(k_actual),complete_V_SHA256=value_sha(v_actual),
                                actual_full_values_compared_before_compact_storage=True))
        self.previous[layer]=(k_actual,v_actual)
        return event

    def finalize(self):
        assert set(self.previous)==set(range(24))
        return dict(complete_physical_KV_layers=24,physical_shape=list(SHAPE),
                    storage_dtype='uint16 BF16 raw bits',occurrences=self.events,
                    journal_SHA256=file_sha(self.path),all_occurrences_preserved=True,
                    scope='storage observation only; not model inference or numerical acceptance')

def independently_reconstruct_all_physical_values(home):
    """Read the journal afresh; never consult the writer's cached state."""
    home=Path(home).resolve();current={};visited=set();occurrences=0
    def load(rec):
        p=(home/rec['array']).resolve()
        assert p.is_relative_to(home/'values') and p.suffix=='.npy'
        assert file_sha(p)==rec['file_SHA256']
        a=np.load(p,allow_pickle=False)
        assert list(a.shape)==rec['shape'] and a.dtype.str==rec['dtype']
        assert value_sha(a)==rec['raw_SHA256']
        visited.add(p);return a.copy()
    with (home/'all_original_KV_occurrences.jsonl').open(encoding='utf8') as f:
        for line in f:
            event=json.loads(line);assert event['occurrence']==occurrences
            layer=event['layer'];assert isinstance(layer,int) and 0<=layer<24
            if event['kind']=='initial_complete_physical_values':
                assert layer not in current
                current[layer]=(_physical(load(event['k'])),_physical(load(event['v'])))
            else:
                assert event['kind'] in ['original_KVCache_update','original_load_encoded_image_prefix']
                assert event['actual_full_values_compared_before_compact_storage'] is True
                k,v=current[layer]
                assert value_sha(k)==event['prior_K_SHA256'] and value_sha(v)==event['prior_V_SHA256']
                pos=_positions(load(event['positions']))
                if event['kind']=='original_load_encoded_image_prefix':
                    assert np.array_equal(pos,np.arange(len(pos),dtype=np.int64))
                for target,key in [(k,'k_operand'),(v,'v_operand')]:
                    operand=load(event[key]);assert operand.dtype==np.uint16
                    assert operand.shape==(1,32,len(pos),64)
                    target[:,:,pos,:]=operand
            k,v=current[layer]
            assert value_sha(k)==event['complete_K_SHA256']
            assert value_sha(v)==event['complete_V_SHA256']
            occurrences+=1
    assert set(current)==set(range(24))
    assert visited==set((home/'values').glob('*.npy'))
    return dict(passed=True,occurrences=occurrences,complete_physical_layers=24,
                every_array_visited=True,all_full_values_reconstructed=True,
                actual_model_forwards=0,scope='independent storage test only')

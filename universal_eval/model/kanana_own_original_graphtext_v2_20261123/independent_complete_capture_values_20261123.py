"""Independently read every complete native observation value and reference.

This validates data completeness, never model arithmetic or a model score.
It consumes standard NPZ/NPY values and does not import the producer journal.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
import torch


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def raw_bytes(array):
    return np.array(array,copy=True,order='C').reshape(-1).view(np.uint8).tobytes()


class IndependentCompleteCapture:
    def __init__(self,manifest_path):
        self.path=Path(manifest_path).resolve()
        boundary=Path('/root/projects/comperror/workspace/diagnostics').resolve()
        if not self.path.is_relative_to(boundary):raise ValueError('Only the exact own diagnostic capture is accepted')
        before=self.path.read_bytes();self.manifest=json.loads(before)
        if self.manifest['schema']!='kanana_complete_native_values_v1':raise ValueError('Unknown complete-value schema')
        archive=Path(self.manifest['full_value_file']).resolve()
        if archive.parent!=self.path.parent or archive.name!=self.path.with_suffix('.npz').name:
            raise ValueError('Manifest must refer to its own exact standard value archive')
        if file_sha(archive)!=self.manifest['full_value_file_SHA256']:raise ValueError('Full value archive SHA changed')
        self.values={}
        with np.load(archive,allow_pickle=False) as data:
            if set(data.files)!=set(self.manifest['tensors']):raise ValueError('A complete array is missing or unreferenced')
            for key,descriptor in self.manifest['tensors'].items():
                value=data[key]
                if value.dtype.hasobject or list(value.shape)!=descriptor['shape'] or value.dtype.str!=descriptor['storage_dtype']:
                    raise ValueError('Complete stored shape or dtype changed: '+key)
                raw=raw_bytes(value)
                if len(raw)!=descriptor['bytes'] or hashlib.sha256(raw).hexdigest()!=descriptor['raw_C_order_SHA256']:
                    raise ValueError('Complete raw values changed: '+key)
                encoded=json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()
                if key!='tensor_'+hashlib.sha256(encoded).hexdigest():raise ValueError('Logical dtype/shape/raw binding changed')
                self.values[key]=np.array(value,copy=True,order='C')
        if self.path.read_bytes()!=before:raise RuntimeError('Manifest changed concurrently')
        self.reference_counts=Counter()
        for event in self.manifest['events']:
            self._audit_tree(event['value'])
        if set(self.reference_counts)!=set(self.values):raise ValueError('A full array has no semantic occurrence')
        count=sum(self.reference_counts.values())
        if count!=self.manifest['tensor_occurrences']:raise ValueError('Tensor occurrence denominator changed')
        if len(self.values)!=self.manifest['unique_complete_arrays']:raise ValueError('Unique full-array count changed')
        if count-len(self.values)!=self.manifest['full_raw_compares_for_shared_occurrences']:
            raise ValueError('Full raw-comparison occurrence count changed')
        if [e['event_index'] for e in self.manifest['events']]!=list(range(len(self.manifest['events']))):
            raise ValueError('Original event order or denominator changed')

    def _audit_tree(self,node):
        if node is None or isinstance(node,(str,bool,int,float)):return
        if not isinstance(node,dict):raise TypeError('Unknown serialized native value')
        kind=node.get('kind')
        if kind=='tensor':
            key=node['key']
            if key not in self.values:raise ValueError('Missing complete tensor reference')
            if {k:v for k,v in node.items() if k not in ['kind','key']}!=self.manifest['tensors'][key]:
                raise ValueError('Semantic occurrence dtype/shape/raw reference changed')
            self.reference_counts[key]+=1
        elif kind in ['mapping','native_cache_state']:
            for child in node['values' if kind=='mapping' else 'attributes'].values():self._audit_tree(child)
        elif kind in ['tuple','list']:
            for child in node['values']:self._audit_tree(child)
        elif kind in ['torch.dtype','torch.device','class']:
            if not isinstance(node.get('value'),str):raise TypeError('Missing original native type metadata')
        else:raise TypeError('Unknown serialized native kind: '+str(kind))

    def tensor(self,reference):
        key=reference['key']
        if {k:v for k,v in reference.items() if k not in ['kind','key']}!=self.manifest['tensors'][key]:
            raise ValueError('A dtype or shape alias cannot be silently substituted')
        array=self.values[key].copy(order='C')
        logical=reference['dtype']
        if logical=='torch.bfloat16':
            if array.dtype!=np.dtype('uint16'):raise TypeError('BF16 must retain all uint16 source bits')
            result=torch.from_numpy(array).view(torch.bfloat16)
        else:
            result=torch.from_numpy(array)
            if str(result.dtype)!=logical:raise TypeError('Stored dtype does not represent the original logical dtype')
        if list(result.shape)!=reference['shape']:raise ValueError('Logical tensor rank was not retained')
        return result

    def decode(self,node):
        if node is None or isinstance(node,(str,bool,int,float)):return node
        kind=node['kind']
        if kind=='tensor':return self.tensor(node)
        if kind=='mapping':return {k:self.decode(v) for k,v in node['values'].items()}
        if kind in ['tuple','list']:
            values=[self.decode(v) for v in node['values']]
            return tuple(values) if kind=='tuple' else values
        if kind=='native_cache_state':
            # Return explicit data, never a surrogate runtime cache/model class.
            return dict(kind=kind,class_name=node['class_name'],attributes={k:self.decode(v) for k,v in node['attributes'].items()})
        if kind in ['torch.dtype','torch.device','class']:return dict(node)
        raise TypeError('Unknown native kind')

    def decoded_events(self):
        return [dict(event_index=e['event_index'],name=e['name'],phase=e['phase'],call_id=e['call_id'],value=self.decode(e['value'])) for e in self.manifest['events']]

    def summary(self):
        return dict(manifest=str(self.path),manifest_SHA256=file_sha(self.path),sample_index=self.manifest['sample_index'],
                    unique_complete_arrays=len(self.values),tensor_occurrences=sum(self.reference_counts.values()),
                    all_complete_arrays_and_semantic_occurrences_independently_reconstructed=True,
                    not_a_model_numerical_or_score_acceptance=True)

"""Read-only, bounded NPZ access with direct ZIP indexing and NumPy decoding.

Array bytes, dtype and shape are decoded by NumPy's public format.read_array.
This reader avoids repeated linear membership scans of large NPZ name lists.
It is not wired into any running verifier or model.
"""
from collections import OrderedDict
from pathlib import Path
import zipfile
import numpy as np

IO_READ_CACHE_STATS = []


class ReadOnlyNPZ:
    def __init__(self, path, allow_pickle=False, max_bytes=32*1024**2, max_entries=64):
        if allow_pickle is not False or max_bytes < 0 or max_entries < 1:
            raise ValueError('Pickle is forbidden; cache bounds must be valid')
        self._zip = zipfile.ZipFile(Path(path), 'r')
        names = self._zip.namelist()
        if len(set(names)) != len(names) or any(not n.endswith('.npy') for n in names):
            self._zip.close()
            raise ValueError('Expected unique NumPy array members only')
        self.files = [n[:-4] for n in names]
        self._members = dict(zip(self.files, names))
        self._members.update({n:n for n in names})
        self._cache = OrderedDict()
        self._bytes = 0
        self._limit, self._entries = int(max_bytes), int(max_entries)
        self._closed = False
        self._stats = dict(path=str(path), calls=0, decode_calls=0, cache_hits=0,
                           evictions=0, max_cached_bytes=0, max_cached_arrays=0,
                           bytes_limit=self._limit, entries_limit=self._entries,
                           logical_bytes_returned=0, actual_bytes_decoded=0,
                           indexing='ZIP dictionary; no linear filename membership')

    def __enter__(self):
        if self._closed:
            raise ValueError('Reader is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._closed:
            return
        assert self._bytes <= self._limit and len(self._cache) <= self._entries
        self._stats.update(final_cached_bytes=self._bytes, final_cached_arrays=len(self._cache))
        IO_READ_CACHE_STATS.append(dict(self._stats))
        self._cache.clear()
        self._bytes = 0
        self._zip.close()
        self._closed = True

    def __getitem__(self, key):
        if self._closed:
            raise ValueError('Reader is closed')
        member = self._members[key]
        self._stats['calls'] += 1
        if member in self._cache:
            value = self._cache.pop(member)
            self._cache[member] = value
            self._stats['cache_hits'] += 1
        else:
            with self._zip.open(member, 'r') as stream:
                value = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=10000)
            if not isinstance(value, np.ndarray):
                raise ValueError('Only NumPy arrays are supported')
            value.setflags(write=False)
            self._stats['decode_calls'] += 1
            self._stats['actual_bytes_decoded'] += value.nbytes
            if value.nbytes <= self._limit:
                while self._cache and (self._bytes + value.nbytes > self._limit or len(self._cache) >= self._entries):
                    _, expired = self._cache.popitem(last=False)
                    self._bytes -= expired.nbytes
                    self._stats['evictions'] += 1
                self._cache[member] = value
                self._bytes += value.nbytes
                self._stats['max_cached_bytes'] = max(self._stats['max_cached_bytes'], self._bytes)
                self._stats['max_cached_arrays'] = max(self._stats['max_cached_arrays'], len(self._cache))
        assert not value.flags.writeable and self._bytes <= self._limit and len(self._cache) <= self._entries
        self._stats['logical_bytes_returned'] += value.nbytes
        return value

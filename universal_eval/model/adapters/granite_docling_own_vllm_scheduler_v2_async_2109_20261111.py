# SPDX-License-Identifier: Apache-2.0
# EncoderCacheManager below is frozen from the actual A100 installed source.
from __future__ import annotations
from collections import OrderedDict
ORIGINAL_A100_ENCODER_CACHE_SOURCE_SHA256='04c2c303859d47933572901e32ee5499fc504d7e5e4cf714dfa70771affb96ce'
ORIGINAL_ENCODER_CLASS_SHA256='26586febb012fbf532de18d2f054146043d32edbda17e9b280db61880d6f15db'

class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data
    item (identified by their hash value) between different requests,
    and eviction takes place at allocation time when there's no free
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.

    NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
    instead of encoder tokens (i.e. all tokens that represent the multimodal data
    in the input sequence). This means all break/text tokens in-between multimodal
    embeddings are not considered with respect to the cache size and the number
    of free slots.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    encoder embeddings from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder embeddings.
        num_free_slots: Current available cache capacity in encoder embeddings.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder embeddings).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_encoder_embeds) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}
        # request_id => set of input_ids cached for that request
        self.request_cached_ids: dict[str, set[int]] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def reset(self) -> None:
        """Reset the encoder cache to its initial state.

        This clears all cached encoder outputs and resets capacity tracking.
        Called when model weights are updated to invalidate stale embeddings.
        """
        self.cached.clear()
        self.request_cached_ids.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        self.request_cached_ids.setdefault(request.request_id, set()).add(input_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True.
        Older entries are evicted first.

        Returns False only if the requested number of tokens exceeds both
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder embeddings allowed to be
                computed when this method is invoked.
            num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder
        output but only the state of EncoderCacheManager.
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # Enough free slots
        if num_embeds <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_embeds > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.request_cached_ids.setdefault(request_id, set()).add(input_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request."""
        return self.request_cached_ids.get(request.request_id, set())

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder embeddings for that input.

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # Always clean up request_cached_ids, even if the mm_hash was
        # already evicted from cache (e.g. by can_allocate).
        if req_id in self.request_cached_ids:
            self.request_cached_ids[req_id].discard(input_id)
            if not self.request_cached_ids[req_id]:
                del self.request_cached_ids[req_id]
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.
        The data stays in memory until eviction is triggered by a future
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        for input_id in list(self.get_cached_input_ids(request)):
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which
            encoder outputs can be removed from their caches. The internal
            list is cleared after this call.
        """
        freed = self.freed
        self.freed = []
        return freed

from collections import deque
from dataclasses import dataclass,field
from types import SimpleNamespace
import hashlib,struct

@dataclass
class QualifiedRequest:
    request_id:str
    input_index:int
    prompt_ids:list
    mm_hash:str
    image_offset:int
    image_length:int
    image_embeds:int
    computed:int=0
    output_ids:list=field(default_factory=list)
    finish_reason:str|None=None
    finish_wave:int|None=None
    retire_wave:int|None=None
    admitted:bool=False
    prefix_count:int=0
    prefix_source:str|None=None
    def __post_init__(self):self.mm_features=[SimpleNamespace(identifier=self.mm_hash)]
    def get_num_encoder_embeds(self,index):assert index==0;return self.image_embeds

class OriginalGraniteDoclingScheduler:
    """Logical reference of the original TP1/async/FCFS scheduling scope.

    Uses the authentic frozen EncoderCacheManager. No A100 answers or fixed
    output-dependent plans are accepted. The caller supplies actual own NPU
    argmax IDs. This is not CUDA paged allocation or acceleration.
    """
    def __init__(self,encoder_manager=None,prefix_blocks=None):
        self.encoder=encoder_manager or EncoderCacheManager(8192)
        self.prefix_blocks={} if prefix_blocks is None else prefix_blocks
        self.wave=0;self.running=[];self.waiting=deque();self.requests={};self.previous=[]
        self.current=[];self.new_encoder=[];self.freed_encoder=[];self.retired=[]
    def add_batch(self,records):
        assert not self.running and not self.waiting and not self.requests
        assert 1<=len(records)<=10
        self.wave=0;self.previous=[]
        for row in records:
            rid=str(row['index']);assert rid not in self.requests
            r=QualifiedRequest(rid,row['index'],list(row['ids']),row['mm_hash'],row['placeholder_offset'],row['placeholder_length'],len(row['image_soft_positions']))
            assert 0<r.image_offset<r.image_offset+r.image_length<len(r.prompt_ids)-4
            assert 0<r.image_embeds<=17*64 and len(r.prompt_ids)+2048<=8192
            self.requests[rid]=r;self.waiting.append(r)
    def add_available_records(self,records):
        """Real frontend FCFS arrivals, independent of output or an A100 trace."""
        for row in records:
            rid=str(row['index']);assert rid not in self.requests
            r=QualifiedRequest(rid,row['index'],list(row['ids']),row['mm_hash'],row['placeholder_offset'],row['placeholder_length'],len(row['image_soft_positions']))
            assert 0<r.image_embeds<=17*64 and len(r.prompt_ids)+2048<=8192
            self.requests[rid]=r;self.waiting.append(r)
    def _block_keys(self,r,tokens=None):
        ids=r.prompt_ids if tokens is None else tokens;parent=b'Original_GraniteDocling_block_equivalence_v1';out=[]
        for start in range(0,len(ids)//16*16,16):
            payload=struct.pack('<16I',*ids[start:start+16])
            if start<r.image_offset+r.image_length and start+16>r.image_offset:payload+=r.mm_hash.encode()
            parent=hashlib.sha256(parent+payload).digest();out.append(parent.hex())
        return out
    def _initial_prefix(self,r):
        count=0;source=None
        for key in self._block_keys(r):
            if count+16>len(r.prompt_ids)-1 or key not in self.prefix_blocks:break
            prior=self.prefix_blocks[key];assert prior['request_id'] in self.requests,'Unqualified inter-batch KV prefix hit'
            count+=16;source=prior['request_id']
        return count,source
    def _cache_blocks(self,r,end):
        for key in self._block_keys(r,r.prompt_ids+r.output_ids)[:end//16]:
            self.prefix_blocks.setdefault(key,{'request_id':r.request_id})
    def _encoder_schedule(self,r,count,budget):
        start=r.computed;end=start+count;lo=r.image_offset;hi=lo+r.image_length
        if count==0 or start>=hi or end<=lo:return count,budget,False
        if self.encoder.check_and_update_cache(r,0):return count,budget,False
        if not self.encoder.can_allocate(r,0,budget,0):
            return (max(0,lo-start) if start<lo else 0),budget,False
        # The qualified multimodal interval may include fake/row/newline text.
        # Native vLLM does not schedule an encoder if a chunk stops before its
        # first actual image embedding. For these own prompts first soft ID is
        # two tokens after the offset; preserve that exact boundary.
        first_soft=r.image_offset+2
        if end<=first_soft:return count,budget,False
        self.encoder.allocate(r,0)
        return count,budget-r.image_embeds,True
    def schedule(self):
        self.wave+=1;self.current=[];self.new_encoder=[];self.retired=[];budget=8192;encoder_budget=8192
        for r in list(self.running):
            if r.retire_wave is not None and r.retire_wave<=self.wave:
                self.running.remove(r);self.encoder.free(r);self.retired.append(r.request_id)
        # FCFS running requests precede newly admitted requests. The GPU runner
        # later stable-sorts by scheduled query length, not by prompt size.
        for r in list(self.running):
            if budget<=0:break
            count=min(len(r.prompt_ids)+len(r.output_ids)-r.computed,budget,8192-r.computed-1)
            if count<=0:continue
            count,new_budget,new_encoder=self._encoder_schedule(r,count,encoder_budget)
            if count<=0:continue
            start=r.computed;self.current.append({'id':r.request_id,'index':r.input_index,'start':start,'count':count,'prefix_source':None,'prefix_count':0})
            if new_encoder:self.new_encoder.append(r.request_id)
            encoder_budget=new_budget;budget-=count;r.computed+=count;self._cache_blocks(r,r.computed)
        while self.waiting and budget>0:
            r=self.waiting[0];prefix,source=self._initial_prefix(r);r.computed=prefix
            count=min(len(r.prompt_ids)-r.computed,budget)
            count,new_budget,new_encoder=self._encoder_schedule(r,count,encoder_budget)
            if count<=0:r.computed=0;break
            self.waiting.popleft();r.admitted=True;r.prefix_count=prefix;r.prefix_source=source;self.running.append(r)
            self.current.append({'id':r.request_id,'index':r.input_index,'start':prefix,'count':count,'prefix_source':source,'prefix_count':prefix})
            if new_encoder:self.new_encoder.append(r.request_id)
            encoder_budget=new_budget;budget-=count;r.computed+=count;self._cache_blocks(r,r.computed)
        # Native async scheduling makes this decision before consuming the
        # preceding GPU output. Only afterwards are its encoder references
        # released, using the already-advanced computed-token counters.
        for rid in self.previous:
            r=self.requests[rid]
            if r.image_offset+r.image_length<=r.computed:
                for image_id in list(self.encoder.get_cached_input_ids(r)):self.encoder.free_encoder_input(r,image_id)
        self.freed_encoder=self.encoder.get_freed_mm_hashes()
        self.previous=[x['id'] for x in self.current]
        self.current.sort(key=lambda x:x['count'])
        assert self.current or (not self.running and not self.waiting),'Original scheduler made no progress'
        return {'ordinal':self.wave-1,'chunks':[dict(x) for x in self.current],'new_encoder_request_ids':list(self.new_encoder),'freed_encoder_hashes':list(self.freed_encoder),'retired_request_ids':list(self.retired),'tokens':sum(x['count'] for x in self.current)}
    def consume_argmax(self,values):
        assert set(values)=={c['id'] for c in self.current};actions=[]
        for chunk in self.current:
            r=self.requests[chunk['id']];token=int(values[r.request_id]);assert 0<=token<100352
            if r.finish_reason is not None:
                actions.append({'id':r.request_id,'kind':'unused_original_async_lookahead','argmax':token});continue
            if r.computed<len(r.prompt_ids):
                actions.append({'id':r.request_id,'kind':'unused_partial_prefill_argmax','argmax':token});continue
            r.output_ids.append(token)
            if token==100257:r.finish_reason='stop'
            elif len(r.output_ids)>=2048 or len(r.prompt_ids)+len(r.output_ids)>=8192:r.finish_reason='length'
            if r.finish_reason:
                r.finish_wave=self.wave
                # The original scheduler suppresses a provably unnecessary
                # extra forward at the hard budget/model-context boundary.
                hard_limit=len(r.output_ids)>=2048 or len(r.prompt_ids)+len(r.output_ids)>=8192
                r.retire_wave=self.wave+(1 if hard_limit else 2)
            actions.append({'id':r.request_id,'kind':'generated','token':token,'output_position':len(r.output_ids)-1,'finish_reason':r.finish_reason})
        return actions
    def done(self):return not self.waiting and all(r.finish_reason is not None and r.retire_wave<=self.wave for r in self.running)
    def end_batch(self):
        assert all(r.finish_reason is not None for r in self.requests.values()) and not self.waiting
        for r in self.running:self.encoder.free(r)
        self.requests={};self.running=[];self.previous=[]

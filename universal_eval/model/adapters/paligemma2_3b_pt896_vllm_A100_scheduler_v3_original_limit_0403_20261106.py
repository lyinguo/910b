"""Qualified FCFS fixed200 scope of the original vLLM asynchronous scheduler.

This is a logical execution reference, not a CUDA/paged-cache accelerator.
The original model has unique image keys in all200 requests. Cache capacities,
the two in-flight stages and partial-prefill behavior remain explicit. A new
duplicate scope must be qualified rather than silently disabling cache reuse.
"""
class OriginalPaliScheduler:
    def __init__(self,prompt_lengths,max_tokens=1024):
        assert prompt_lengths and len(prompt_lengths)<=10
        assert all(n==4111 for n in prompt_lengths) and max_tokens==1024
        self.requests=[dict(id=str(i),prompt=n,past=0,generated=0,ended=False,
                            finish=None,finished=False,encoder_reserved=False)
                       for i,n in enumerate(prompt_lengths)]
        self.waiting=list(self.requests);self.running=[];self.ordinal=0
        self.used_encoder_tokens=0;self.encoder_free_events=[]
        self.pending=None;self.previous_plan_request_ids=set()

    def plan(self):
        assert self.pending is None
        budget=8192;encoder_budget=8192;planned=[];encode=[]
        for r in self.running:
            assert not r['finished']
            q=min(r['prompt']-r['past'] if r['past']<r['prompt'] else 1,budget)
            if not q:continue
            planned.append(dict(id=r['id'],past=r['past'],query_length=q,
                                prefill=r['past']<r['prompt']))
            budget-=q
        while self.waiting and budget:
            r=self.waiting[0]
            assert r['past']==0 and not r['encoder_reserved']
            if self.used_encoder_tokens+4096>8192 or encoder_budget<4096:break
            q=min(r['prompt'],budget)
            # Exact original _try_schedule_encoder_inputs checks past<offset.
            # At offset0/past0 that is False, allowing an actual partial image
            # prefill, as independently observed on genuine A100 batches3/10.
            self.used_encoder_tokens+=4096;encoder_budget-=4096
            r['encoder_reserved']=True;self.waiting.pop(0);self.running.append(r)
            encode.append(r['id']);planned.append(dict(id=r['id'],past=0,
                query_length=q,prefill=True));budget-=q
        assert planned and sum(p['query_length'] for p in planned)<=8192
        planned.sort(key=lambda p:(p['query_length']!=1,p['query_length']))
        self.pending={'ordinal':self.ordinal,'requests':planned,
                      'encoder_requests':encode,'encoder_reserved_tokens':self.used_encoder_tokens}
        return self.pending

    def advance(self,selected_token_ids):
        """Use fresh actual head argmax only; no gold output is stored here."""
        p=self.pending;assert p is not None and len(selected_token_ids)==len(p['requests'])
        actions=[];released=[]
        for scheduled,token in zip(p['requests'],selected_token_ids):
            r=self.requests[int(scheduled['id'])]
            assert r['past']==scheduled['past'];r['past']+=scheduled['query_length']
            if r['past']<r['prompt']:
                actions.append(dict(request_id=r['id'],kind='discard_partial_prefill_head',actual_argmax=token))
            elif r['ended']:
                actions.append(dict(request_id=r['id'],kind='original_unused_async_lookahead',actual_argmax=token))
                r['finished']=True;released.append(r['id'])
            else:
                r['generated']+=1
                at_limit=r['generated']>=1024 or r['past']+1>=8192
                if token==1:r.update(ended=True,finish='stop')
                elif at_limit:r.update(ended=True,finish='length')
                # Original Scheduler.schedule suppresses another async plan
                # at max_tokens irrespective of an EOS on that last sample.
                # Keep stop-before-length classification, but no extra forward.
                if r['ended'] and at_limit:r['finished']=True;released.append(r['id'])
                actions.append(dict(request_id=r['id'],kind='reported_sample',token_id=token,
                    generation_count=r['generated'],finished=r['ended'],finish_reason=r['finish']))
        # vLLM schedules the next wave before processing the previous output.
        # update_from_output->_free_encoder_inputs then sees computed counts
        # already advanced by the new plan. Preserve that two-stage boundary.
        freed=[]
        if self.ordinal>=1:
            for r in self.running:
                if r['id'] in self.previous_plan_request_ids and r['encoder_reserved'] and (r['past']>=4097 or r['finished']):
                    r['encoder_reserved']=False;self.used_encoder_tokens-=4096;freed.append(r['id'])
        for rid in released:
            r=self.requests[int(rid)]
            if r['encoder_reserved']:
                r['encoder_reserved']=False;self.used_encoder_tokens-=4096;freed.append(rid)
            self.running.remove(r)
        assert 0<=self.used_encoder_tokens<=8192
        self.encoder_free_events.append(dict(ordinal=self.ordinal,requests=freed))
        self.previous_plan_request_ids={r['id'] for r in p['requests']}
        self.pending=None;self.ordinal+=1
        return actions,released,freed

    @property
    def complete(self):return not self.running and not self.waiting
